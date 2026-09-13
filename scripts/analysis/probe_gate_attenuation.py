"""
Where does the impulse lose its size between `h_state` and the gate?

PCI reads the gate at 0.0000 while its control reads 0.16. Something between the
perturbed recurrent state and the gate's five nodes removes the impulse. This probe
finds WHERE by instrumenting every stage on the path and reporting, per stage, the
maximum absolute clean-vs-perturbed difference in the response window.

A stage whose difference grows 100x when the impulse magnitude grows 100x passes the
impulse through. The first stage whose difference grows less than that is where the
signal is lost.

Stages, in order, matching the forward in `probe_pci._rollout`:

  h_state        the perturbed carrier itself. Control: must scale with magnitude.
  tectum_content the tectum's output vector
  vision_bid     the scalar bid, `tanh(kl_div)` at sensory_tectum.py:456
  broadcast      what `reentrant.settle` returns
  gate_in        broadcast truncated or padded to the gate's hidden_size
  enriched       gate_in plus the temporal feedback projection
  logits         the 4 pre-sigmoid linear outputs inside the gate
  gate_state     the 5 post-sigmoid node values, which is what PCI reads

Two cautions, both of which changed a conclusion once already:

1. Gate node values are float32 near 0.485, where one unit in the last place is
   2.98e-08. A response of 5.96e-08 is 2 ulp, that is, numerical resolution and not
   a measurement. Check `max_response` against that scale before reading any growth
   factor computed from it.
2. The baseline sd at the gate moves by 84x across PROBE SEEDS on one checkpoint,
   while the response moves by 2x. A ratio summarised across probe seeds hides this.
   Report per seed.

Verdict: `docs/results/pci_gate_attenuation_2026_09.md`.

Read-only. No training. One clean rollout plus one rollout per magnitude, about
17 seconds for three magnitudes.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.analysis.probe_pci import (  # noqa: E402
    _build_gate,
    _gate_vector,
    _impulse_like,
    _make_env,
    _rssm_vector,
    _seed_everything,
)
from scripts.analysis.probe_perception_decodability import (  # noqa: E402
    _build_components,
    _compute_broadcast,
)
from scripts.training.train_rlhf import frame_to_tensor  # noqa: E402

STAGES = (
    "h_state",
    "tectum_content",
    "vision_bid",
    "broadcast",
    "gate_in",
    "enriched",
    "logits",
    "gate_state",
)

LOGIT_NETS = ("attention_net", "coherence_net", "stability_net", "confidence_net")


def _flat(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().float().reshape(-1).cpu().numpy().astype(np.float64)
    return np.asarray(t, dtype=np.float64).reshape(-1)


def _rollout(env_name, seed, n_steps, actions, perturb_step, magnitude,
             impulse_seed, load_tectum, latent_mode, capsule_source):
    """One rollout, returning {stage: array [n_channels, n_steps]}."""
    _seed_everything(seed)
    action_dim = 5 if env_name == "dmts" else 2
    config, tectum, workspace, reentrant, self_model, memory, mock_sem = _build_components(
        env_name, action_dim=action_dim, seed=seed, mock_semantic=False,
        load_tectum=load_tectum, latent_mode=latent_mode,
        capsule_workspace_source=capsule_source,
    )
    gate, gate_loaded = _build_gate(config, load_tectum)
    gate.eval()
    if load_tectum and not gate_loaded:
        print("  WARNING: gate is randomly initialised (no sibling gate checkpoint); "
              "gate-level readings below are not from the trained system")

    # Tap the pre-sigmoid linear output of each gate net. Each net is
    # Sequential(Linear, GELU, Linear, Sigmoid), so index 2 is the final Linear.
    captured: dict[str, np.ndarray] = {}

    def _mk_hook(name):
        def hook(_module, _inp, out):
            captured[name] = _flat(out)
        return hook

    handles = [getattr(gate, n)[2].register_forward_hook(_mk_hook(n))
               for n in LOGIT_NETS]

    enriched_capture: dict[str, np.ndarray] = {}

    def _enriched_hook(_module, inp, _out):
        # attention_net's input is cat([enriched, prev_conf]); drop the last entry.
        enriched_capture["v"] = _flat(inp[0])[:-1]

    handles.append(gate.attention_net[0].register_forward_hook(_enriched_hook))

    env = _make_env(env_name, seed)
    obs, _ = env.reset(seed=seed)
    impulse_rng = np.random.default_rng(impulse_seed)
    device = config["device"]
    ws_dim = config["workspace_dim"]

    traces: dict[str, list[np.ndarray]] = {s: [] for s in STAGES}

    with torch.no_grad():
        for step in range(n_steps):
            frame = frame_to_tensor(obs, device)
            audio = torch.zeros(1, config["tectum_feature_dim"], 2, device=device)
            tectum_content, vision_bid = tectum(frame, audio)

            if (perturb_step is not None and step == perturb_step
                    and getattr(tectum, "h_state", None) is not None):
                tectum.h_state = tectum.h_state + _impulse_like(
                    tectum.h_state, magnitude, impulse_rng, device
                )

            broadcast = _compute_broadcast(
                config, tectum, workspace, reentrant, self_model, memory,
                mock_sem, tectum_content, vision_bid, obs,
            )
            if broadcast is None:
                broadcast = np.zeros(ws_dim, dtype=np.float64)

            gate_in = torch.tensor(broadcast[:ws_dim], dtype=torch.float32, device=device)
            if gate_in.numel() < gate.hidden_size:
                gate_in = torch.nn.functional.pad(
                    gate_in, (0, gate.hidden_size - gate_in.numel()))
            gate_in = gate_in[: gate.hidden_size].unsqueeze(0)

            gate(gate_in)

            traces["h_state"].append(_rssm_vector(tectum))
            traces["tectum_content"].append(_flat(tectum_content))
            traces["vision_bid"].append(np.array([float(vision_bid)]))
            traces["broadcast"].append(np.asarray(broadcast, dtype=np.float64).ravel())
            traces["gate_in"].append(_flat(gate_in))
            traces["enriched"].append(enriched_capture["v"])
            traces["logits"].append(np.concatenate([captured[n] for n in LOGIT_NETS]))
            traces["gate_state"].append(_gate_vector(gate))

            action = int(actions[step]) % action_dim
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=seed + 1)

    for h in handles:
        h.remove()
    return {s: np.array(v).T for s, v in traces.items()}


def _stage_stats(clean, pert, perturb_step):
    """Baseline sd from the pre-impulse window, response from the post window."""
    base = clean[:, :perturb_step]
    sd = np.std(base, axis=1, ddof=1)
    median_sd = float(np.median(sd))
    diff = np.abs(pert[:, perturb_step:] - clean[:, perturb_step:])
    resp = float(diff.max()) if diff.size else 0.0
    return median_sd, resp


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="dmts", choices=["dmts", "dark_room"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perturb-step", type=int, default=40)
    p.add_argument("--response-window", type=int, default=60)
    p.add_argument("--magnitudes", type=float, nargs="+",
                   default=[1000.0, 10000.0, 100000.0])
    p.add_argument("--load-tectum", default=None)
    p.add_argument("--latent-mode", default="discrete",
                   choices=["discrete", "continuous"])
    p.add_argument("--capsule-workspace-source", default="final",
                   choices=["final", "all_levels"])
    p.add_argument("--out", default=None, help="Optional CSV path")
    args = p.parse_args()

    n_steps = args.perturb_step + args.response_window
    actions = np.zeros(n_steps, dtype=int)

    common = dict(env_name=args.env, seed=args.seed, n_steps=n_steps, actions=actions,
                  # The same offset probe_pci uses, so the impulse DIRECTION
                  # matches that probe's trial 0 exactly and the two are comparable.
                  impulse_seed=args.seed + 7919,
                  load_tectum=args.load_tectum,
                  latent_mode=args.latent_mode,
                  capsule_source=args.capsule_workspace_source)

    print("Clean rollout, %d steps, seed %d" % (n_steps, args.seed))
    clean = _rollout(perturb_step=None, magnitude=0.0, **common)

    rows = []
    per_mag = {}
    for mag in args.magnitudes:
        print("Perturbed rollout, magnitude %g" % mag)
        pert = _rollout(perturb_step=args.perturb_step, magnitude=mag, **common)
        pre = float(np.abs(pert["h_state"][:, :args.perturb_step]
                           - clean["h_state"][:, :args.perturb_step]).max())
        print("  pre-impulse divergence at h_state: %.1e" % pre)
        stats = {}
        for s in STAGES:
            sd, resp = _stage_stats(clean[s], pert[s], args.perturb_step)
            stats[s] = (sd, resp)
            rows.append(dict(magnitude=mag, stage=s, median_baseline_sd=sd,
                             max_response=resp,
                             ratio=(resp / sd) if sd > 0 else float("nan"),
                             pre_impulse_divergence=pre))
        per_mag[mag] = stats

    mags = list(args.magnitudes)
    print("\nMax absolute clean-vs-perturbed difference, by stage and magnitude")
    header = "%-16s" % "stage" + "".join("%14s" % ("m=%g" % m) for m in mags)
    if len(mags) > 1:
        header += "%10s%10s" % ("growth", "expected")
    print(header)
    print("-" * len(header))
    for s in STAGES:
        line = "%-16s" % s + "".join("%14.4e" % per_mag[m][s][1] for m in mags)
        if len(mags) > 1:
            first, last = per_mag[mags[0]][s][1], per_mag[mags[-1]][s][1]
            grow = (last / first) if first > 0 else float("nan")
            line += "%10.2f%10.0f" % (grow, mags[-1] / mags[0])
        print(line)

    print("\nResponse divided by that stage's own baseline sd (PCI needs > 3)")
    header2 = "%-16s%14s" % ("stage", "baseline sd") + "".join(
        "%16s" % ("ratio m=%g" % m) for m in mags)
    print(header2)
    print("-" * len(header2))
    for s in STAGES:
        sd0 = per_mag[mags[0]][s][0]
        line = "%-16s%14.4e" % (s, sd0)
        for m in mags:
            sd_m, resp = per_mag[m][s]
            line += "%16.4f" % ((resp / sd_m) if sd_m > 0 else float("nan"))
        print(line)

    print("\nA stage whose growth matches expected passes the impulse through.")
    print("The first stage where growth falls far below expected is where it is lost.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
