"""
Perturbational Complexity Index probe (read-only).

Implements the TMS/EEG analogue from Koch, Massimini, Boly & Tononi (2016): perturb
the system, record the CAUSAL response, and score that response for integration and
differentiation at once. The measure itself lives in
models/evaluation/perturbational_complexity.py; this script supplies the two
replays it needs. Background and the alignment audit: docs/thalamic_gating_evidence.md.

WHY THIS EXISTS. Three implemented signature instruments have each been found
degenerate on this agent because they read the SPONTANEOUS trajectory, which is
frozen: ei_gates was bit-identical at the constant-trajectory Laplace floor, CE 2.0
was reproduced exactly by a frozen input, and phi sits near zero. A perturbational
measure supplies its own variation and does not inherit that failure mode.

HOW THE CAUSAL RESPONSE IS OBTAINED. Two rollouts from the same seed with the same
fixed action stream:

    rollout A   clean
    rollout B   identical, except one impulse injected at --perturb-step

Determinism makes them bit-identical up to the impulse, so everything after it is
the causal effect of the impulse and nothing else. This replaces snapshot/restore
of RSSM buffers, workspace reverberation, binding phases and gate state, which
would be the same computation with more ways to be silently wrong.

THE ACTION STREAM IS FIXED ON PURPOSE. Both rollouts take the same pre-generated
actions rather than acting on their own policy. If the policy were allowed to
respond, the perturbation would change the observations and the measurement would
be of behaviour, not of the internal causal response. The subject does not act
during a TMS/EEG PCI measurement either.

WHERE THE IMPULSE GOES. Default is the RSSM recurrent state (`--perturb-site rssm`),
because it is the only genuinely recurrent carrier in the perception chain, so the
impulse propagates across steps. The `broadcast` and `gate` sites are available but
carry a caveat printed at runtime: this harness recomputes both from the tectum on
every step, so an impulse there cannot propagate and its "response" lasts exactly
one step by construction. A near-zero PCI from those sites is a property of the
harness, not a finding about the agent.

WHERE THE RESPONSE IS READ. Three sites, every run:
  rssm       the recurrent state itself, pooled to one channel per feature map.
             CONTROL, not a result. It answers "did the impulse propagate at all?",
             so a zero at the downstream sites can be attributed to the architecture
             rather than to a broken harness. If PCI is zero HERE, the measurement
             failed and nothing downstream may be interpreted.
  gate       the 5 ConsciousnessGate nodes. Primary, because phi / EI / CE 2.0 are
             computed on this same substrate, so PCI is directly comparable to the
             instruments it is meant to supplement.
  broadcast  the workspace broadcast vector. Exploratory, because the gate's micro
             level is documented frozen (4 of 5 nodes inside a ~0.01 band,
             `adaptation` at std 6.08e-06; docs/results/gate_binning_2026_07.md).
             A gate-only PCI near zero could not be told apart from a dead
             substrate. The broadcast is where the 2026-07 signature ablation found
             a response that replicated across 3 seeds.

The rssm control is what makes a zero readable. In the default configuration
(discrete latent, `final` capsule projection) an impulse that moves h_state by ~18
units for ~8 steps changes tectum_content by ~4e-09, i.e. float32 noise. The
perturbation propagates and then dies at the capsule stage. That is the same locus
the collapse probes localized independently, now visible causally rather than by
decoding.

HONESTY. Single seed is a hypothesis. Nothing from this probe may be reported as a
result without at least 3-seed replication. The published PCI scale does not
transfer to this system; the cutoff near 0.31 in the human literature is meaningless
here and must never be quoted against these numbers. See the module docstring of
models/evaluation/perturbational_complexity.py for the normalization deviation.

Usage:
    python -m scripts.analysis.probe_pci --env dmts --seed 42
    python -m scripts.analysis.probe_pci --env dmts --load-tectum runs/x/tectum.pt \\
        --trials 10 --magnitude 1.0
"""
from __future__ import annotations

import argparse
import csv
import os
import warnings
from pathlib import Path

import numpy as np
import torch

from models.evaluation.perturbational_complexity import (
    DEFAULT_NOISE_EPS,
    DEFAULT_RELATIVE_FRAC,
    DEFAULT_VAR_FLOOR,
    compute_pci,
)
from scripts.analysis.probe_perception_decodability import (
    _build_components,
    _compute_broadcast,
)
from scripts.training.train_rlhf import frame_to_tensor
from simulations.environments.dmts_env import DMTSEnv
from simulations.environments.simple_visual_env import SimpleVisualEnv

PERTURB_SITES = ("rssm", "broadcast", "gate")
# Order matters for the printed summary: the control is read first so a zero at the
# downstream sites is interpretable on sight.
READ_SITES = ("rssm", "gate", "broadcast")


def _make_env(env_name: str, seed: int):
    if env_name == "dmts":
        env = DMTSEnv(num_trials=20)
    else:
        env = SimpleVisualEnv(width=224, height=224)
    env.reset(seed=seed)
    return env


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# Field order matches the gate_* columns in scripts/training/metrics_logger.py, so
# a channel index here means the same node it means there.
GATE_FIELDS = (
    "attention_level",
    "stability_score",
    "adaptation_rate",
    "meta_memory_coherence",
    "narrator_confidence",
)


def _gate_vector(gate) -> np.ndarray:
    """The 5 ConsciousnessGate node values as a flat vector, in logger order."""
    state = getattr(gate, "state", None)
    if state is None:
        return np.zeros(len(GATE_FIELDS), dtype=np.float64)
    return np.array(
        [float(getattr(state, name, 0.0)) for name in GATE_FIELDS], dtype=np.float64
    )


def _rollout(
    env_name: str,
    seed: int,
    n_steps: int,
    actions: np.ndarray,
    perturb_step: int | None,
    perturb_site: str,
    magnitude: float,
    impulse_seed: int,
    load_tectum: str | None,
    latent_mode: str,
    capsule_source: str,
):
    """
    One rollout. Returns (rssm_trace, gate_trace, broadcast_trace), each
    [n_channels, n_steps].

    `perturb_step=None` gives the clean rollout. Both rollouts must draw the same
    number of random numbers so that the RNG streams stay in lockstep after the
    impulse; the impulse itself uses its own generator for exactly that reason.
    """
    _seed_everything(seed)
    action_dim = 5 if env_name == "dmts" else 2
    config, tectum, workspace, reentrant, self_model, memory, mock_sem = _build_components(
        env_name,
        action_dim=action_dim,
        seed=seed,
        mock_semantic=False,
        load_tectum=load_tectum,
        latent_mode=latent_mode,
        capsule_workspace_source=capsule_source,
    )
    gate, gate_loaded = _build_gate(config, load_tectum)
    gate.eval()
    if load_tectum and not gate_loaded:
        print("  WARNING: gate is randomly initialised (no sibling gate checkpoint); "
              "gate-level readings below are not from the trained system")

    env = _make_env(env_name, seed)
    obs, _ = env.reset(seed=seed)

    impulse_rng = np.random.default_rng(impulse_seed)
    device = config["device"]
    ws_dim = config["workspace_dim"]

    rssm_trace: list[np.ndarray] = []
    gate_trace: list[np.ndarray] = []
    broadcast_trace: list[np.ndarray] = []

    with torch.no_grad():
        for step in range(n_steps):
            frame = frame_to_tensor(obs, device)
            audio = torch.zeros(1, config["tectum_feature_dim"], 2, device=device)
            tectum_content, vision_bid = tectum(frame, audio)

            # rssm impulse: applied to the recurrent carrier AFTER the forward, so
            # it enters the next step's recurrence and genuinely propagates.
            if (
                perturb_step is not None
                and step == perturb_step
                and perturb_site == "rssm"
                and getattr(tectum, "h_state", None) is not None
            ):
                tectum.h_state = tectum.h_state + _impulse_like(
                    tectum.h_state, magnitude, impulse_rng, device
                )

            broadcast = _compute_broadcast(
                config, tectum, workspace, reentrant, self_model, memory,
                mock_sem, tectum_content, vision_bid, obs,
            )

            if (
                perturb_step is not None
                and step == perturb_step
                and perturb_site == "broadcast"
            ):
                broadcast = broadcast + magnitude * _unit_noise(
                    broadcast.shape, impulse_rng
                )

            gate_in = torch.tensor(
                broadcast[:ws_dim], dtype=torch.float32, device=device
            )
            if gate_in.numel() < gate.hidden_size:
                gate_in = torch.nn.functional.pad(
                    gate_in, (0, gate.hidden_size - gate_in.numel())
                )
            gate_in = gate_in[: gate.hidden_size].unsqueeze(0)

            if (
                perturb_step is not None
                and step == perturb_step
                and perturb_site == "gate"
            ):
                gate_in = gate_in + magnitude * torch.tensor(
                    _unit_noise(tuple(gate_in.shape), impulse_rng),
                    dtype=torch.float32,
                    device=device,
                )

            gate(gate_in)

            rssm_trace.append(_rssm_vector(tectum))
            gate_trace.append(_gate_vector(gate))
            broadcast_trace.append(np.asarray(broadcast, dtype=np.float64).ravel())

            action = int(actions[step]) % action_dim
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=seed + 1)

    return (
        np.array(rssm_trace).T,
        np.array(gate_trace).T,
        np.array(broadcast_trace).T,
    )


def _rssm_vector(tectum) -> np.ndarray:
    """
    The RSSM recurrent state pooled to one channel per feature map.

    h_state is [B, feature_dim, grid, grid], which is far more channels than a PCI
    montage should have, so it is spatially averaged to feature_dim channels. This
    is the control read site: it establishes that the impulse propagated.
    """
    h = getattr(tectum, "h_state", None)
    if h is None:
        return np.zeros(1, dtype=np.float64)
    pooled = h.detach().float().mean(dim=(-2, -1)).reshape(-1)
    return pooled.cpu().numpy().astype(np.float64)


def _build_gate(config, load_tectum: str | None = None):
    """
    Construct the ConsciousnessGate with the same arguments init_components uses
    (train_rlhf.py), and load its trained weights when a checkpoint is available.

    Weight loading was added 2026-08-11. Before that this function returned a
    randomly initialised gate while its docstring claimed it read the substrate the
    training loop reads. It did not: the gate trains inside tectum_optimizer but was
    never written to disk, so every offline gate reading scored a random
    `gate_feedback`. Checkpoints written before that date have no sibling gate file,
    and for those the gate is still random. That is reported, not hidden: callers get
    `loaded=False` and must say so in any result.
    """
    from models.core.consciousness_gating import (
        ConsciousnessGate, gate_checkpoint_path,
    )

    gate = ConsciousnessGate(
        {
            "hidden_size": config["workspace_dim"],
            "ablate_feedback": config.get("ablate_gate_feedback", False),
            "use_self_vector": config.get("enable_self_vector_gating", False),
            "self_vector_dim": config.get("self_vector_dim", 64),
        }
    )

    loaded = False
    if load_tectum:
        gate_path = gate_checkpoint_path(load_tectum)
        if os.path.exists(gate_path):
            gate.load_state_dict(torch.load(gate_path, map_location="cpu"))
            loaded = True
        else:
            warnings.warn(
                f"No gate checkpoint at {gate_path}, so the gate is RANDOMLY "
                f"INITIALISED while the tectum is trained. Gate-level readings from "
                f"this run measure an untrained gate_feedback and must not be "
                f"compared against training-time gate metrics. Re-run training to "
                f"write one.",
                RuntimeWarning,
            )
    return gate, loaded


def _unit_noise(shape, rng: np.random.Generator) -> np.ndarray:
    """Random direction of unit norm, so `magnitude` is the impulse size."""
    vec = rng.normal(size=shape)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _impulse_like(tensor: torch.Tensor, magnitude: float, rng, device) -> torch.Tensor:
    noise = _unit_noise(tuple(tensor.shape), rng)
    return magnitude * torch.tensor(noise, dtype=tensor.dtype, device=device)


def run_trial(args, trial_index: int) -> list[dict]:
    """One perturbation trial: clean rollout, perturbed rollout, PCI at both sites."""
    seed = args.seed + trial_index
    n_steps = args.perturb_step + args.response_window
    rng = np.random.default_rng(seed)
    action_dim = 5 if args.env == "dmts" else 2
    # A fixed stream of no-response actions with occasional movement, identical in
    # both rollouts. Action 0 is "wait" in DMTS, the correct non-response.
    actions = np.where(rng.random(n_steps) < args.action_rate,
                       rng.integers(0, action_dim, size=n_steps), 0)

    common = dict(
        env_name=args.env,
        seed=seed,
        n_steps=n_steps,
        actions=actions,
        perturb_site=args.perturb_site,
        magnitude=args.magnitude,
        impulse_seed=seed + 7919,
        load_tectum=args.load_tectum,
        latent_mode=args.latent_mode,
        capsule_source=args.capsule_workspace_source,
    )

    rssm_clean, gate_clean, bc_clean = _rollout(perturb_step=None, **common)
    rssm_pert, gate_pert, bc_pert = _rollout(perturb_step=args.perturb_step, **common)

    # Per-site weight provenance, recorded in every row.
    #
    # The probe prints a warning when the gate falls back to random init, but that
    # warning lives only in the console. On 2026-09-12 a four-checkpoint study read
    # the CSVs, missed the warning, and reported random-gate readings as a finding
    # about the trained system. The CSV had 15 columns and none of them could have
    # caught it. These two can.
    #
    # The mapping follows the data flow in _rollout: the broadcast is computed from
    # the tectum and workspace and is then fed INTO the gate, so gate weights cannot
    # reach the rssm or broadcast traces.
    from models.core.consciousness_gating import gate_checkpoint_path as _gate_ckpt
    tectum_trained = bool(args.load_tectum)
    gate_trained = tectum_trained and os.path.isfile(_gate_ckpt(args.load_tectum))
    provenance = {
        "rssm": tectum_trained,
        "broadcast": tectum_trained,
        "gate": gate_trained,
    }

    rows = []
    for site, clean, pert in (
        ("rssm", rssm_clean, rssm_pert),
        ("gate", gate_clean, gate_pert),
        ("broadcast", bc_clean, bc_pert),
    ):
        pre = clean[:, : args.perturb_step]
        post_clean = clean[:, args.perturb_step:]
        post_pert = pert[:, args.perturb_step:]
        response = post_pert - post_clean

        # Sanity check the determinism the whole method rests on: the two rollouts
        # must be identical BEFORE the impulse. If they are not, the measurement is
        # meaningless and should fail loudly rather than produce a number.
        pre_divergence = float(np.abs(pert[:, : args.perturb_step] - pre).max())

        result = compute_pci(
            response, pre, threshold_sigma=args.threshold_sigma,
            var_floor=args.var_floor,
            var_floor_mode=args.var_floor_mode,
            relative_frac=args.relative_frac,
            noise_eps=args.noise_eps,
        )
        rows.append(
            {
                "trial": trial_index,
                "seed": seed,
                "read_site": site,
                # "trained" or "RANDOM". A RANDOM row says nothing about the
                # trained system and must not be reported as if it did.
                "weights": "trained" if provenance[site] else "RANDOM",
                "perturb_site": args.perturb_site,
                "magnitude": args.magnitude,
                # The floor that decided which channels could register at all.
                # A zero PCI is unreadable without it. Under relative mode this is
                # derived per site, so record what was USED, not what was asked for.
                "var_floor": f"{result.resolved_var_floor:.6e}",
                "var_floor_mode": args.var_floor_mode,
                "pci": round(result.pci, 6),
                "pci_casali": round(result.pci_casali, 6),
                "lz_complexity": result.lz_complexity,
                "active_fraction": round(result.active_fraction, 6),
                "source_entropy": round(result.source_entropy, 6),
                "n_channels": result.n_channels,
                "n_timesteps": result.n_timesteps,
                # Raw response size and baseline scale, so a PCI of 0.0 can be
                # attributed: no response at all, or a response too small to clear
                # the channel's own fluctuation.
                "max_abs_response": f"{float(np.abs(response).max()):.3e}",
                "median_baseline_sd": f"{float(np.median(pre.std(axis=1, ddof=1))):.3e}",
                "pre_impulse_divergence": f"{pre_divergence:.3e}",
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default="dmts", choices=["dmts", "dark_room"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=5,
                        help="Perturbation trials; each is a clean + perturbed rollout pair")
    parser.add_argument("--perturb-step", type=int, default=40,
                        help="Step at which the impulse is injected. Steps before it "
                             "are the baseline window.")
    parser.add_argument("--response-window", type=int, default=60,
                        help="Steps of causal response recorded after the impulse")
    parser.add_argument("--perturb-site", default="rssm", choices=PERTURB_SITES)
    parser.add_argument("--magnitude", type=float, default=1.0,
                        help="L2 norm of the injected impulse")
    parser.add_argument("--var-floor", type=float, default=DEFAULT_VAR_FLOOR,
                        help="Channels whose baseline std is at or below this are "
                             "treated as dead and can never mark significant. The "
                             "default 1e-4 is ABOVE the gate substrate's own "
                             "fluctuation on most checkpoints (gate nodes sit at "
                             "4.1e-05 to 6.9e-05), so at the default the gate "
                             "cannot register a response no matter how large it is. "
                             "Lower it to measure the gate; state the value used.")
    parser.add_argument("--var-floor-mode", default="absolute",
                        choices=["absolute", "relative"],
                        help="absolute (default) uses --var-floor as given, which is "
                             "the historical behaviour and keeps every published "
                             "number bit-identical. relative derives the floor from "
                             "each site's OWN channels, so one setting can serve "
                             "sites whose scales differ by orders of magnitude.")
    parser.add_argument("--relative-frac", type=float, default=DEFAULT_RELATIVE_FRAC,
                        help="relative mode: a channel is dead below this fraction "
                             "of its site's median baseline std.")
    parser.add_argument("--noise-eps", type=float, default=DEFAULT_NOISE_EPS,
                        help="relative mode: absolute guard so a site that is "
                             "entirely float noise does not scale its own floor down "
                             "and admit that noise.")
    parser.add_argument("--threshold-sigma", type=float, default=3.0,
                        help="Significance threshold in baseline standard deviations")
    parser.add_argument("--action-rate", type=float, default=0.0,
                        help="Fraction of steps taking a non-zero action. 0.0 keeps "
                             "the agent still, which is the TMS/EEG analogue.")
    parser.add_argument("--load-tectum", default=None)
    parser.add_argument("--latent-mode", default="discrete",
                        choices=["discrete", "continuous"])
    parser.add_argument("--capsule-workspace-source", default="final",
                        choices=["final", "all_levels"])
    parser.add_argument("--out", default=None, help="Optional CSV path")
    args = parser.parse_args()

    if args.perturb_site in ("broadcast", "gate"):
        print(
            "WARNING: this harness recomputes the broadcast and the gate from the\n"
            "tectum on every step, so an impulse at either site cannot propagate\n"
            "across steps. Its response lasts exactly one step BY CONSTRUCTION, and\n"
            "a near-zero PCI here is a property of the harness, not of the agent.\n"
            "Use --perturb-site rssm for a propagating perturbation.\n"
        )

    print(f"PCI probe: env={args.env} perturb_site={args.perturb_site} "
          f"magnitude={args.magnitude} trials={args.trials}")
    print(f"  baseline window: steps 0..{args.perturb_step}")
    print(f"  response window: steps {args.perturb_step}.."
          f"{args.perturb_step + args.response_window}")
    print()

    all_rows = []
    for trial in range(args.trials):
        rows = run_trial(args, trial)
        all_rows.extend(rows)
        for row in rows:
            print(f"  trial {row['trial']:>2}  {row['read_site']:<10} "
                  f"pci={row['pci']:.4f}  active={row['active_fraction']:.4f}  "
                  f"|resp|max={row['max_abs_response']}  "
                  f"base_sd={row['median_baseline_sd']}")

    print()
    labels = {"rssm": "CONTROL", "gate": "PRIMARY", "broadcast": "exploratory"}
    summary = {}
    divergences = {}
    for site in READ_SITES:
        vals = [r["pci"] for r in all_rows if r["read_site"] == site]
        active = [r["active_fraction"] for r in all_rows if r["read_site"] == site]
        if not vals:
            continue
        casali = [r["pci_casali"] for r in all_rows if r["read_site"] == site]
        summary[site] = float(np.mean(vals))
        prov = {r["weights"] for r in all_rows if r["read_site"] == site}
        tag = "trained" if prov == {"trained"} else "RANDOM WEIGHTS, NOT A RESULT"
        print(f"{site:<10} ({labels[site]:<11}) [{tag}] pci mean={np.mean(vals):.4f} "
              f"sd={np.std(vals):.4f} min={min(vals):.4f} max={max(vals):.4f} "
              f"| casali mean={np.mean(casali):.4f} "
              f"| active_fraction mean={np.mean(active):.4f}")
        divergences[site] = (float(np.mean(vals)), float(np.mean(casali)),
                             float(np.mean(active)))

    max_pre = max(float(r["pre_impulse_divergence"]) for r in all_rows)
    print()
    if max_pre > 1e-6:
        print(f"FAILED: the clean and perturbed rollouts diverged BEFORE the "
              f"impulse (max {max_pre:.3e}). The rollouts are not deterministic, so "
              f"the responses above are not causal effects and must not be reported.")
        return
    print(f"Determinism check passed: pre-impulse divergence {max_pre:.3e}.")

    # Refuse to let an untrained site be read as a finding. This is the check that
    # the 2026-09-12 four-checkpoint study needed and did not have.
    random_sites = sorted({r["read_site"] for r in all_rows if r["weights"] == "RANDOM"})
    if random_sites:
        print("")
        print(f"DISQUALIFIED: {', '.join(random_sites)} ran on RANDOMLY INITIALISED "
              f"weights, so those readings are not about the trained system and must "
              f"not be reported as results. Only sites marked [trained] above may be "
              f"cited. To fix, use a checkpoint whose sibling gate file exists; gate "
              f"weights were not saved before 2026-08-11.")

    # The module docstring says `pci_casali` diverges for sparse responses and that
    # conditions must not be ranked by it. It was reported only in the CSV, so the
    # sparse regime was visible only to whoever opened the file. On 2026-09-12 a
    # multi-checkpoint study came close to reporting that documented effect as a new
    # finding for exactly that reason. The warning below states it at the point of
    # reading instead.
    for site, (pci_mean, casali_mean, active_mean) in divergences.items():
        if pci_mean <= 0.0:
            continue
        ratio = casali_mean / pci_mean
        if ratio >= 2.0:
            print("")
            print(f"WARNING: at {site}, pci_casali ({casali_mean:.4f}) is "
                  f"{ratio:.1f}x pci ({pci_mean:.4f}), with active_fraction "
                  f"{active_mean:.4f}. The Casali normalization divides by the "
                  f"observed activity level, so it inflates on a sparse response. "
                  f"Cite pci. Do not rank conditions by pci_casali, and do not "
                  f"quote either against the published human 0.31 cutoff.")

    # Attribute any zero before anyone reads the table.
    control = summary.get("rssm", 0.0)
    if control <= 0.0:
        print("\nFAILED: the control site shows no causal response, so the impulse "
              "did not propagate even in the recurrent state. This is a harness or "
              "configuration failure, NOT a finding about the agent. Raise "
              "--magnitude or check --perturb-site before interpreting anything.")
    else:
        dead = [s for s in ("gate", "broadcast") if summary.get(s, 0.0) <= 0.0]
        if dead:
            print(f"\nThe impulse propagated at the control site (pci={control:.4f}) "
                  f"but produced NO measurable causal response at: {', '.join(dead)}. "
                  f"The perturbation dies between the recurrent state and those "
                  f"sites. Check --capsule-workspace-source and --latent-mode: the "
                  f"default (final / discrete) is the configuration the collapse "
                  f"probes already found lossy.")

    print("\nSingle seed is a hypothesis. Replicate across >= 3 seeds before "
          "reporting any of these numbers.")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
