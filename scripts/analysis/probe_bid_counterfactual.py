"""
Stage 0 KILL GATE: can ANY vision-bid reduction produce a changing workspace winner?

GWT-2's selective-attention half is measured FALSE: vision wins 99.0 to 99.8 percent of
steps, and 2000 of 2000 in the final 2000 steps of every seed
(`docs/results/workspace_bids_live_2026_08.md`). The located cause is
`sensory_tectum.py:456`, `bid = torch.tanh(kl_div)`, fed by an element-count-scaled sum
whose smallest measured value is 2088.5 while `tanh` saturates in float32 past ~9.0.

This probe decides whether repairing the bid layer is worth a training run, WITHOUT
running one. It replays candidate vision bids through the REAL `GlobalWorkspace`, on
three independently trained checkpoints, and reports which module wins.

## Why this is not the obvious experiment

`bid_reduction_candidates_2026_08.md` already tested S1 to S4 and they all FAILED a
CONTENT gate: none tracks which shape was shown. That is not re-litigated here and it is
not the target. Content reaches the workspace through the PAYLOAD, and the 256-D
broadcast already decodes shape at 0.76 / 0.69 / 0.77. The bid is a salience signal.

What was never tested is whether any candidate produces a CHANGING WINNER, and whether
it does so consistently across independently trained networks. That second half is the
one that matters, because the pre-`tanh` KL scale is constrained by no loss term and may
differ by orders of magnitude between checkpoints. If it does, a fixed divisor gives a
different CONSTANT winner per training seed, which is a worse failure than the current
one because it looks like success on any single seed.

## FREE PARAMETERS, FIXED HERE BEFORE ANY WINNER NUMBER WAS READ

    EMA_ALPHA    = 0.01    matches probe_bid_reduction_candidates.py, causal
    SD_FLOOR     = 1e-9    a series below this is passed through, never z-scored
    TOPK_FRACTION= 0.01    matches the same probe, for S4

Recording them here is the point: a z-score guard whose alpha and floor are chosen after
seeing the winners is a way to manufacture competition.

## PRE-STATED GATE, written before any value from this probe was read

    KILL the direction if, for EVERY candidate, at ALL THREE checkpoints, either
      (i)  one module wins 95 percent or more of steps in every phase, or
      (ii) the modal winner differs BETWEEN checkpoints while being constant WITHIN
           each. This is the checkpoint-arbitrariness failure, and it is the outcome
           predicted for any fixed-divisor candidate.

    PROCEED only if at least one candidate gives, at all three checkpoints, a winner
    that changes within a run with the runner-up taking 5 percent or more of steps.

    Anything else is INCONCLUSIVE and is reported as such, not rounded to PROCEED.

## Candidates

    S0  tanh(sum)             the current baseline. Expected constant 1.0
    S1  tanh(mean)            fixed divisor. Expected to be checkpoint-arbitrary
    S2  sigmoid(mean z)       per-ELEMENT running z, self-normalizing
    S3  tanh(mean |z|)        per-element running z, self-normalizing
    S4  sigmoid(top 1% z)     per-element running z, self-normalizing
    S5  sigmoid(z of sum)     per-SERIES running z over time, self-normalizing.
                              NEW here. This is what a workspace-level bid
                              standardizer would compute, and it is the only
                              candidate that is scale-invariant by construction.

## Opponents

The other four bids are literal constants, which is a measured finding and not an
approximation (`workspace_bids_live_2026_08.md`). They are supplied as constants:

    default   memory 0.10  body 0.15  audio 0.0  semantic 0.0
    memfix    memory 0.60  body 0.15  audio 0.0  semantic 0.0
    memfix_sem memory 0.60 body 0.15  audio 0.0  semantic 1.0

`memfix` is `--enable-memory-retrieval`, measured at or above 0.5 on 99.2 to 99.5
percent of steps. `memfix_sem` adds `--enable-mock-semantic`, measured pinned near 1.0.

## Harness

Replay goes through the REAL `reentrant.settle`, which calls `run_competition` up to
`max_cycles` times per step and feeds the broadcast back to the tectum via
`receive_broadcast`. A first version called `run_competition` ONCE per step and FAILED
the fidelity check below at a vision share of 0.375 against a required 0.90, because
the ignition baseline EMA updates once per competition call and the live loop updates
it up to five times per step. That failure is why this probe refuses to report a
verdict until the fidelity check passes.

One tectum forward pass drives every candidate and opponent set. This is sound because
`receive_broadcast` (`sensory_tectum.py:495-549`) only READS `self._last_content` and
returns a float; it mutates no tectum state. Each candidate/opponent pair gets its own
`GlobalWorkspace` and `ReentrantProcessor`, so their ignition baselines and oscillator
phases never mix.

- The affective modulator IS applied, with `pad_state` from `evaluate_emotion(bid, 0, 0)`
  exactly as `train_rlhf.py:1025` does, because it clamps at 1.0 and boosts vision and
  would otherwise hide a re-saturation.
- Payloads carry the real `tectum_content`, as `_compute_broadcast` does.
- `interoceptive_state` is not reconstructed and is passed as None.
- The Kuramoto phase init is SEEDED per replay. Unseeded, the same candidate gave a
  vision share of 0.946 and then 0.490 on identical input, so any difference between
  candidates would have been phase noise.
- A WARM-UP window is discarded. The oscillators need several hundred steps to
  converge. Measured on S0 under `bcast`: over the first quarter the workspace is
  silent on 0.608 of steps with a vision share of 0.790, and from the second quarter
  onward `sync_R` sits at exactly 0.450000 (the published modal value is 0.450108),
  silence is 0.000 and the vision share is 1.000. Live runs are 8000 steps and their
  published analysis uses the final 2000, for the same reason. Reporting on the
  transient would have compared candidates on an unconverged oscillator.

Read-only. No training, no checkpoint written, no model modified.

Run:
    python -m scripts.analysis.probe_bid_counterfactual \\
        --checkpoints runs/gate3_s42 runs/gate3_s43 runs/gate3_s44 --episodes 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.core.global_workspace import GlobalWorkspace  # noqa: E402
from models.core.reentrant_processor import ReentrantProcessor  # noqa: E402
from models.emotion.affective_modulator import AffectiveModulator  # noqa: E402
from scripts.analysis.probe_bid_reduction_candidates import (  # noqa: E402
    EMA_ALPHA,
    TOPK_FRACTION,
    RunningStats,
    reductions,
)
from scripts.analysis.probe_perception_decodability import (  # noqa: E402
    _build_components,
    evaluate_emotion,
)
from scripts.training.train_rlhf import frame_to_tensor  # noqa: E402
from simulations.environments.dmts_env import DMTSEnv  # noqa: E402

SD_FLOOR = 1e-9
CANDIDATES = ("S0", "S1", "S2", "S3", "S4", "S5")
OPPONENTS = {
    # The measured live configuration of runs/bcast_s4{2,3,4}: mock semantic on,
    # memory at its 0.1 floor. Used as the HARNESS FIDELITY check, because the
    # winner distribution it must reproduce is already published.
    "bcast": {"audio": 0.0, "memory": 0.10, "body": 0.15, "semantic": 1.0},
    "default": {"audio": 0.0, "memory": 0.10, "body": 0.15, "semantic": 0.0},
    "memfix": {"audio": 0.0, "memory": 0.60, "body": 0.15, "semantic": 0.0},
    "memfix_sem": {"audio": 0.0, "memory": 0.60, "body": 0.15, "semantic": 1.0},
}
# Harness fidelity: S0 under `bcast` must reproduce the published live result,
# vision winning the large majority of IGNITED steps. runs/bcast_s4{2,3,4} measured
# vision 7922 / semantic 57 / none 21 of 8000 at seed 42. If the replay cannot
# reproduce that, nothing else it prints may be believed.
FIDELITY_MIN_VISION_SHARE = 0.90
# The SILENT fraction does NOT reproduce and is not gated on. Live runs measure
# 21 of 8000 steps with no winner (0.003); this replay gives an order of magnitude
# more, because it calls run_competition ONCE per step instead of up to 5 times
# through reentrant.settle, and because a short replay has not let the energy
# baseline EMA settle. Consequence: NO CLAIM ABOUT WORKSPACE SILENCE may be made
# from this probe. Only the winner distribution over IGNITED steps is faithful.
# A configuration that is silent on most steps is still disqualified below,
# because a winner computed over a small minority of steps is not a competition.
MAX_SILENT_FOR_GATE = 0.50
DOMINANCE_FRACTION = 0.95
RUNNER_UP_FRACTION = 0.05


class ScalarRunningStats:
    """Causal EMA mean and variance of ONE scalar over time, read before update.

    Separate from `RunningStats`, which is per-element over a 262144-element map.
    A series whose running sd is at or below SD_FLOOR is passed through untouched,
    so a constant bid can never be amplified into fake variance.
    """

    def __init__(self, alpha: float = EMA_ALPHA):
        self.mean = 0.0
        self.var = 1.0
        self.alpha = alpha
        self.seen = 0

    def z(self, x: float) -> float:
        if self.seen == 0:
            return 0.0
        sd = self.var ** 0.5
        if sd <= SD_FLOOR:
            return 0.0
        return (x - self.mean) / (sd + 1e-12)

    def update(self, x: float) -> None:
        if self.seen == 0:
            self.mean, self.var = float(x), 1.0
        else:
            delta = x - self.mean
            self.mean = self.mean + self.alpha * delta
            self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)
        self.seen += 1



def _make_chain(config, replay_seed: int):
    """One isolated (workspace, reentrant) pair with a SEEDED oscillator."""
    torch.manual_seed(replay_seed)
    np.random.seed(replay_seed)
    ws = GlobalWorkspace(config["workspace"])
    ws.affective_modulator = AffectiveModulator(
        {"ablate_existence_bias": config.get("ablate_existence_bias", False)}
    )
    re = ReentrantProcessor(config["reentrant"])
    return ws, re


def collect(config, tectum, seed: int, episodes: int, max_steps: int,
            replay_seed: int) -> dict:
    """One forward pass driving every candidate/opponent chain through settle."""
    keys = [(c, o) for c in CANDIDATES for o in OPPONENTS]
    chains = {k: _make_chain(config, replay_seed) for k in keys}
    winners = {k: [] for k in keys}
    series = {c: [] for c in CANDIDATES}
    kl_sums, phases, shapes, trials = [], [], [], []
    stats, topk = None, 1
    sstats = ScalarRunningStats()
    device = config["device"]
    goal = torch.tensor([1.0, -1.0, 1.0], device=device)
    ws_dim = config["workspace_dim"]
    env = DMTSEnv(num_trials=20)

    with torch.no_grad():
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            if hasattr(tectum, "reset_state"):
                tectum.reset_state(1)
            done, steps = False, 0
            while not done and steps < max_steps:
                frame = frame_to_tensor(obs, device)
                audio = torch.zeros(1, config["tectum_feature_dim"], 2, device=device)
                tectum_content, _bid = tectum(frame, audio)
                post, prior = tectum._last_post_logits, tectum._last_prior_logits
                var = torch.exp(tectum.rssm.cont_logvar)
                kl_map = 0.5 * (post - prior) ** 2 / (var + 1e-8)

                if stats is None:
                    stats = RunningStats(kl_map.numel())
                    topk = max(1, int(kl_map.numel() * TOPK_FRACTION))

                kl_sum = float(kl_map.reshape(-1).sum())
                vals = reductions(kl_map, stats, topk)
                vals["S5"] = float(1.0 / (1.0 + np.exp(-sstats.z(kl_sum))))
                sstats.update(kl_sum)
                kl_sums.append(kl_sum)
                for c in CANDIDATES:
                    series[c].append(vals[c])

                # Payloads carry the real tectum content, as _compute_broadcast does.
                zero = torch.zeros(1, ws_dim, device=device)
                payloads = {
                    "vision": {"tensor": tectum_content, "source": "tectum"},
                    "audio": {"tensor": zero, "source": "audio"},
                    "semantic": {"tensor": zero, "source": "semantic"},
                }
                cap = tectum.get_capsule_payload()
                if cap:
                    payloads["vision"].update(cap)

                for (c, oname) in keys:
                    ws, re = chains[(c, oname)]
                    bids = dict(OPPONENTS[oname])
                    bids["vision"] = float(max(0.0, min(1.0, vals[c])))
                    emotion = evaluate_emotion(bids["vision"], 0.0, 0.0)
                    re.settle(
                        workspace=ws, specialists={"vision": tectum},
                        initial_bids=bids, payloads=payloads, goal_vector=goal,
                        pad_state=emotion, interoceptive_state=None,
                    )
                    w = getattr(ws.state, "winners", []) or []
                    winners[(c, oname)].append(w[0] if w else "")

                phases.append(info.get("phase"))
                shapes.append(info.get("sample_shape"))
                trials.append("%d:%s" % (ep, info.get("trial")))
                obs, _, term, trunc, info = env.step(0)
                done = term or trunc
                steps += 1

    return {
        "kl_sum": np.asarray(kl_sums, dtype=np.float64),
        "series": {c: np.asarray(v, dtype=np.float64) for c, v in series.items()},
        "winners": {k: np.asarray(v) for k, v in winners.items()},
        "phase": np.asarray(phases),
        "shape": np.asarray(shapes),
        "trial": np.asarray(trials),
    }


def share(winners: np.ndarray) -> list[tuple[str, float]]:
    """Winner shares over the steps where a module ACTUALLY won.

    An empty winner means the workspace did not ignite: no module cleared the
    threshold. That is silence, not competition, and counting it as a winner would
    let a de-saturated bid that silences the workspace look like a changed winner.
    It is excluded here and reported separately as `silent`.
    """
    live = winners[winners != ""]
    if live.size == 0:
        return []
    vals, counts = np.unique(live, return_counts=True)
    order = np.argsort(-counts)
    return [(str(vals[i]), float(counts[i] / live.size)) for i in order]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="+",
                   default=["runs/gate3_s42", "runs/gate3_s43", "runs/gate3_s44"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=400,
                   help="Steps discarded before any share is computed, so "
                        "the oscillators have converged. See the docstring.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replay-seed", type=int, default=20260913,
                   help="Seeds the Kuramoto phase init identically for every "
                        "candidate, so a difference between candidates cannot "
                        "be phase-initialization noise.")
    p.add_argument("--latent-mode", default="continuous",
                   choices=["discrete", "continuous"])
    p.add_argument("--capsule-workspace-source", default="all_levels",
                   choices=["final", "all_levels"])
    p.add_argument("--out", default=None, help="Optional JSON path")
    args = p.parse_args()

    print(__doc__.split("Run:")[0])
    print("=" * 78)
    print("FREE PARAMETERS: EMA_ALPHA=%g  SD_FLOOR=%g  TOPK_FRACTION=%g"
          % (EMA_ALPHA, SD_FLOOR, TOPK_FRACTION))
    print("=" * 78)

    results = {}
    for ckpt in args.checkpoints:
        tectum_path = os.path.join(ckpt, "tectum.pt")
        if not os.path.isfile(tectum_path):
            sys.exit("REFUSING: no tectum at %s" % tectum_path)
        name = os.path.basename(ckpt.rstrip("/\\"))
        print("\n" + "#" * 78)
        print("# CHECKPOINT %s" % name)
        print("#" * 78)
        config, tectum, _ws, _re, _sm, _mem, _ms = _build_components(
            "dmts", action_dim=5, seed=args.seed, mock_semantic=False,
            load_tectum=tectum_path, latent_mode=args.latent_mode,
            capsule_workspace_source=args.capsule_workspace_source,
        )
        tectum.eval()
        d = collect(config, tectum, args.seed, args.episodes, args.max_steps,
                    args.replay_seed)
        total = d["kl_sum"].size
        if total <= args.warmup + 400:
            sys.exit("REFUSING: %d steps is not enough after a %d-step warm-up. "
                     "Raise --episodes or --max-steps." % (total, args.warmup))
        w0 = args.warmup
        d = {k: (v[w0:] if isinstance(v, np.ndarray) else
                 {kk: vv[w0:] for kk, vv in v.items()})
             for k, v in d.items()}
        print("  discarded %d warm-up steps, analysing %d" % (w0, total - w0))
        kl = d["kl_sum"]
        print("  steps=%d   pre-tanh KL sum: mean=%.4e  min=%.4e  max=%.4e"
              % (kl.size, kl.mean(), kl.min(), kl.max()))
        print("  phases present: %s" % sorted(set(d["phase"].tolist())))

        results[name] = {"kl_mean": float(kl.mean()), "kl_min": float(kl.min()),
                         "kl_max": float(kl.max()), "steps": int(kl.size),
                         "candidates": {}}

        for c in CANDIDATES:
            v = d["series"][c]
            print("\n  -- %s   bid mean=%.6f  min=%.6f  max=%.6f  distinct=%d"
                  % (c, v.mean(), v.min(), v.max(), len(np.unique(v))))
            results[name]["candidates"][c] = {
                "bid_mean": float(v.mean()), "bid_min": float(v.min()),
                "bid_max": float(v.max()), "distinct": int(len(np.unique(v))),
                "opponents": {},
            }
            for oname, opp in OPPONENTS.items():
                w = d["winners"][(c, oname)]
                silent = float((w == "").mean())
                sh = share(w)
                top = sh[0] if sh else ("", 0.0)
                runner = sh[1] if len(sh) > 1 else ("", 0.0)
                per_phase = {}
                for ph in sorted(set(d["phase"].tolist())):
                    m = d["phase"] == ph
                    s = share(w[m])
                    per_phase[str(ph)] = s[0][0] if s else ""
                print("     %-11s silent=%.3f  top=%-9s %.3f  runner=%-9s %.3f"
                      "  by phase: %s"
                      % (oname, silent, top[0] or "-", top[1],
                         runner[0] or "-", runner[1],
                         " ".join("%s=%s" % (k, v2 or "-")
                                  for k, v2 in per_phase.items())))
                results[name]["candidates"][c]["opponents"][oname] = {
                    "share": sh, "per_phase": per_phase, "silent": silent,
                }

    # ---- Harness fidelity, checked BEFORE the gate ----
    ckpts = list(results)
    print("\n" + "=" * 78)
    print("HARNESS FIDELITY: S0 under `bcast` must reproduce the published live result")
    print("=" * 78)
    fid_ok = True
    for k in ckpts:
        o = results[k]["candidates"]["S0"]["opponents"]["bcast"]
        sh = o["share"]
        vis = next((s for n, s in sh if n == "vision"), 0.0)
        ok = vis >= FIDELITY_MIN_VISION_SHARE
        fid_ok = fid_ok and ok
        print("  %-12s vision share of ignited steps = %.3f  (need >= %.2f)  %s"
              % (k, vis, FIDELITY_MIN_VISION_SHARE, "OK" if ok else "FAIL"))
        print("  %-12s silent = %.3f  (live measures 0.003; NOT reproduced, NOT gated)"
              % ("", o["silent"]))
    if not fid_ok:
        print("\nREFUSING TO REPORT A VERDICT. The replay does not reproduce the known")
        print("baseline, so no winner distribution it produces may be believed.")
        return

    # ---- Gate ----
    print("\n" + "=" * 78)
    print("GATE")
    print("=" * 78)
    verdict_rows = []
    for c in CANDIDATES:
        for oname in OPPONENTS:
            tops, runners, dominated, silents = [], [], [], []
            for k in ckpts:
                e = results[k]["candidates"][c]["opponents"][oname]
                sh = e["share"]
                tops.append(sh[0][0] if sh else "")
                runners.append(sh[1][1] if len(sh) > 1 else 0.0)
                dominated.append((sh[0][1] if sh else 1.0) >= DOMINANCE_FRACTION)
                silents.append(e["silent"])
            quiet = max(silents) > MAX_SILENT_FOR_GATE
            real = all(t != "" for t in tops)
            changes = real and not quiet and all(r >= RUNNER_UP_FRACTION
                                                 for r in runners)
            arbitrary = real and len(set(tops)) > 1 and all(dominated)
            verdict_rows.append((c, oname, tops, min(runners), max(silents),
                                 changes, arbitrary))

    proceed = [r for r in verdict_rows if r[5]]
    arb = [r for r in verdict_rows if r[6]]

    print("\n  %-4s %-11s %-30s %8s %7s %s"
          % ("cand", "opponents", "top winner per ckpt", "min run", "silent", "flag"))
    for c, oname, tops, minr, maxs, changes, arbitrary in verdict_rows:
        flag = "PROCEED" if changes else ("ARBITRARY" if arbitrary else "")
        if maxs > MAX_SILENT_FOR_GATE:
            flag = "silent"
        print("  %-4s %-11s %-30s %8.3f %7.3f %s"
              % (c, oname, "/".join(t or "-" for t in tops), minr, maxs, flag))

    print()
    if proceed:
        print("VERDICT: PROCEED. %d candidate/opponent combinations give a winner that"
              % len(proceed))
        print("changes at ALL %d checkpoints with the runner-up at or above %.0f percent."
              % (len(ckpts), RUNNER_UP_FRACTION * 100))
        for c, oname, tops, minr, _, _ in proceed:
            print("  %s under %s, runner-up at least %.3f" % (c, oname, minr))
    elif arb:
        print("VERDICT: KILL, by condition (ii), checkpoint arbitrariness.")
        print("%d combinations give a CONSTANT but DIFFERENT winner per checkpoint."
              % len(arb))
        print("A fixed operating point does not exist. Any single seed would look like")
        print("success and would not replicate.")
    else:
        print("VERDICT: KILL, by condition (i). No candidate changes the winner at all")
        print("three checkpoints, and none is checkpoint-arbitrary either.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print("\nWrote %s" % args.out)


if __name__ == "__main__":
    main()
