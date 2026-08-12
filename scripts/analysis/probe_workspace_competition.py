"""
Does the workspace competition ever switch winner, and should it?

GWT-1 ("multiple specialized systems operating in parallel") and GWT-2 ("limited
capacity workspace with a bottleneck plus selective attention") are both marked
IMPLEMENTED in the indicator rubric. A selection mechanism that always selects the same
module is worth examining before either status is trusted.

A first look on the trained checkpoint found vision winning 200 of 200 steps, with raw
bids on incommensurable scales: vision saturated at exactly 1.0 by `torch.tanh(kl_div)`
in sensory_tectum.py, against memory around 0.10 and body around 0.074.

Vision dominating a VISUAL task is not automatically a defect, so "is vision too strong"
is the wrong question. DMTS has a phase structure that makes a better one decidable:
during the DELAY there is no sample on screen and the task depends on held information,
so a workspace doing state-dependent attention should plausibly favour memory there, and
vision during SAMPLE and CHOICE.

PRE-STATED GATE, written before any number from this probe was read:

    DEGENERATE:     one module wins in >= 95 percent of steps in EVERY phase, including
                    delay. The selection mechanism is not selecting.
    PHASE-TRACKING: the modal winner differs between sample and delay. Selection works
                    and there is nothing here to fix.
    WEAK:           the winner varies but not with phase. Competition is live but not
                    task-driven, which is a third finding distinct from both.

STEP 2, the cause discriminator, runs only if the verdict is DEGENERATE. It rescales the
bids OFFLINE, without touching the model, by standardizing each module's bid across the
run, and recomputes who would have won. That asks "which module is most surprised
relative to its own normal", which is what a salience competition should compare. If
vision still wins everywhere the cause is genuine salience and the bid scale is
innocent. If the winner starts moving, the cause is the scale.

Read-only. No training run, no checkpoint written, no model modified.

TWO LIMITATIONS, both stated up front because both bear on how the result reads.

1. `_build_components` hardcodes `enable_audio=False` and this probe does not override
   it, so audio and semantic bid exactly 0.0. This is a three-module race between
   vision, memory and body.

2. HARNESS FIDELITY. `_compute_broadcast` in probe_perception_decodability.py:231-237
   hardcodes `memory: 0.1` and `body: 0.05` as literals. The training loop
   (train_rlhf.py:1056-1073) computes them: `memory_bid` rises with retrieval score to a
   cap of 0.6, and `body` is a two-valued switch, 0.15 when interoceptive energy < 0.4
   and 0.05 otherwise. So the probe measures a race with the two non-vision bids pinned
   at their floors.

   Limitation 2 does not spoil the verdict, and the reason is arithmetic rather than
   measurement. `raw_bids["vision"]` is `min(1.0, vision_bid)` in both the probe and the
   training loop. Memory's cap is 0.6 and body's ceiling is 0.15. Whenever the vision bid
   reaches 0.6 the competition is decided before any module competes. This probe
   therefore reports the raw vision bid distribution as its primary number: if vision
   sits at or above 0.6, no retrieval score and no energy state can change the winner,
   and the floors the probe uses are irrelevant to who wins.

Run:
    python -m scripts.analysis.probe_workspace_competition \\
        --load-tectum runs/gate_ckpt_s42/tectum.pt
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.probe_perception_decodability import (
    _build_components, _compute_broadcast,
)
from scripts.training.train_rlhf import frame_to_tensor
from simulations.environments.dmts_env import DMTSEnv

PHASES = ("fixation", "sample", "delay", "choice")
DOMINANCE_THRESHOLD = 0.95     # pre-stated
SEEDS = (42, 43, 44)

# Ceilings read from the training loop, train_rlhf.py:1056-1073. Not measured here.
MEMORY_BID_CAP = 0.6           # min(0.6, 0.1 + retrieval_score * 0.5)
BODY_BID_CAP = 0.15            # 0.15 when interoceptive energy < 0.4, else 0.05
DECIDING_VISION_BID = max(MEMORY_BID_CAP, BODY_BID_CAP)
# Literals the probe harness substitutes for the two computed bids.
HARNESS_CONSTANT_BIDS = {"memory": 0.1, "body": 0.05}


def collect(config, tectum, workspace, reentrant, self_model, memory, mock_sem,
            episodes: int, seed: int) -> list[dict]:
    """Record raw bids, bound bids, winner and phase per step.

    Raw bids are captured by wrapping `binding_system.bind_bids`, which is the single
    point every bid passes through on its way to the competition. Read-only: the wrapper
    records and delegates.
    """
    captured: list[tuple[dict, dict]] = []
    binder = workspace.binding_system
    real_bind = binder.bind_bids

    def spy(bids, *a, **kw):
        bound, sync = real_bind(bids, *a, **kw)
        captured.append((dict(bids), dict(bound)))
        return bound, sync

    # Capture the pre-tanh KL that produces the bid. `sensory_tectum.py:456` is
    # `bid = torch.tanh(kl_div)` on a 0-dim scalar; the only other tanh in the tectum
    # (line 239, the GRU candidate) is multi-element, so numel() separates them.
    pre_tanh: list[float] = []
    real_tanh = torch.tanh

    def tanh_spy(x, *a, **kw):
        out = real_tanh(x, *a, **kw)
        if isinstance(x, torch.Tensor) and x.numel() == 1:
            pre_tanh.append(float(x.item()))
        return out

    binder.bind_bids = spy
    torch.tanh = tanh_spy
    rows = []
    try:
        env = DMTSEnv(num_trials=20)
        with torch.no_grad():
            for ep in range(episodes):
                obs, info = env.reset(seed=seed + ep)
                if hasattr(tectum, "reset_state"):
                    tectum.reset_state(1)
                done, steps = False, 0
                while not done and steps < 200:
                    before = len(captured)
                    frame = frame_to_tensor(obs, config["device"])
                    audio = torch.zeros(1, config["tectum_feature_dim"], 2,
                                        device=config["device"])
                    content, vision_bid = tectum(frame, audio)
                    _compute_broadcast(config, tectum, workspace, reentrant,
                                       self_model, memory, mock_sem,
                                       content, vision_bid, obs)
                    if len(captured) > before:
                        raw, bound = captured[-1]
                        live = {k: v for k, v in bound.items() if v != 0.0}
                        if live:
                            ordered = sorted(live.items(), key=lambda kv: -kv[1])
                            rows.append({
                                "phase": info.get("phase"),
                                "kl_pre_tanh": pre_tanh[-1] if pre_tanh else float("nan"),
                                # Tectum output BEFORE `min(1.0, ...)` clamping. This is
                                # the number that decides the competition analytically.
                                "vision_bid_preclamp": float(vision_bid),
                                "raw": raw,
                                "bound": bound,
                                "winner": ordered[0][0],
                                "margin": (ordered[0][1] - ordered[1][1]
                                           if len(ordered) > 1 else float("nan")),
                            })
                    obs, _, term, trunc, info = env.step(0)
                    done = term or trunc
                    steps += 1
    finally:
        binder.bind_bids = real_bind
        torch.tanh = real_tanh
    return rows


def winners_by_phase(rows) -> dict:
    out = {}
    for phase in PHASES:
        sel = [r["winner"] for r in rows if r["phase"] == phase]
        if sel:
            out[phase] = collections.Counter(sel)
    return out


def standardized_rewinner(rows) -> tuple[collections.Counter | None, dict]:
    """STEP 2: who would win if every module's bid were on a common scale?

    Standardize each module's RAW bid across the whole run (z-score), then recompute the
    winner. If vision still wins the cause is genuine salience; if the winner moves the
    cause is scale.

    The discriminator is only DEFINED when the bids carry variance. A constant bid has
    zero standard deviation and no z-score exists for it. An earlier version of this
    function pinned those at 0.0 and returned a winner anyway, which made
    `max()` fall through to alphabetical order and reported "audio wins 400 of 400"
    for a module whose bid is exactly 0.0 at every step. That was an artifact of the
    tie-break, not a finding, so this version refuses to answer instead.

    Returns (counter, diagnostics). `counter` is None when the discriminator is
    undefined, and `diagnostics` always reports the per-module standard deviation so a
    reader can see why.
    """
    names = sorted({k for r in rows for k in r["raw"]})
    series = {n: np.array([r["raw"].get(n, 0.0) for r in rows], dtype=float)
              for n in names}
    sds = {n: float(arr.std()) for n, arr in series.items()}
    varying = [n for n, sd in sds.items() if sd > 1e-12]
    diagnostics = {"sd": sds, "varying": varying}

    # Fewer than two varying bids means there is no competition to rescale: at most one
    # module carries any signal and the rest are constants with no defined z-score.
    if len(varying) < 2:
        return None, diagnostics

    z = {n: (series[n] - series[n].mean()) / sds[n] for n in varying}
    out = [max(varying, key=lambda n: z[n][i]) for i in range(len(rows))]
    return collections.Counter(out), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-tectum", default="runs/gate_ckpt_s42/tectum.pt")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--latent-mode", default="continuous",
                        choices=["discrete", "continuous"])
    parser.add_argument("--capsule-workspace-source", default="all_levels",
                        choices=["final", "all_levels"])
    args = parser.parse_args()

    per_seed = {}
    for seed in SEEDS:
        cfg, tectum, ws, re_, sm, mem, msem = _build_components(
            "dmts", action_dim=5, seed=seed, mock_semantic=False,
            load_tectum=args.load_tectum, latent_mode=args.latent_mode,
            capsule_workspace_source=args.capsule_workspace_source)
        per_seed[seed] = collect(cfg, tectum, ws, re_, sm, mem, msem,
                                 args.episodes, seed)

    print("=" * 84)
    print("WINNER DISTRIBUTION BY PHASE, per seed")
    print("=" * 84)
    for seed, rows in per_seed.items():
        print(f"\n  seed {seed}  ({len(rows)} steps)")
        for phase, counter in winners_by_phase(rows).items():
            total = sum(counter.values())
            top, n = counter.most_common(1)[0]
            margins = [r["margin"] for r in rows
                       if r["phase"] == phase and np.isfinite(r["margin"])]
            share = n / total
            print(f"    {phase:<9} n={total:>4}  winner={top:<9} "
                  f"share={share:>6.1%}  1st-2nd margin={np.mean(margins):.4f}"
                  if margins else
                  f"    {phase:<9} n={total:>4}  winner={top:<9} share={share:>6.1%}")

    print("\n" + "=" * 84)
    print("RAW VISION BID (tectum output, before the min(1.0, .) clamp)")
    print("=" * 84)
    print(f"  The competition is decided analytically whenever this reaches "
          f"{DECIDING_VISION_BID}, the")
    print(f"  highest value any other module can reach (memory cap "
          f"{MEMORY_BID_CAP}, body cap {BODY_BID_CAP}).\n")
    for seed, rows in per_seed.items():
        v = np.array([r["vision_bid_preclamp"] for r in rows], dtype=float)
        above = float((v >= DECIDING_VISION_BID).mean())
        print(f"  seed {seed}  n={len(v):>4}  min={v.min():.9f}  max={v.max():.9f}  "
              f"distinct={len(np.unique(v))}  >= {DECIDING_VISION_BID}: {above:.1%}")

    print("\n" + "=" * 84)
    print("RAW BIDS ENTERING THE COMPETITION (per module)")
    print("=" * 84)
    print("  'harness' marks a value the probe substitutes for a bid the training loop")
    print("  computes. See limitation 2 in the module docstring.\n")
    for seed, rows in per_seed.items():
        print(f"  seed {seed}")
        names = sorted({k for r in rows for k in r["raw"]})
        for n in names:
            arr = np.array([r["raw"].get(n, 0.0) for r in rows], dtype=float)
            tag = "  <- harness constant" if n in HARNESS_CONSTANT_BIDS else ""
            print(f"    {n:<9} min={arr.min():.6f}  max={arr.max():.6f}  "
                  f"distinct={len(np.unique(arr)):>4}{tag}")

    print("\n" + "=" * 84)
    print("PRE-STATED GATE")
    print("=" * 84)
    verdicts = []
    for seed, rows in per_seed.items():
        by_phase = winners_by_phase(rows)
        shares = {p: c.most_common(1)[0][1] / sum(c.values()) for p, c in by_phase.items()}
        tops = {p: c.most_common(1)[0][0] for p, c in by_phase.items()}
        dominated = all(s >= DOMINANCE_THRESHOLD for s in shares.values())
        phase_tracks = tops.get("sample") != tops.get("delay")
        verdicts.append("PHASE-TRACKING" if phase_tracks
                        else ("DEGENERATE" if dominated else "WEAK"))
    agreed = set(verdicts)
    print(f"  per-seed verdicts: {verdicts}")
    if len(agreed) == 1:
        v = verdicts[0]
        print(f"\n  VERDICT: {v} at all {len(SEEDS)} seeds.")
        if v == "DEGENERATE":
            print("  One module wins >= 95% of steps in every phase, delay included.")
            print("  The selection mechanism is not selecting.")
        elif v == "PHASE-TRACKING":
            print("  The modal winner differs between sample and delay. Nothing to fix.")
        else:
            print("  The winner varies but does not track phase.")
    else:
        print(f"\n  VERDICT: seeds DISAGREE {agreed}. Not decisive; report as such.")

    if "DEGENERATE" in agreed:
        print("\n" + "=" * 84)
        print("STEP 2: cause discriminator (offline rescale, model untouched)")
        print("=" * 84)
        print("  Standardizing each module's raw bid across the run, then recomputing")
        print("  the winner. Requires at least two bids with non-zero variance.\n")
        undefined = 0
        for seed, rows in per_seed.items():
            live = collections.Counter(r["winner"] for r in rows)
            rescaled, diag = standardized_rewinner(rows)
            print(f"  seed {seed}")
            print(f"    bid sd    : " + "  ".join(
                f"{n}={sd:.3e}" for n, sd in sorted(diag["sd"].items())))
            print(f"    as-is     : {dict(live)}")
            if rescaled is None:
                undefined += 1
                print(f"    rescaled  : UNDEFINED. Only {len(diag['varying'])} bid(s) "
                      f"carry any variance: {diag['varying']}")
            else:
                print(f"    rescaled  : {dict(rescaled)}")

        if undefined == len(per_seed):
            print("\n  STEP 2 CANNOT RUN, at every seed.")
            print("  The discriminator asks which module is most surprised relative to")
            print("  its own normal. That question needs the bids to vary. They do not.")
            print("  So the cause is neither of the two the step was built to separate:")
            print("  the bids carry no salience signal to be mis-scaled in the first")
            print("  place. Reported as an undefined test, not as a third outcome.")
        else:
            print("\n  If the rescaled winner moves, the cause is SCALE and the bid is at")
            print("  fault. If vision still wins everywhere, the cause is genuine")
            print("  SALIENCE and the bid scale is innocent.")

        print("\n" + "=" * 84)
        print("PRE-TANH KL (the quantity the vision bid is computed from)")
        print("=" * 84)
        print("  `sensory_tectum.py:456` is bid = tanh(kl_div). float32 tanh returns")
        print("  exactly 1.0 above an input of roughly 9.0.\n")
        for seed, rows in per_seed.items():
            k = np.array([r["kl_pre_tanh"] for r in rows], dtype=float)
            k = k[np.isfinite(k)]
            if k.size:
                print(f"  seed {seed}  n={k.size:>4}  min={k.min():.1f}  "
                      f"max={k.max():.1f}  mean={k.mean():.1f}  sd={k.std():.1f}")

        print("\n" + "=" * 84)
        print("WOULD RESCALING THE BID BUY PHASE-TRACKING? (scale-free test)")
        print("=" * 84)
        print("  Rescaling the bid is a monotone transform of the pre-tanh KL, so it")
        print("  cannot create a phase difference that the KL does not already carry.")
        print("  Any winner produced by picking a scale is a property of the scale")
        print("  chosen, not of the model. The scale-free question is whether the KL")
        print("  separates sample from delay at all.\n")
        for seed, rows in per_seed.items():
            byp = {p: np.array([r["kl_pre_tanh"] for r in rows
                                if r["phase"] == p and np.isfinite(r["kl_pre_tanh"])],
                               dtype=float)
                   for p in PHASES}
            byp = {p: a for p, a in byp.items() if a.size}
            print(f"  seed {seed}  " + "  ".join(
                f"{p}={a.mean():.1f}" for p, a in byp.items()))
            s, d = byp.get("sample"), byp.get("delay")
            if s is not None and d is not None and s.size > 1 and d.size > 1:
                pooled = np.sqrt((s.var(ddof=1) + d.var(ddof=1)) / 2.0)
                cohen = (s.mean() - d.mean()) / pooled if pooled > 1e-12 else float("nan")
                print(f"            sample-vs-delay Cohen's d = {cohen:+.3f}")


if __name__ == "__main__":
    main()
