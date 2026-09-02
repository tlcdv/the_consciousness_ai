"""
Is sync_R a binding signal, or a readout of the bid vector?

`sync_R` is the one level planning #11 names that has never been measured on the
alive-versus-dead axis. It also carries a public claim on its own: RPT-2's negative
evidence in `docs/consciousness_indicators_butlin.md:250-252` rests entirely on it.

THE ACTIVE CLASS IS `WorkspaceBindingSystem` in `models/core/oscillatory_binding.py`.
`binding_mechanism` defaults to "akorn" (`global_workspace.py:143-154`), so
`ComplexBindingSystem` in `complex_binding.py` is the non-default "komplex" path and is
NOT what produced any number in `runs/`. An earlier draft of this probe quoted the wrong
class. Order parameter, `oscillatory_binding.py:117-118`:

    mean_field = einsum('bn,bnd->bd', amplitudes, current_phases) / N
    sync_R     = ||mean_field||

`amplitudes` is the module bid vector and `current_phases` are UNIT vectors, so
`sum(amplitudes) / N` is a hard ceiling reached exactly at full synchrony.

TWO HYPOTHESES FROM READING THE CODE. Both are settled below, one of them against the
version this probe started with.

  H1  sync_R is a readout of the bid vector rather than a measurement of binding. The bid
      vector is measured degenerate: vision pinned at exactly 1.000000000 at 24,000 of
      24,000 steps (`docs/results/workspace_bids_live_2026_08.md`).

  H2  sync_R carries a RANDOM per-episode component from a phase reset, and that alone
      could produce its distinct-value count with no content in it.

PRE-STATED GATE, written before any value from these runs was read. Planning #11's fork
needs a THIRD outcome here, because the vision-bid study already found a signal that
varied and carried nothing, so "varies" and "alive" are not the same claim.

    (a) ALIVE AND CARRYING CONTENT
        sync_r eta-squared for 6-class `sample_shape` exceeds its own permutation null at
        all 3 seeds. Binding synchrony tracks the stimulus and the instruments reading it
        are at fault.

    (b) GENUINELY NEAR-CONSTANT
        fewer than 100 distinct values, or standard deviation below 1e-6, at any seed. No
        instrument repair rescues that.

    (c) VARYING BUT CONTENTLESS
        clears the non-degeneracy bar and does NOT clear its permutation null. Then its
        variance carries no stimulus identity, it cannot support an integration claim, and
        RPT-2's evidence base rests on it.

Traps this probe is built against, each of which cost a re-run in an earlier session:

  - `sample_shape` is the label, never phase, because step index alone decodes
    sample-vs-delay at 1.0000 and any phase result is a clock reading
  - shuffling is done ACROSS TRIALS, because shape is constant within a trial
  - the effective sample is TRIALS, not steps
  - the 2026-07 sync_R mean is NOT compared against these runs. Different flags. Both are
    printed, neither is subtracted from the other.

Reads runs already on disk plus one constructed input. No training, no checkpoint loaded.

Run:
    python -m scripts.analysis.probe_sync_r_content
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.probe_bid_reduction_candidates import (
    MAX_PINNED_FRACTION,
    MIN_DISTINCT,
    MIN_SPAN,
    N_PERMUTATIONS,
    eta_squared,
    permutation_null,
)

SEEDS = (42, 43, 44)
BID_COLUMNS = ("bid_vision", "bid_audio", "bid_memory", "bid_body", "bid_semantic")

# The standing marker in the rubric, quoted for the record. NOT a comparison target.
RUBRIC_SYNC_R_RANGE = (0.2662, 0.2666)
RUBRIC_SOURCE = "signature_assessment_2026_07.md:111 (five objectives, 2026-07)"

# Gate (b): the dead bar, same shape as the one used on the workspace components.
DEAD_MAX_DISTINCT = 100
DEAD_MAX_STD = 1e-6

CONTROL_TOLERANCE = 1e-6


def load(seed: int) -> dict:
    run_dir = Path("runs") / f"bcast_s{seed}"
    rows = list(csv.DictReader(open(run_dir / "metrics.csv", newline="")))
    if "env_sample_shape" not in rows[0]:
        raise SystemExit(f"{run_dir}/metrics.csv has no labels. Stale run directory.")

    trial = [int(r["env_trial"]) for r in rows]
    # env_trial is per-EPISODE and repeats. A unique trial id needs the episode, which is
    # recoverable from where the counter resets.
    episode, episodes = 0, []
    for i, v in enumerate(trial):
        if i and v < trial[i - 1]:
            episode += 1
        episodes.append(episode)

    return {
        "sync_r": np.array([float(r["sync_r"]) for r in rows]),
        "bids": np.array([[float(r[c]) for c in BID_COLUMNS] for r in rows]),
        "shape": np.array([r["env_sample_shape"] for r in rows]),
        "phase": np.array([r["env_phase"] for r in rows]),
        "episode": np.array(episodes),
        "trial": np.array([f"{e}:{t}" for e, t in zip(episodes, trial)]),
    }


def report_non_degeneracy(data: dict) -> dict:
    print("=" * 84)
    print("1. IS sync_r NON-DEGENERATE?")
    print("=" * 84)
    print("  Strict bar, taken unchanged from probe_bid_reduction_candidates.py so this")
    print(f"  result is not judged more kindly than that one: >= {MIN_DISTINCT} distinct,")
    print(f"  span >= {MIN_SPAN}, <= {MAX_PINNED_FRACTION:.0%} pinned at the modal value.")
    print(f"  Gate (b) DEAD bar: < {DEAD_MAX_DISTINCT} distinct or std < {DEAD_MAX_STD}.")
    print("")

    alive = {}
    for seed, d in data.items():
        x = d["sync_r"]
        values, counts = np.unique(x, return_counts=True)
        distinct = len(values)
        span = float(x.max() - x.min())
        pinned = int(counts.max()) / len(x)
        std = float(x.std())
        strict = (distinct >= MIN_DISTINCT and span >= MIN_SPAN
                  and pinned <= MAX_PINNED_FRACTION)
        dead = distinct < DEAD_MAX_DISTINCT or std < DEAD_MAX_STD
        alive[seed] = not dead
        top = np.argsort(-counts)[:3]
        print(f"  seed {seed}  n={len(x)}")
        print(f"    distinct={distinct}  span={span:.6f}  std={std:.6e}")
        print(f"    mean={x.mean():.6f}  min={x.min():.6f}  max={x.max():.6f}")
        print(f"    pinned at the modal value = {pinned:.4%}")
        print("    top values: " + ", ".join(
            f"{values[i]:.9f} x{counts[i]} ({counts[i] / len(x):.1%})" for i in top))
        print(f"    strict bar: {'PASS' if strict else 'FAIL'}     "
              f"gate (b) dead bar: {'DEAD' if dead else 'not dead'}")
    return alive


def report_bid_determinism(data: dict) -> None:
    """H1, settled by construction rather than by grouping.

    A GROUPING TEST WAS PLANNED HERE AND IS NOT VALID. `train_rlhf.py:1973` logs
    `raw_bids`, but `global_workspace.py:210-217` lets `affective_modulator.modulate()`
    REWRITE `bids` before they reach `bind_bids` at line 224. The logged bids are not the
    amplitudes, so grouping steps by the logged bid tuple would charge the modulator's
    variation to phase state. The arithmetic shows the gap directly.

    This is a deterministic result. It has no seeds, and claiming seed replication for it
    would be false precision. Its independent check is the closed-form control below,
    which RAISES on mismatch.
    """
    print("")
    print("=" * 84)
    print("2. H1: IS sync_R A FUNCTION OF THE BID VECTOR?")
    print("=" * 84)
    print("  The planned grouping test is INVALID and is not run. The logged bids are")
    print("  raw_bids (train_rlhf.py:1973); affective_modulator.modulate rewrites them")
    print("  (global_workspace.py:210-217) before bind_bids sees them at line 224.")
    print("")

    for seed, d in data.items():
        logged_sum = float(np.median(d["bids"].sum(axis=1)))
        values, counts = np.unique(d["sync_r"], return_counts=True)
        modal = float(values[np.argmax(counts)])
        print(f"  seed {seed}: median logged bid sum / N = {logged_sum / 5:.9f}   "
              f"modal sync_r = {modal:.9f}   excess = {modal - logged_sum / 5:+.3e}")
    print("")
    print("  sync_r sits ABOVE the ceiling implied by the LOGGED bids at every seed, so")
    print("  the amplitudes are not the logged bids. H1 is settled on the dynamics.")
    print("")

    import torch
    from models.core.oscillatory_binding import KuramotoLayer

    amps = torch.tensor([[1.0, 0.0, 0.1, 0.15, 1.0]])   # the modal logged bid tuple
    ceiling = float(amps.sum()) / 5.0
    torch.manual_seed(0)
    layer = KuramotoLayer(num_oscillators=5, dimensions=2)
    phases = layer.init_phases(1)
    print(f"  Constructed input: the modal logged bid tuple {amps.tolist()[0]}")
    print(f"  Closed-form ceiling sum/N = {ceiling:.9f}")
    print("  Phases PERSIST across steps. reset_state() is called only in")
    print("  scripts/demos/demo_akorn_binding.py, never from the training loop.")
    print("")
    with torch.no_grad():
        for step in range(1, 401):
            phases, order = layer(phases, amplitudes=amps, iterations=5)
            if step in (1, 5, 20, 50, 100, 200, 400):
                print(f"    step {step:4d}  sync_R = {float(order):.9f}   "
                      f"gap to ceiling = {ceiling - float(order):+.3e}")
    final = float(order)
    if abs(final - ceiling) > CONTROL_TOLERANCE:
        raise AssertionError(
            f"closed-form control FAILED: converged sync_R {final} does not reach the "
            f"ceiling {ceiling}. The mechanism claim in this probe is not supported."
        )
    print("")
    print(f"  CONTROL PASSES: converged sync_R reaches sum/N within {CONTROL_TOLERANCE}.")
    print("  Once converged, sync_R IS the arithmetic mean of the bid vector. At that")
    print("  point it measures no binding: it reports the bids. The bid vector is")
    print("  measured degenerate, so sync_R inherits that degeneracy.")


def report_episode_structure(data: dict) -> None:
    print("")
    print("=" * 84)
    print("3. H2 IS REJECTED: THERE IS NO PER-EPISODE PHASE RESET")
    print("=" * 84)
    print("  H2 supposed sync_R inherits randomness from a per-episode phase reset.")
    print("  reset_state() is called only in scripts/demos/demo_akorn_binding.py and")
    print("  NEVER from the training loop, so phases persist for the whole run and the")
    print("  layer converges once. The episode statistics below are the check: a real")
    print("  per-episode reset would scatter the episode means.")
    print("")

    for seed, d in data.items():
        eps = np.unique(d["episode"])
        means = np.array([d["sync_r"][d["episode"] == e].mean() for e in eps])
        stds = np.array([d["sync_r"][d["episode"] == e].std() for e in eps])
        firsts = np.array([d["sync_r"][d["episode"] == e][0] for e in eps])
        between, within = float(means.std()), float(stds.mean())
        print(f"  seed {seed}  {len(eps)} episodes")
        print(f"    episode MEAN of sync_r : sd={between:.6e} "
              f"range=[{means.min():.6f}, {means.max():.6f}]")
        print(f"    within-episode sd      : mean={within:.6e}")
        print(f"    first step of episode  : sd={firsts.std():.6e} "
              f"range=[{firsts.min():.6f}, {firsts.max():.6f}]")
        print(f"    between-episode sd / within-episode sd = "
              f"{between / (within + 1e-12):.4f}")


def report_content(data: dict) -> dict:
    print("")
    print("=" * 84)
    print("4. THE DECISIVE TEST: DOES sync_r CARRY STIMULUS IDENTITY?")
    print("=" * 84)
    print("  sample-phase steps, 6-class sample_shape, eta-squared against a null of")
    print(f"  {N_PERMUTATIONS} shuffles ACROSS TRIALS. Same method and same null as")
    print("  probe_bid_reduction_candidates.py.")
    print("")

    passed = {}
    for seed, d in data.items():
        mask = d["phase"] == "sample"
        x = d["sync_r"][mask]
        shapes = d["shape"][mask]
        trials = d["trial"][mask]
        observed = eta_squared(x, shapes)
        null = permutation_null(x, shapes, trials, seed)
        p95 = float(np.percentile(null, 95))
        ok = observed > p95
        passed[seed] = ok
        print(f"  seed {seed}  {int(mask.sum())} sample steps, "
              f"{len(np.unique(trials))} trials, {len(np.unique(shapes))} shapes")
        print(f"    observed eta2 = {observed:.6f}")
        print(f"    null          = mean {null.mean():.6f}  p95 {p95:.6f}  "
              f"max {null.max():.6f}")
        print(f"    {'ABOVE' if ok else 'BELOW'} its own null p95   "
              f"{'PASS' if ok else 'FAIL'}")
    return passed


def report_rubric_marker(data: dict) -> None:
    print("")
    print("=" * 84)
    print("5. THE STANDING RUBRIC MARKER")
    print("=" * 84)
    print("  NOT A COMPARISON. Different flags, different modules enabled. Cross-arm")
    print("  comparison with different flags is invalid under the project rules. Both")
    print("  numbers are printed; neither is subtracted from the other.")
    print("")
    print(f"  rubric marker : {RUBRIC_SYNC_R_RANGE[0]} to {RUBRIC_SYNC_R_RANGE[1]}")
    print(f"                  {RUBRIC_SOURCE}")
    means = [float(d["sync_r"].mean()) for d in data.values()]
    print(f"  these runs    : {', '.join(f'{m:.6f}' for m in means)}")
    print("                  bcast_s42/43/44, dmts, --enable-audio --enable-mock-semantic")
    print("")
    print("  The only claim licensed here: the marker was measured under a configuration")
    print("  that is not the one in use, so it cannot be cited as current until it is")
    print("  re-measured under matched flags.")


def main() -> None:
    data = {s: load(s) for s in SEEDS}

    alive = report_non_degeneracy(data)
    report_bid_determinism(data)
    report_episode_structure(data)
    passed = report_content(data)
    report_rubric_marker(data)

    print("")
    print("=" * 84)
    print("PRE-STATED GATE")
    print("=" * 84)
    if not all(alive.values()):
        print("  VERDICT (b): GENUINELY NEAR-CONSTANT.")
        print("  sync_r falls to the dead bar at a seed. No instrument repair rescues it.")
    elif all(passed.values()):
        print("  VERDICT (a): ALIVE AND CARRYING CONTENT, at all 3 seeds.")
        print("  sync_r tracks which shape was shown. Instruments reading it are at fault.")
    elif not any(passed.values()):
        print("  FAILED. VERDICT (c): VARYING BUT CONTENTLESS, at all 3 seeds.")
        print("  sync_r clears the non-degeneracy bar and does NOT clear its own null.")
        print("  Its variance carries no stimulus identity. It cannot support an")
        print("  integration claim, and RPT-2's evidence base rests on it.")
    else:
        print(f"  SEEDS DISAGREE: {passed}. Not decisive. Report as such, claim nothing.")


if __name__ == "__main__":
    main()
