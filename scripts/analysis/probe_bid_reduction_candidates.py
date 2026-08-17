"""
Which reduction of `kl_map` should compute the vision bid?

`sensory_tectum.py:443-444` builds a 262,144-element prediction-error map and reduces it
with `kl_map.sum()`, an EQUAL-WEIGHT reduction. The sum is dominated by a constant offset
(measured mean 344,166, sd 30,370, coefficient of variation 0.088), and `torch.tanh` then
returns exactly 1.0 at every step. The bid is a constant and the workspace competition
downstream never switches module.

`docs/results/klmap_phase_information_2026_08.md` established that the map itself carries
stimulus identity at close to its own ceiling. This probe asks which scalar reduction of
that map makes the best bid.

WHAT A BID CAN AND CANNOT DO. A bid is a scalar. It cannot carry 6-way shape identity in
any useful sense, and no reduction here is claimed to. Stimulus identity reaches the
workspace through the payload, not the bid. What a reduction CAN do is make the bid vary
with task-relevant structure instead of tracking a constant offset. That is what is
scored.

CANDIDATES, all unsupervised, because no training signal for salience exists:

    S0  tanh(kl_map.sum())                          current baseline, equal weight
    S1  tanh(kl_map.mean())                         equal weight, de-saturates only
    S2  per-element running z-score -> sigmoid(mean(z))       non-uniform
    S3  per-element running z-score -> tanh(mean(abs(z)))     non-uniform
    S4  per-element running z-score -> sigmoid(mean(top 1%))  non-uniform

S2 and S4 use `sigmoid` rather than `tanh` because a z-score is SIGNED and `tanh` of a
signed quantity goes negative, which violates the bid range requirement below. A first
formulation of S4 divided the top-1 percent mean by the overall mean, which gives a ratio
far outside `tanh`'s responsive range and pinned it at 1.0. Both were corrected before the
scoring run. Neither correction was motivated by how a candidate scored on the outcome
metric; both fix violations of clause 1 and 2, which were stated in advance.

S1 is included because it is the cheap option and has to be beaten on evidence rather
than dismissed. S0 is the baseline every candidate is measured against.

CAUSAL STATISTICS. S2, S3 and S4 need per-element running statistics. They are computed
as an EMA over PAST STEPS ONLY, matching what the live model could compute. A full-sample
variant is reported beside it as an upper bound, because a full-sample statistic is not
implementable in a forward pass and must not be what the decision rests on.

PRE-STATED GATE, written before any number from this probe was read. A candidate is
RECOMMENDED only if ALL THREE hold at ALL 3 seeds:

    1. NON-DEGENERATE   >= 100 distinct bid values over the run, and the squashed bid
                        spans >= 0.2. Clause 1 of the acceptance bar, enforced in code.
    2. COMPARABLE       the squashed bid lies inside [0, 1] and is pinned at neither end
                        (no more than 1 percent of steps within 1e-6 of either bound).
    3. BEATS BASELINE   eta-squared over stimulus shape exceeds the 95th percentile of
                        its OWN permutation null AND exceeds S0's eta-squared.

NO CANDIDATE PASSES is a live outcome. If none clears the gate, nothing is implemented.

Scoring uses eta-squared, the fraction of the scalar's variance explained by
`sample_shape`, restricted to sample-phase steps. A 1-D scalar cannot classify 6 ways, so
variance-explained is the fair measure. The null shuffles shape labels ACROSS TRIALS and
never across steps, because shape is constant for all ~20 steps of a trial and shuffling
steps would destroy that structure and produce a falsely low floor.

Read-only. No training run, no checkpoint written, no model modified.

Run:
    python -m scripts.analysis.probe_bid_reduction_candidates \\
        --load-tectum runs/gate_ckpt_s42/tectum.pt --episodes 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.probe_perception_decodability import _build_components
from scripts.training.train_rlhf import frame_to_tensor
from simulations.environments.dmts_env import DMTSEnv

SEEDS = (42, 43, 44)
EMA_ALPHA = 0.01            # per-element running statistics, causal
TOPK_FRACTION = 0.01        # S4 keeps the largest 1 percent of elements
N_PERMUTATIONS = 100
NULL_PERCENTILE = 95

# Pre-stated gate thresholds.
MIN_DISTINCT = 100
MIN_SPAN = 0.2
PINNED_TOL = 1e-6
MAX_PINNED_FRACTION = 0.01

CANDIDATES = ("S0", "S1", "S2", "S3", "S4")
LABELS = {
    "S0": "S0 tanh(sum)          current baseline, equal weight",
    "S1": "S1 tanh(mean)         equal weight, de-saturates only",
    "S2": "S2 sigmoid(mean z)    per-element running z, non-uniform",
    "S3": "S3 tanh(mean |z|)     per-element running z, non-uniform",
    "S4": "S4 sigmoid(top 1% z)  largest z-scores only, non-uniform",
}


class RunningStats:
    """Causal per-element EMA mean and variance.

    Updated AFTER the value for the current step is read, so the statistic at step t is a
    function of steps < t only. That ordering is what makes the offline number match what
    a forward pass could compute.
    """

    def __init__(self, shape, alpha: float = EMA_ALPHA):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.alpha = alpha
        self.seen = 0

    def z(self, x: torch.Tensor) -> torch.Tensor:
        if self.seen == 0:
            return torch.zeros_like(x)
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

    def update(self, x: torch.Tensor) -> None:
        if self.seen == 0:
            self.mean = x.clone()
            self.var = torch.ones_like(x)
        else:
            delta = x - self.mean
            self.mean = self.mean + self.alpha * delta
            self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)
        self.seen += 1


def reductions(kl_map: torch.Tensor, stats: RunningStats, topk: int) -> dict:
    """Every candidate scalar for one step. `stats` is read before it is updated."""
    flat = kl_map.reshape(-1)
    z = stats.z(flat)
    out = {
        "S0": float(torch.tanh(flat.sum())),
        "S1": float(torch.tanh(flat.mean())),
        # sigmoid, not tanh: z is signed and a bid may not go negative.
        "S2": float(torch.sigmoid(z.mean())),
        "S3": float(torch.tanh(z.abs().mean())),
        "S4": float(torch.sigmoid(torch.topk(z, topk).values.mean())),
    }
    stats.update(flat)
    return out


def collect(config, tectum, episodes: int, seed: int, max_steps: int = 200) -> dict:
    series = {c: [] for c in CANDIDATES}
    shapes, phases, trials = [], [], []
    stats = None
    env = DMTSEnv(num_trials=20)
    with torch.no_grad():
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            if hasattr(tectum, "reset_state"):
                tectum.reset_state(1)
            done, steps = False, 0
            while not done and steps < max_steps:
                frame = frame_to_tensor(obs, config["device"])
                audio = torch.zeros(1, config["tectum_feature_dim"], 2,
                                    device=config["device"])
                tectum(frame, audio)
                post, prior = tectum._last_post_logits, tectum._last_prior_logits
                var = torch.exp(tectum.rssm.cont_logvar)
                kl_map = 0.5 * (post - prior) ** 2 / (var + 1e-8)

                if stats is None:
                    stats = RunningStats(kl_map.numel())
                    topk = max(1, int(kl_map.numel() * TOPK_FRACTION))
                vals = reductions(kl_map, stats, topk)
                for c in CANDIDATES:
                    series[c].append(vals[c])
                shapes.append(info.get("sample_shape"))
                phases.append(info.get("phase"))
                trials.append(f"{ep}:{info.get('trial')}")

                obs, _, term, trunc, info = env.step(0)
                done = term or trunc
                steps += 1
    return {
        "series": {c: np.asarray(v, dtype=np.float64) for c, v in series.items()},
        "shape": np.asarray(shapes),
        "phase": np.asarray(phases),
        "trial": np.asarray(trials),
    }


def eta_squared(x: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of the scalar's variance explained by group membership."""
    total = float(((x - x.mean()) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for g in np.unique(groups):
        xs = x[groups == g]
        between += len(xs) * (xs.mean() - x.mean()) ** 2
    return float(between / total)


def permutation_null(x: np.ndarray, groups: np.ndarray, trials: np.ndarray,
                     seed: int) -> np.ndarray:
    """Shuffle shape labels ACROSS TRIALS, never across steps.

    Shape is constant for all ~20 steps of a trial. Shuffling steps would break that and
    give a floor far below anything achievable, which would make every candidate look
    significant.
    """
    rng = np.random.default_rng(seed)
    uniq_trials = np.unique(trials)
    trial_label = {t: groups[trials == t][0] for t in uniq_trials}
    labels = np.array([trial_label[t] for t in uniq_trials])
    out = []
    for _ in range(N_PERMUTATIONS):
        permuted = rng.permutation(labels)
        mapping = dict(zip(uniq_trials, permuted))
        out.append(eta_squared(x, np.array([mapping[t] for t in trials])))
    return np.asarray(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-tectum", default="runs/gate_ckpt_s42/tectum.pt")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--latent-mode", default="continuous",
                        choices=["discrete", "continuous"])
    parser.add_argument("--capsule-workspace-source", default="all_levels",
                        choices=["final", "all_levels"])
    args = parser.parse_args()

    per_seed = {}
    for seed in SEEDS:
        cfg, tectum, *_ = _build_components(
            "dmts", action_dim=5, seed=seed, mock_semantic=False,
            load_tectum=args.load_tectum, latent_mode=args.latent_mode,
            capsule_workspace_source=args.capsule_workspace_source)
        per_seed[seed] = collect(cfg, tectum, args.episodes, seed)

    print("\n" + "=" * 84)
    print("CLAUSE 1 and 2: is the bid non-degenerate and comparable?")
    print("=" * 84)
    print(f"  Pre-stated: >= {MIN_DISTINCT} distinct values, span >= {MIN_SPAN}, "
          f"inside [0,1], <= {MAX_PINNED_FRACTION:.0%} pinned at a bound.\n")
    shape_ok = {c: {} for c in CANDIDATES}
    for seed, d in per_seed.items():
        print(f"  seed {seed}  n={len(d['shape'])}")
        for c in CANDIDATES:
            x = d["series"][c]
            distinct, span = len(np.unique(x)), float(x.max() - x.min())
            pinned = float(np.mean((x <= PINNED_TOL) | (x >= 1.0 - PINNED_TOL)))
            ok = (distinct >= MIN_DISTINCT and span >= MIN_SPAN
                  and x.min() >= 0.0 and x.max() <= 1.0
                  and pinned <= MAX_PINNED_FRACTION)
            shape_ok[c][seed] = ok
            print(f"    {LABELS[c]}  min={x.min():.6f} max={x.max():.6f} "
                  f"distinct={distinct:>5} span={span:.4f} pinned={pinned:.1%} "
                  f"{'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 84)
    print("CLAUSE 3: does the bid track stimulus shape more than the baseline?")
    print("=" * 84)
    print(f"  eta-squared over `sample_shape`, sample-phase steps only.")
    print(f"  Null = {N_PERMUTATIONS} shuffles of shape ACROSS TRIALS; "
          f"bar is the {NULL_PERCENTILE}th percentile.\n")
    eta_ok = {c: {} for c in CANDIDATES}
    for seed, d in per_seed.items():
        m = d["phase"] == "sample"
        groups, trials = d["shape"][m], d["trial"][m]
        base = eta_squared(d["series"]["S0"][m], groups)
        print(f"  seed {seed}  n={int(m.sum())}  trials={len(np.unique(trials))}  "
              f"S0 baseline eta2={base:.4f}")
        for c in CANDIDATES:
            x = d["series"][c][m]
            eta = eta_squared(x, groups)
            null = permutation_null(x, groups, trials, seed)
            bar = float(np.percentile(null, NULL_PERCENTILE))
            ok = (eta > bar) and (eta > base if c != "S0" else True)
            eta_ok[c][seed] = ok
            print(f"    {LABELS[c]}  eta2={eta:.4f}  null p{NULL_PERCENTILE}={bar:.4f}  "
                  f"{'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 84)
    print("PRE-STATED GATE")
    print("=" * 84)
    recommended = []
    for c in CANDIDATES:
        if c == "S0":
            continue
        all_ok = all(shape_ok[c].values()) and all(eta_ok[c].values())
        n_shape = sum(shape_ok[c].values())
        n_eta = sum(eta_ok[c].values())
        print(f"  {LABELS[c]}  non-degenerate {n_shape}/3  beats-baseline {n_eta}/3  "
              f"{'RECOMMENDED' if all_ok else 'rejected'}")
        if all_ok:
            recommended.append(c)

    if not recommended:
        print("\n  VERDICT: NO CANDIDATE PASSES.")
        print("  Nothing is implemented. The equal-weight sum is not replaced on this")
        print("  evidence, and the model is not touched.")
    else:
        print(f"\n  VERDICT: {len(recommended)} candidate(s) pass: {recommended}")
        print("  A passing candidate is a recommendation for a DEFAULT-OFF flag, not a")
        print("  default flip. Phase-tracking selection is still NOT demonstrated: step")
        print("  index alone decodes sample versus delay at 1.0000.")


if __name__ == "__main__":
    main()
