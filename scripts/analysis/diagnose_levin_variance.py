"""
Controlled input-sensitivity probe for the Levin consciousness metrics
(Phase 5 deliverable 4 de-risk, 2026-05-29).

Question: do the 5 LevinConsciousnessMetrics actually respond to input, or are
they inert (near-constant regardless of input)? The Step-1 smoke run showed
collective_intelligence pinned at ~2e-6 across every step. That run was
degenerate (the dark_room agent got stuck, so the broadcast barely changed), so
it could not distinguish "metric is inert" from "input did not vary".

This probe removes that confound: it feeds the HolonicSystem +
LevinConsciousnessEvaluator a batch of deliberately DIVERSE inputs and reports,
per metric, min / max / std / unique-count. A metric whose std is ~0 and whose
unique-count is ~1 across very different inputs is inert BY CONSTRUCTION.

It mirrors the exact call path used in scripts/training/train_rlhf.py run_episode
so the numbers transfer: holonic_system(holon_in) -> evaluate_levin_consciousness
with bioelectric_fields, the holonic output, a rolling past-state history, and
component_states built from broadcast / tectum / gate tensors.

Run:
    python -m scripts.analysis.diagnose_levin_variance
    python -m scripts.analysis.diagnose_levin_variance --trials 64 --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.self_model.holonic_intelligence import HolonicSystem
from models.evaluation.levin_consciousness_metrics import LevinConsciousnessEvaluator

METRIC_KEYS = [
    "bioelectric_complexity",
    "morphological_adaptation",
    "collective_intelligence",
    "goal_directed_behavior",
    "basal_cognition",
]

# A metric is "usable" if it both moves (std above a small floor) and takes more
# than a couple of distinct values across diverse inputs. These thresholds are
# deliberately lenient: the point is to catch near-constant (inert) metrics, not
# to grade quality.
STD_FLOOR = 1e-4
UNIQUE_FLOOR = 5


def _diverse_input(dim: int, i: int, rng: np.random.Generator) -> torch.Tensor:
    """Generate one input of a deliberately different character per index."""
    kind = i % 5
    if kind == 0:
        v = rng.standard_normal(dim)
    elif kind == 1:
        v = rng.uniform(-1.0, 1.0, dim)
    elif kind == 2:  # sparse one-hot-ish
        v = np.zeros(dim)
        v[rng.integers(0, dim)] = rng.uniform(1.0, 5.0)
    elif kind == 3:  # smooth sinusoid
        t = np.linspace(0, rng.uniform(1, 8) * np.pi, dim)
        v = np.sin(t + rng.uniform(0, np.pi))
    else:  # scaled normal (varied magnitude)
        v = rng.standard_normal(dim) * rng.uniform(0.1, 10.0)
    return torch.tensor(v, dtype=torch.float32).reshape(1, dim)


def run_probe(trials: int, seed: int, hidden_size: int, num_holons: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    cfg = {
        "hidden_size": hidden_size,
        "num_holons": num_holons,
        "field_dimension": 128,
        "bioelectric_channels": 8,
        "signaling_layers": 3,
        "gap_junction_heads": 4,
        "gap_junction_dropout": 0.0,
        "integration_heads": 4,
    }
    system = HolonicSystem(cfg)
    system.eval()
    evaluator = LevinConsciousnessEvaluator(cfg)

    history: list[dict] = []
    rows: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}

    with torch.no_grad():
        for i in range(trials):
            holon_in = _diverse_input(hidden_size, i, rng)
            out = system(holon_in)
            current = {"integrated_state": out["integrated_state"].detach()}
            component_states = {
                "broadcast": holon_in,
                "tectum": _diverse_input(hidden_size, i + 1, rng),
                "gate": torch.tensor(rng.uniform(0, 1, 5), dtype=torch.float32),
            }
            res = evaluator.evaluate_levin_consciousness(
                bioelectric_state=out.get("bioelectric_fields", {}),
                holonic_output=out,
                past_states=history,
                current_state=current,
                actions=[], goals=[], outcomes=[],
                component_states=component_states,
            )
            for k in METRIC_KEYS:
                rows[k].append(float(res[k]))
            history.append(current)
            if len(history) > 5:
                del history[0]

    summary = {}
    for k in METRIC_KEYS:
        arr = np.array(rows[k])
        # round to 9 dp before counting uniques so float noise is not mistaken
        # for genuine variation
        uniq = len(np.unique(np.round(arr, 9)))
        summary[k] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std()),
            "unique": uniq,
            "usable": bool(arr.std() > STD_FLOOR and uniq >= UNIQUE_FLOOR),
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-holons", type=int, default=8)
    args = ap.parse_args()

    summary = run_probe(args.trials, args.seed, args.hidden_size, args.num_holons)

    print(f"\nLevin metric input-sensitivity probe "
          f"(trials={args.trials}, seed={args.seed}, "
          f"hidden_size={args.hidden_size}, num_holons={args.num_holons})")
    print(f"usable = std > {STD_FLOOR:g} AND unique >= {UNIQUE_FLOOR}\n")
    header = f"{'metric':32s} {'min':>12s} {'max':>12s} {'std':>12s} {'uniq':>6s}  verdict"
    print(header)
    print("-" * len(header))
    for k in METRIC_KEYS:
        s = summary[k]
        verdict = "USABLE" if s["usable"] else "INERT"
        print(f"{k:32s} {s['min']:12.6g} {s['max']:12.6g} "
              f"{s['std']:12.6g} {s['unique']:6d}  {verdict}")
    print()
    inert = [k for k in METRIC_KEYS if not summary[k]["usable"]]
    print(f"INERT metrics: {inert if inert else 'none'}")


if __name__ == "__main__":
    main()
