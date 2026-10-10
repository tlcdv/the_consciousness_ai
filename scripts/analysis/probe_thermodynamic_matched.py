"""Readout trained on channel outputs: does the encoding gain survive a fair readout?

Background. docs/results/thermodynamic_encoding_2026_10.md found that rotating each vector into its
principal axes and scaling the axes before the channel cut the events needed to reach a stated accuracy
by 16 to 256 times. In that study every readout was fitted on CLEAN training vectors, with a penalty
chosen on clean data. Such a readout can be brittle to decoder noise, and it hurts the plain encoding
most, because the plain encoding keeps class information in low-variance directions. So the gain may be
smaller with a readout that has seen the noise. This probe tests that. It is a software emulation. It
says nothing about hardware, consciousness or Phi.

TWO TRAINING REGIMES, for each encoding arm (plain, pca0, half, full, as in the encoding probe), each
channel and each dose T:
    clean-trained    the readout of the encoding probe, fitted on clean training vectors, penalty from
                     5-fold grouped cross validation on clean data. It is applied to noisy test vectors.
    matched-trained  the readout is fitted on training vectors that went through the SAME channel at the
                     SAME dose T. Each training row is sent through the channel twice (two independent
                     noisy copies, same trial id). The penalty comes from 5-fold cross validation grouped
                     by trial on the noisy training copies. It is applied to noisy test vectors.
Both regimes score the same noisy test vectors, so they are paired. Everything else is as in the
encoding probe: the 30 cached recordings (models 42, 43, 44; stimulus draws 300 to 309; 1200 trials per
model), outer folds seeded the same way, REPEATS = 2 noisy test copies averaged, LADDER = (4, ..., 16384),
penalty grid PENALTIES_LOW, the sigmoid map, exact binomial counts. Both channels are run, p-bit and
Poisson (spike scale 0.2, an assumption). The p-bit channel is the primary one.

PREDICTIONS, written before the run, with no thresholds.
  Q1. Matched training raises accuracy at low and middle doses for every arm, and most for plain, whose
      clean-trained readout is the most brittle.
  Q2. The equalisation advantage over plain shrinks under matched training in budget terms. I expect the
      factor of 16 to 256 to fall. I do not know whether it falls below 2, or whether it disappears.
  Q3. At very low Poisson doses (spikes below one per dimension) matched training may be WORSE than clean training,
      because two noisy copies of 480 rows are too little data to fit a readout. The p-bit channel should not show this.
      Matched training cannot raise an arm above its clean ceiling by much, so the lower ceiling of the
      rotated arms for workspace_broadcast stays.

PILOT NOTE. A pilot on one new recording (model 44, draw 999, not part of the data, discarded) ran the code
end to end in 7 s per recording. That made the Poisson channel cheap, so it was added to the plan before
the data run. On the pilot, matched training helped the plain arm on workspace_broadcast (0.61 to 0.67 at
the top dose) and did not rescue the plain arm on model 44 tectum_content and z_state. No arm, prediction
or report item was changed after the pilot.

WHAT IS REPORTED, fixed before the run, no pass line.
  1. Accuracy A(T) per arm, channel and regime, and the paired gain of matched over clean-trained with a 95%
     trial bootstrap interval.
  2. Under matched training: the paired difference of each arm to plain, with 95% intervals.
  3. T90* and T95* per arm and regime: the smallest dose at which accuracy reaches 90% and 95% of the
     above-chance accuracy of the plain arm's CLEAN readout on clean vectors (rho*), as a point estimate
     and as the smallest dose where the lower 95% bound reaches it. "none" means above 16384.
  4. The budget saving of each arm over plain, T95*(plain) / T95*(arm), for each regime, where both
     are defined, beside the clean-trained values.
  5. rho* only where the lower bound of the plain clean accuracy exceeds 1/6 + 0.05.
Nothing is pooled across models. A loss of the advantage is a result, and so is a gain that stays.

Usage:
    python -m scripts.analysis.probe_thermodynamic_matched evaluate
    python -m scripts.analysis.probe_thermodynamic_matched summarise --output-json summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis.probe_thermodynamic_cost import Recording, make_folds, normalise  # noqa: E402
from scripts.analysis.probe_thermodynamic_encoding import (  # noqa: E402
    ARMS,
    CHANNELS,
    PENALTIES_LOW,
    binomial_estimates,
    encode_arm,
    paired_difference,
    resample_index,
    rho_star,
    score_trials,
)
from scripts.analysis.probe_thermodynamic_fidelity import (  # noqa: E402
    LADDER,
    N_CLASSES,
    REPEATS,
    Readout,
    estimable,
    first_dose,
)

COPIES = 2
REGIMES = ("clean-trained", "matched")
STATUS = "UNPROVEN"


# ----------------------------------------------------------------------------- fast penalty choice


def select_penalty_fast(x: torch.Tensor, y: torch.Tensor, trials: torch.Tensor, seed: int,
                        penalties: Sequence[float] = PENALTIES_LOW) -> float:
    """The penalty of probe_thermodynamic_fidelity.select_penalty, computed from one SVD per inner fold.

    Ridge weights for centred features are V diag(s / (s^2 + lambda)) U^T Yc, so one SVD serves every penalty.
    Ties go to the larger penalty.
    """
    onehot = torch.nn.functional.one_hot(y, N_CLASSES).to(x.dtype)
    pens = torch.tensor(list(penalties), dtype=x.dtype)
    errors = torch.zeros(len(penalties), dtype=torch.float64)
    for held in make_folds(trials, seed):
        xt, xh, yt = x[~held], x[held], onehot[~held]
        mean_x, mean_y = xt.mean(0), yt.mean(0)
        u, s, vh = torch.linalg.svd(xt - mean_x, full_matrices=False)
        target = u.T @ (yt - mean_y)
        shrink = s[None, :] / (s[None, :] ** 2 + pens[:, None])
        logits = torch.einsum("hr,pr,rc->phc", (xh - mean_x) @ vh.T, shrink, target) + mean_y
        errors += (logits.argmax(-1) != y[held][None, :]).sum(1).to(torch.float64)
    best, best_error = penalties[-1], float("inf")
    for penalty, error in zip(penalties, errors.tolist()):
        if error <= best_error:
            best, best_error = penalty, error
    return best


# ------------------------------------------------------------------------------------ one recording


def evaluate_stream(z: torch.Tensor, labels: torch.Tensor, trials: torch.Tensor, seed: int) -> Dict[str, object]:
    """Per-trial accuracy for every arm, channel, dose and training regime of one stream."""
    hits: Dict[str, Dict[int, List[float]]] = {}
    for k, held in enumerate(make_folds(trials, seed)):
        ztr, zte = normalise(z[~held], z[held])
        ytr, ttr = labels[~held], trials[~held]
        for a, (arm, alpha) in enumerate(ARMS.items()):
            utr, ute = encode_arm(ztr, zte, alpha)
            xtr, xte = torch.sigmoid(utr), torch.sigmoid(ute)
            clean_readout = Readout(xtr, ytr, select_penalty_fast(xtr, ytr, ttr, seed + k))
            score_trials(hits, f"{arm}|clean", clean_readout.predict(xte), held, labels, trials)
            for c, channel in enumerate(CHANNELS):
                score_channel(hits, arm, channel, (xtr, xte), (ytr, ttr), clean_readout, held, (labels, trials),
                              100000 * seed + 1000 * k + 100 * a + 20 * c, seed + k)
    return {"trial_accuracy": {name: [float(np.mean(c[t])) for t in sorted(c)] for name, c in hits.items()}}


def score_channel(hits, arm, channel, features, train_labels, clean_readout, held, all_labels, base, seed) -> None:
    """Score both training regimes at every dose of one channel for one arm and fold."""
    xtr, xte = features
    ytr, ttr = train_labels
    labels, trials = all_labels
    tests = [binomial_estimates(xte, LADDER, torch.Generator().manual_seed(base + 3 * r), channel) for r in range(REPEATS)]
    copies = [binomial_estimates(xtr, LADDER, torch.Generator().manual_seed(base + 10 + c), channel) for c in range(COPIES)]
    for i, dose in enumerate(LADDER):
        x_noisy = torch.cat([c[i] for c in copies])
        y_noisy, t_noisy = ytr.repeat(COPIES), ttr.repeat(COPIES)
        matched = Readout(x_noisy, y_noisy, select_penalty_fast(x_noisy, y_noisy, t_noisy, seed))
        for r in range(REPEATS):
            score_trials(hits, f"{arm}|clean-trained|{channel}|{dose}", clean_readout.predict(tests[r][i]), held, labels, trials)
            score_trials(hits, f"{arm}|matched|{channel}|{dose}", matched.predict(tests[r][i]), held, labels, trials)


def evaluate_recording(rec: Recording, seed: int) -> Dict[str, object]:
    return {name: evaluate_stream(z, rec.labels, rec.trials, seed) for name, z in rec.streams.items()}


# ----------------------------------------------------------------------------------- summaries


def curve(per_trial: Dict[str, np.ndarray], arm: str, regime: str, channel: str, plain_clean: np.ndarray,
          index: np.ndarray) -> Dict[str, object]:
    """Accuracy, paired difference to plain (same regime and channel) and rho* at every dose for one arm."""
    rows = []
    for dose in LADDER:
        a = per_trial[f"{arm}|{regime}|{channel}|{dose}"]
        row = {"accuracy": float(a.mean()), **paired_difference(a, per_trial[f"plain|{regime}|{channel}|{dose}"], index)}
        row.update(rho_star(a, plain_clean, index))
        row["gain_over_clean_trained"] = paired_difference(a, per_trial[f"{arm}|clean-trained|{channel}|{dose}"], index)
        rows.append(row)
    return {"curve": rows, "T90": first_dose(rows, 0.90, "rho"), "T95": first_dose(rows, 0.95, "rho"),
            "T90_lower": first_dose(rows, 0.90, "rho_low"), "T95_lower": first_dose(rows, 0.95, "rho_low")}


def summarise_stream(per_trial: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, object]:
    plain_clean = per_trial["plain|clean"]
    index = resample_index(rng, len(plain_clean))
    clean_low = float(np.quantile(plain_clean[index].mean(1), 0.025))
    out: Dict[str, object] = {"estimable": estimable(clean_low), "clean_low_plain": clean_low, "arms": {}}
    for arm in ARMS:
        entry: Dict[str, object] = {"clean_accuracy": float(per_trial[f"{arm}|clean"].mean())}
        for regime in REGIMES:
            entry[regime] = {channel: curve(per_trial, arm, regime, channel, plain_clean, index) for channel in CHANNELS}
        out["arms"][arm] = entry
    return out


# ------------------------------------------------------------------------------------------ IO


def matched_path(out_dir: Path, model: int, draw: int) -> Path:
    return out_dir / f"mat_m{model}_s{draw}.json"


def progress(message: str, started: float) -> None:
    print(f"[{time.time() - started:7.0f} s] {message}", file=sys.stderr, flush=True)


def command_evaluate(args: argparse.Namespace) -> None:
    out_dir, started = Path(args.out_dir), time.time()
    for path in sorted(out_dir.glob("rec_m*_s*.pt")):
        model, draw = [int(p[1:]) for p in path.stem[4:].split("_")]
        target = matched_path(out_dir, model, draw)
        if target.exists() or (args.models and model not in args.models):
            continue
        data = torch.load(path, weights_only=False)
        rec = Recording(data["streams"], data["labels"], data["trials"])
        target.write_text(json.dumps(evaluate_recording(rec, seed=1000 * model + draw)))
        progress(f"matched model {model} draw {draw}", started)


def load_pooled(out_dir: Path) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    pooled: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    for path in sorted(out_dir.glob("mat_m*_s*.json")):
        model = int(path.stem[5:].split("_")[0])
        for stream, res in json.loads(path.read_text()).items():
            slot = pooled.setdefault(model, {}).setdefault(stream, {})
            for name, values in res["trial_accuracy"].items():
                slot.setdefault(name, []).extend(values)
    return {m: {s: {c: np.array(v) for c, v in conds.items()} for s, conds in streams.items()} for m, streams in pooled.items()}


def command_summarise(args: argparse.Namespace) -> None:
    pooled = load_pooled(Path(args.out_dir))
    rng = np.random.default_rng(0)
    summary = {"status": STATUS, "ladder": list(LADDER), "arms": list(ARMS), "regimes": list(REGIMES), "models": {}}
    for model in sorted(pooled):
        summary["models"][str(model)] = {s: summarise_stream(c, rng) for s, c in pooled[model].items()}
    print_summary(summary)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))


def show(value: Optional[int]) -> str:
    return "none" if value is None else str(value)


def print_summary(summary: Dict[str, object]) -> None:
    print(f"matched-readout study, instruments {summary['status']}, doses {summary['ladder']}")
    for model, streams in summary["models"].items():
        for stream, e in streams.items():
            if not e["estimable"]:
                print(f"model {model} {stream}: NOT ESTIMABLE")
                continue
            print(f"model {model} {stream}")
            for arm, a in e["arms"].items():
                for regime in REGIMES:
                    for channel in CHANNELS:
                        c = a[regime][channel]
                        accuracy = " ".join(f"{r['accuracy']:.2f}" for r in c["curve"])
                        print(f"   {arm:6s} {regime:13s} {channel:7s} clean {a['clean_accuracy']:.3f}  A(T) {accuracy}  "
                              f"T90* {show(c['T90'])} ({show(c['T90_lower'])})  T95* {show(c['T95'])} ({show(c['T95_lower'])})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "summarise"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", default="runs/thermo_fidelity")
        if name == "evaluate":
            p.add_argument("--models", type=int, nargs="+")
        else:
            p.add_argument("--output-json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    {"evaluate": command_evaluate, "summarise": command_summarise}[args.command](args)


if __name__ == "__main__":
    main()
