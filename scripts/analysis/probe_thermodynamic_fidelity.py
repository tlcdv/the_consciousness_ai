"""Fidelity of recorded representations through a stochastic spike channel: a dose response.

Question. How many stochastic events per dimension does a recorded vector need before the
verified readout (a tuned linear readout of the stimulus class) works on the channel output as
well as on the clean vector? The answer is a curve of readable class information against
budget, with intervals. It is NOT a pass or fail test. Earlier pass/fail gates (docs/results/
thermodynamic_transduction_*.md) were unstable, and their reference readout used a ridge
penalty of 100 that was never tuned. This probe replaces the reference with a tuned one.
It is a software emulation. It says nothing about any hardware, about consciousness or about Phi.

STEP 1, the reference. For each stream, model and stimulus draw, a ridge readout of the 6-class
sample_shape is fitted on x = sigmoid(z) of the training rows, where z is the training-fold
normalised vector of probe_thermodynamic_cost.normalise (per-dimension z score for 256-D
streams, training-fold PCA to 256 components for larger streams). Features and the one-hot
target are centred on the training rows. The ridge penalty is chosen inside each outer
training fold by 5-fold cross validation grouped by trial, from PENALTIES, with ties going to
the larger penalty. Outer folds are 5-fold, grouped by trial. A legacy arm, the old probe's
reference (ridge on z, fixed penalty 100), is reported beside it for comparison.

STEP 2, the channel. The test vector x is sent through one of two channels, and the SAME
readout (fitted on clean training rows only) is applied to the decoded vector.
    poisson  Bernoulli bins with spike probability F_DT * x per bin, through
             snn_bridge.rate_encode; the decoder is count / (T * F_DT), which equals rate_decode.
    pbit     T samples of an independent p-bit per dimension with bias h = atanh(2x - 1)
             (beta = 1), using the p-bit update rule of p_bit_emulator; the decoder is
             (1 + mean of the +-1 samples) / 2. A test checks the rule against BlockGibbsSampler.
Doses are T in LADDER = (4, 16, 64, 256, 1024, 4096, 16384), nested prefixes of one random stream.
Each condition is repeated REPEATS = 2 times with different random streams and the correct
counts are averaged. Assumptions, not measurements: F_DT = 0.2, the sigmoid map, the use of the
normalised <=256-D vector, and training the readout on clean vectors only.

PILOT NOTE. A pilot on model 42, draw 999 (not part of the data, discarded) showed that the selected
penalty sat at the lower edge of the first grid (0.03) and that several curves had not reached
rho 0.9 at the highest dose (4096). Both grids were extended before the data run. Nothing else
was changed after the pilot.

DATA. Models 42, 43, 44 (the three capfix checkpoints). Stimulus draws 300 to 309 (ten per
model, unused before), 120 DMTS trials per draw, phase sample, 6 classes. Each trial is one
unit of analysis. Rows of a trial are scored together.

WHAT IS REPORTED, fixed before the run, with no pass line.
  1. Clean accuracy with the tuned readout, per model and stream, with a 95% bootstrap
     interval over trials, beside the legacy value and the median selected penalty, and the
     earlier project numbers for comparison (broadcast 0.69 to 0.77, docs/results/
     broadcast_geometry_2026_08.md). Descriptive only.
  2. Retained information rho(T) = (A_T - 1/6) / (A_clean - 1/6), per model, stream, channel
     and dose, with a 95% bootstrap interval (2000 resamples of trials, the same resample for
     A_T and A_clean).
  3. T90 and T95, the smallest dose whose point estimate of rho reaches 0.90 and 0.95, and the
     smallest dose whose lower interval bound reaches it. "none" means above 16384.
  4. rho is reported only where the lower bound of A_clean exceeds 1/6 + 0.05. Otherwise the
     stream is listed as not estimable for that model.
  5. Events per decision: bins (poisson) or samples (pbit) per dimension, and the measured mean
     spike count per dimension for poisson. Energy is derived only on request from a stated
     assumed energy per event (--joules-per-event). It is an assumption, not a measurement.
Nothing is pooled across models. A result from three checkpoints is a result about three
checkpoints. Model 44 has not carried the class in tectum_content or z_state in earlier runs.

Usage:
    python -m scripts.analysis.probe_thermodynamic_fidelity record --models 42 43 44 --draws 300 301
    python -m scripts.analysis.probe_thermodynamic_fidelity evaluate
    python -m scripts.analysis.probe_thermodynamic_fidelity summarise --output-json summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.thermodynamic.interfaces.snn_bridge import rate_encode  # noqa: E402
from scripts.analysis.probe_thermodynamic_cost import (  # noqa: E402
    Recording,
    load_recording,
    make_folds,
    normalise,
    ridge_weights,
)

LADDER = (4, 16, 64, 256, 1024, 4096, 16384)
F_MAX_HZ, DT_S = 200.0, 1e-3
F_DT = F_MAX_HZ * DT_S
PENALTIES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
N_CLASSES = 6
CHANCE = 1.0 / N_CLASSES
REPEATS = 2
CHUNK = 256
EPS = 1e-4
CHANNELS = ("poisson", "pbit")
MODELS = {42: "runs/capfix_alllevels/tectum.pt", 43: "runs/capfix_seed43/tectum.pt",
          44: "runs/capfix_seed44/tectum.pt"}
ESTIMABLE_MARGIN = 0.05
BOOTSTRAPS = 2000
STATUS = "UNPROVEN"


# ----------------------------------------------------------------------------------- readout


class Readout:
    """Ridge readout on centred features and centred one-hot targets."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, penalty: float):
        self.mean_x = x.mean(0)
        onehot = torch.nn.functional.one_hot(y, N_CLASSES).to(x.dtype)
        self.mean_y = onehot.mean(0)
        xc = x - self.mean_x
        gram = xc.T @ xc + penalty * torch.eye(x.shape[1], dtype=x.dtype)
        self.weights = torch.linalg.solve(gram, xc.T @ (onehot - self.mean_y))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean_x) @ self.weights + self.mean_y).argmax(1)


def select_penalty(x: torch.Tensor, y: torch.Tensor, trials: torch.Tensor, seed: int,
                   penalties: Sequence[float] = PENALTIES) -> float:
    """Penalty with the lowest 5-fold grouped cross validation error on the rows given. Ties go larger."""
    folds = make_folds(trials, seed)
    best, best_error = penalties[-1], float("inf")
    for penalty in penalties:
        wrong = 0
        for held in folds:
            wrong += int((Readout(x[~held], y[~held], penalty).predict(x[held]) != y[held]).sum())
        if wrong / len(y) <= best_error:
            best, best_error = penalty, wrong / len(y)
    return best


# ----------------------------------------------------------------------------------- channels


def poisson_estimates(x: torch.Tensor, ladder: Sequence[int], generator: torch.Generator) -> Tuple[List[torch.Tensor], List[float]]:
    """Decoded vectors and mean spike counts per dimension, at nested doses of one random stream."""
    total, done = torch.zeros_like(x), 0
    estimates, spikes = [], []
    for dose in ladder:
        left = dose - done
        while left > 0:
            step = min(left, CHUNK)
            total += rate_encode(x, step, F_MAX_HZ, DT_S, generator).sum(0)
            left -= step
        done = dose
        estimates.append(total / (dose * F_DT))
        spikes.append(float(total.mean()))
    return estimates, spikes


def pbit_estimates(x: torch.Tensor, ladder: Sequence[int], generator: torch.Generator) -> Tuple[List[torch.Tensor], List[float]]:
    """Decoded vectors from independent p-bits with P(+1) = x, at nested doses of one random stream.

    The update rule is the one in p_bit_emulator: m = +1 when tanh(beta * h) > r, r uniform on (-1, 1).
    """
    field = torch.atanh((2 * x - 1).clamp(-1 + EPS, 1 - EPS))
    drive = torch.tanh(field)
    total, done = torch.zeros_like(x), 0
    estimates = []
    for dose in ladder:
        left = dose - done
        while left > 0:
            step = min(left, CHUNK)
            noise = 2 * torch.rand((step,) + tuple(x.shape), generator=generator, dtype=x.dtype) - 1
            total += torch.where(drive > noise, 1.0, -1.0).to(x.dtype).sum(0)
            left -= step
        done = dose
        estimates.append((1 + total / dose) / 2)
    return estimates, [float(dose) for dose in ladder]


CHANNEL_FUNCTIONS = {"poisson": poisson_estimates, "pbit": pbit_estimates}


# ------------------------------------------------------------------------------- one recording


def trial_accuracy(correct: Dict[int, List[float]]) -> List[float]:
    """Mean correct fraction per trial, ordered by trial id."""
    return [float(np.mean(correct[t])) for t in sorted(correct)]


def evaluate_stream(z: torch.Tensor, labels: torch.Tensor, trials: torch.Tensor, seed: int) -> Dict[str, object]:
    """Per-trial accuracy for the clean, legacy and channel conditions of one stream."""
    hits: Dict[str, Dict[int, List[float]]] = {}
    penalties, spike_means = [], {}

    def score(name: str, pred: torch.Tensor, held: torch.Tensor) -> None:
        for t in trials[held].unique():
            rows = held & (trials == t)
            hits.setdefault(name, {}).setdefault(int(t), []).append(float((pred[rows[held]] == labels[rows]).float().mean()))

    for k, held in enumerate(make_folds(trials, seed)):
        ztr, zte = normalise(z[~held], z[held])
        xtr, xte = torch.sigmoid(ztr), torch.sigmoid(zte)
        ytr = labels[~held]
        penalty = select_penalty(xtr, ytr, trials[~held], seed + k)
        penalties.append(penalty)
        readout = Readout(xtr, ytr, penalty)
        score("clean", readout.predict(xte), held)
        score("legacy100", (zte @ ridge_weights(ztr, ytr, N_CLASSES)).argmax(1), held)
        for channel, function in CHANNEL_FUNCTIONS.items():
            for repeat in range(REPEATS):
                generator = torch.Generator().manual_seed(100000 * seed + 1000 * k + 10 * repeat + len(channel))
                estimates, spikes = function(xte, LADDER, generator)
                spike_means.setdefault(channel, []).append(spikes)
                for dose, estimate in zip(LADDER, estimates):
                    score(f"{channel}:{dose}", readout.predict(estimate), held)
    return {"trial_accuracy": {name: trial_accuracy(c) for name, c in hits.items()},
            "penalties": penalties,
            "mean_spikes_per_dim": {c: np.mean(v, axis=0).tolist() for c, v in spike_means.items()}}


def evaluate_recording(rec: Recording, seed: int) -> Dict[str, object]:
    return {name: evaluate_stream(z, rec.labels, rec.trials, seed) for name, z in rec.streams.items()}


# ----------------------------------------------------------------------------------- statistics


def bootstrap_rho(clean: np.ndarray, dosed: np.ndarray, rng: np.random.Generator, n: int = BOOTSTRAPS) -> Dict[str, float]:
    """rho = (A_T - 1/6) / (A_clean - 1/6) with a trial bootstrap. clean and dosed are per-trial accuracies."""
    a_clean, a_dose = float(clean.mean()), float(dosed.mean())
    index = rng.integers(0, len(clean), size=(n, len(clean)))
    c, d = clean[index].mean(1), dosed[index].mean(1)
    clean_low = float(np.quantile(c, 0.025))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (d - CHANCE) / (c - CHANCE)
    rho = rho[np.isfinite(rho)]
    return {"accuracy": a_dose, "clean": a_clean, "clean_low": clean_low,
            "rho": (a_dose - CHANCE) / (a_clean - CHANCE) if a_clean != CHANCE else float("nan"),
            "rho_low": float(np.quantile(rho, 0.025)), "rho_high": float(np.quantile(rho, 0.975))}


def estimable(clean_low: float) -> bool:
    return clean_low > CHANCE + ESTIMABLE_MARGIN


def first_dose(rows: Sequence[Dict[str, float]], level: float, key: str) -> Optional[int]:
    """Smallest dose whose `key` reaches `level`, or None."""
    for dose, row in zip(LADDER, rows):
        if row[key] >= level:
            return dose
    return None


def summarise_model_stream(per_trial: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, object]:
    clean = per_trial["clean"]
    base = bootstrap_rho(clean, clean, rng)
    out: Dict[str, object] = {"clean_accuracy": base["clean"], "clean_low": base["clean_low"],
                              "legacy100": float(per_trial["legacy100"].mean()), "estimable": estimable(base["clean_low"]),
                              "n_trials": int(len(clean))}
    if not out["estimable"]:
        return out
    for channel in CHANNELS:
        rows = [bootstrap_rho(clean, per_trial[f"{channel}:{dose}"], rng) for dose in LADDER]
        out[channel] = {"curve": rows,
                        "T90": first_dose(rows, 0.90, "rho"), "T95": first_dose(rows, 0.95, "rho"),
                        "T90_lower": first_dose(rows, 0.90, "rho_low"), "T95_lower": first_dose(rows, 0.95, "rho_low")}
    return out


# ----------------------------------------------------------------------------------------- IO


def recording_path(out_dir: Path, model: int, draw: int) -> Path:
    return out_dir / f"rec_m{model}_s{draw}.pt"


def result_path(out_dir: Path, model: int, draw: int) -> Path:
    return out_dir / f"res_m{model}_s{draw}.json"


def progress(message: str, started: float) -> None:
    print(f"[{time.time() - started:7.0f} s] {message}", file=sys.stderr, flush=True)


def command_record(args: argparse.Namespace) -> None:
    out_dir, started = Path(args.out_dir), time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        for draw in args.draws:
            path = recording_path(out_dir, model, draw)
            if path.exists():
                continue
            rec = load_recording(MODELS[model], args.episodes, model, "sample", draw)
            torch.save({"streams": rec.streams, "labels": rec.labels, "trials": rec.trials}, path)
            progress(f"recorded model {model} draw {draw}, {len(rec.labels)} rows", started)


def command_evaluate(args: argparse.Namespace) -> None:
    out_dir, started = Path(args.out_dir), time.time()
    for path in sorted(out_dir.glob("rec_m*_s*.pt")):
        model, draw = [int(p[1:]) for p in path.stem[4:].split("_")]
        target = result_path(out_dir, model, draw)
        if target.exists():
            continue
        data = torch.load(path, weights_only=False)
        rec = Recording(data["streams"], data["labels"], data["trials"])
        target.write_text(json.dumps(evaluate_recording(rec, seed=1000 * model + draw)))
        progress(f"evaluated model {model} draw {draw}", started)


def load_pooled(out_dir: Path) -> Tuple[Dict[int, Dict[str, Dict[str, np.ndarray]]], Dict[Tuple[int, str], List[float]]]:
    """Per-trial accuracy pooled over draws, {model: {stream: {condition: array}}}, and the selected penalties."""
    pooled: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    penalties: Dict[Tuple[int, str], List[float]] = {}
    for path in sorted(out_dir.glob("res_m*_s*.json")):
        model = int(path.stem[5:].split("_")[0])
        for stream, res in json.loads(path.read_text()).items():
            slot = pooled.setdefault(model, {}).setdefault(stream, {})
            for name, values in res["trial_accuracy"].items():
                slot.setdefault(name, []).extend(values)
            penalties.setdefault((model, stream), []).extend(res["penalties"])
    arrays = {m: {s: {c: np.array(v) for c, v in conds.items()} for s, conds in streams.items()}
              for m, streams in pooled.items()}
    return arrays, penalties


def command_summarise(args: argparse.Namespace) -> None:
    pooled, penalties = load_pooled(Path(args.out_dir))
    rng = np.random.default_rng(0)
    summary = {"status": STATUS, "ladder": list(LADDER), "f_dt": F_DT, "repeats": REPEATS, "models": {}}
    for model in sorted(pooled):
        summary["models"][str(model)] = {}
        for stream, per_trial in pooled[model].items():
            entry = summarise_model_stream(per_trial, rng)
            entry["median_penalty"] = float(np.median(penalties[(model, stream)]))
            summary["models"][str(model)][stream] = entry
    print_summary(summary)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))


def print_summary(summary: Dict[str, object]) -> None:
    print(f"fidelity dose response, instruments {summary['status']}, doses {summary['ladder']}, f*dt {summary['f_dt']}")
    for model, streams in summary["models"].items():
        for stream, e in streams.items():
            head = (f"model {model} {stream:20s} clean {e['clean_accuracy']:.3f} (low {e['clean_low']:.3f}) "
                    f"legacy100 {e['legacy100']:.3f} penalty {e['median_penalty']:g} trials {e['n_trials']}")
            if not e["estimable"]:
                print(head + "  NOT ESTIMABLE")
                continue
            print(head)
            for channel in CHANNELS:
                c = e[channel]
                rho = " ".join(f"{r['rho']:.2f}" for r in c["curve"])
                print(f"     {channel:8s} rho {rho}   T90 {c['T90']} (lower {c['T90_lower']})  T95 {c['T95']} (lower {c['T95_lower']})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "evaluate", "summarise"):
        p = sub.add_parser(name)
        p.add_argument("--out-dir", default="runs/thermo_fidelity")
        if name == "record":
            p.add_argument("--models", type=int, nargs="+", default=[42, 43, 44])
            p.add_argument("--draws", type=int, nargs="+", default=list(range(300, 310)))
            p.add_argument("--episodes", type=int, default=120)
        if name == "summarise":
            p.add_argument("--output-json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    {"record": command_record, "evaluate": command_evaluate, "summarise": command_summarise}[args.command](args)


if __name__ == "__main__":
    main()
