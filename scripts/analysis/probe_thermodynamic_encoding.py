"""Variance-equalised encoding: does it cut the event budget of a stochastic spike channel?

Background. In docs/results/thermodynamic_fidelity_2026_10.md the budget needed to keep the class
readable depended on where the class information sits in the variance spectrum of the vector.
A channel with independent noise per dimension removes low-variance directions first. Here the
vector is rotated into its principal axes and each axis is scaled before transduction, so the
channel noise hits every axis alike. Everything else is as in the fidelity probe. It is a software
emulation. It says nothing about hardware, consciousness or Phi.

ARMS, all with the same number of dimensions, so the same events per decision at equal dose T:
    plain  u = z, the encoding of the fidelity probe.
    pca0   u = V^T z, a rotation into the training-fold principal axes (a control: rotation only).
    half   u_k = (V^T z)_k / (var_k + eps)^(1/4), partial equalisation.
    full   u_k = (V^T z)_k / (var_k + eps)^(1/2), full equalisation (whitening with shrinkage).
eps = 0.01 * mean(var_k). Each arm is rescaled by one constant so the mean training variance per
dimension is 1, so every arm has the same average signal power. x = sigmoid(u) is then sent through
the channel, decoded, and read out by a ridge readout fitted on the CLEAN training x of the same arm,
with its penalty chosen by 5-fold cross validation grouped by trial from PENALTIES_LOW (the fidelity
grid extended by two decades below 0.001). Channels, doses and the folds are those of the fidelity
probe: pbit and poisson, LADDER = (4, ..., 16384), REPEATS = 2, outer folds seeded the same way.
Counts are drawn from the exact binomial law of the per-bin process. A test checks the mean and
variance against the closed forms.

DATA. The 30 cached recordings of the fidelity study: models 42, 43, 44, stimulus draws 300 to 309.
Nothing new is recorded. The fidelity study found that the pbit channel needs less budget than the
Poisson channel and that model 44 is weak in tectum_content and z_state. Those results are known.

PREDICTIONS, written before the run, with no thresholds.
  P1. Equalisation lowers the budget most where the class sits in low-variance directions: model 44
      z_state and tectum_content, and model 42 z_state.
  P2. It helps least, or hurts, for obs_map, which is already cheap.
  P3. The pca0 control is close to plain for obs_map and z_state, whose vectors are already principal
      scores, so any change in the other arms is not a rotation effect.
  P4. Full equalisation may hurt where tail axes are noise, because it spends events on them.

WHAT IS REPORTED, fixed before the run, no pass line.
  1. Clean accuracy per arm, per model and stream (95% bootstrap interval over trials).
  2. Accuracy A(T) per arm, channel and dose, and the paired difference to plain with a 95% trial bootstrap interval.
  3. T90* and T95*: the smallest dose at which A_arm(T) reaches 90% and 95% of the PLAIN arm's clean
     above-chance accuracy, rho* = (A_arm(T) - 1/6) / (A_plain,clean - 1/6), as a point estimate and as the
     smallest dose where the lower 95% bound reaches it. "none" means above 16384. Using the plain clean
     accuracy as the common yardstick keeps the arms comparable if their own clean accuracies differ.
  4. rho* is reported only where the lower bound of A_plain,clean exceeds 1/6 + 0.05.
Nothing is pooled across models. This study can show a gain, a loss or no change. All three are results.

PILOT NOTE. A pilot on one new recording (model 43, draw 999, not part of the data, discarded) ran the
code end to end in 58 s per recording and showed large changes for half and full on three streams. It
contradicted P2, because obs_map gained as well. It supported P1. It supported P4 for full on
workspace_broadcast, where accuracy at the top dose stayed far below clean. A likely cause is a readout
fitted on clean vectors with a very small penalty, which is brittle to decoder noise. Nothing in the
arms, the predictions or the report plan was changed after the pilot. P1 to P4 stand as written.

Usage:
    python -m scripts.analysis.probe_thermodynamic_encoding evaluate
    python -m scripts.analysis.probe_thermodynamic_encoding summarise --output-json summary.json
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

from scripts.analysis.probe_thermodynamic_cost import Recording, make_folds, normalise  # noqa: E402
from scripts.analysis.probe_thermodynamic_fidelity import (  # noqa: E402
    CHANCE,
    CHANNELS,
    EPS,
    F_DT,
    LADDER,
    REPEATS,
    Readout,
    estimable,
    first_dose,
    select_penalty,
)

ARMS = {"plain": None, "pca0": 0.0, "half": 0.5, "full": 1.0}
EPS_FRACTION = 0.01
PENALTIES_LOW = (1e-5, 1e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
BOOTSTRAPS = 2000
STATUS = "UNPROVEN"
STREAMS = ("tectum_content", "workspace_broadcast", "obs_map", "z_state")


# --------------------------------------------------------------------------------- encodings


def encode_arm(ztr: torch.Tensor, zte: torch.Tensor, alpha: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate into the training principal axes and scale each by (var + eps)^(-alpha / 2).

    alpha None returns the inputs unchanged. The result is rescaled so the mean training variance per
    dimension is 1.
    """
    if alpha is None:
        return ztr, zte
    count = ztr.shape[0]
    _, singular, vh = torch.linalg.svd(ztr, full_matrices=False)
    var = singular ** 2 / (count - 1)
    scale = (var + EPS_FRACTION * var.mean()) ** (-alpha / 2)
    utr, ute = (ztr @ vh.T) * scale, (zte @ vh.T) * scale
    gain = utr.var(0).mean().rsqrt()
    return utr * gain, ute * gain


# ----------------------------------------------------------------------------------- channels


def binomial_estimates(x: torch.Tensor, ladder: Sequence[int], generator: torch.Generator, channel: str) -> List[torch.Tensor]:
    """Decoded vectors at nested doses, from exact binomial counts of the per-bin process.

    pbit: independent p-bits with P(+1) = x, decoder = (+1 count) / T (the same as (1 + mean of +-1) / 2).
    poisson: Bernoulli bins with probability F_DT * x, decoder = count / (T * F_DT).
    """
    prob = x.clamp(EPS / 2, 1 - EPS / 2) if channel == "pbit" else F_DT * x
    scale = 1.0 if channel == "pbit" else F_DT
    total, done, estimates = torch.zeros_like(x), 0, []
    for dose in ladder:
        total += torch.binomial(torch.full_like(x, float(dose - done)), prob, generator=generator)
        done = dose
        estimates.append(total / (dose * scale))
    return estimates


# ------------------------------------------------------------------------------ one recording


def score_trials(hits: Dict[str, Dict[int, List[float]]], name: str, pred: torch.Tensor, held: torch.Tensor,
                 labels: torch.Tensor, trials: torch.Tensor) -> None:
    for t in trials[held].unique():
        rows = held & (trials == t)
        hits.setdefault(name, {}).setdefault(int(t), []).append(float((pred[rows[held]] == labels[rows]).float().mean()))


def evaluate_stream(z: torch.Tensor, labels: torch.Tensor, trials: torch.Tensor, seed: int) -> Dict[str, object]:
    """Per-trial accuracy for every arm, channel and dose of one stream."""
    hits: Dict[str, Dict[int, List[float]]] = {}
    penalties: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    for k, held in enumerate(make_folds(trials, seed)):
        ztr, zte = normalise(z[~held], z[held])
        ytr = labels[~held]
        for a, (arm, alpha) in enumerate(ARMS.items()):
            utr, ute = encode_arm(ztr, zte, alpha)
            xtr, xte = torch.sigmoid(utr), torch.sigmoid(ute)
            penalty = select_penalty(xtr, ytr, trials[~held], seed + k, PENALTIES_LOW)
            penalties[arm].append(penalty)
            readout = Readout(xtr, ytr, penalty)
            score_trials(hits, f"{arm}|clean", readout.predict(xte), held, labels, trials)
            for c, channel in enumerate(CHANNELS):
                for repeat in range(REPEATS):
                    generator = torch.Generator().manual_seed(100000 * seed + 1000 * k + 100 * a + 10 * repeat + c)
                    for dose, estimate in zip(LADDER, binomial_estimates(xte, LADDER, generator, channel)):
                        score_trials(hits, f"{arm}|{channel}:{dose}", readout.predict(estimate), held, labels, trials)
    trial_accuracy = {name: [float(np.mean(c[t])) for t in sorted(c)] for name, c in hits.items()}
    return {"trial_accuracy": trial_accuracy, "penalties": penalties}


def evaluate_recording(rec: Recording, seed: int) -> Dict[str, object]:
    return {name: evaluate_stream(z, rec.labels, rec.trials, seed) for name, z in rec.streams.items()}


# ----------------------------------------------------------------------------------- statistics


def resample_index(rng: np.random.Generator, n_trials: int) -> np.ndarray:
    return rng.integers(0, n_trials, size=(BOOTSTRAPS, n_trials))


def paired_difference(arm: np.ndarray, plain: np.ndarray, index: np.ndarray) -> Dict[str, float]:
    """Mean difference arm - plain with a percentile interval, resampling trials jointly."""
    diff = (arm - plain)[index].mean(1)
    return {"difference": float((arm - plain).mean()), "low": float(np.quantile(diff, 0.025)), "high": float(np.quantile(diff, 0.975))}


def rho_star(arm: np.ndarray, plain_clean: np.ndarray, index: np.ndarray) -> Dict[str, float]:
    """rho* = (A_arm - 1/6) / (A_plain,clean - 1/6) with a trial bootstrap."""
    c, d = plain_clean[index].mean(1), arm[index].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (d - CHANCE) / (c - CHANCE)
    rho = rho[np.isfinite(rho)]
    point = (float(arm.mean()) - CHANCE) / (float(plain_clean.mean()) - CHANCE)
    return {"rho": point, "rho_low": float(np.quantile(rho, 0.025)), "rho_high": float(np.quantile(rho, 0.975))}


def summarise_stream(per_trial: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, object]:
    plain_clean = per_trial["plain|clean"]
    index = resample_index(rng, len(plain_clean))
    clean_low = float(np.quantile(plain_clean[index].mean(1), 0.025))
    out: Dict[str, object] = {"estimable": estimable(clean_low), "clean_low_plain": clean_low, "arms": {}}
    for arm in ARMS:
        clean = per_trial[f"{arm}|clean"]
        entry: Dict[str, object] = {"clean_accuracy": float(clean.mean()),
                                    "clean_low": float(np.quantile(clean[index].mean(1), 0.025))}
        for channel in CHANNELS:
            rows = []
            for dose in LADDER:
                a = per_trial[f"{arm}|{channel}:{dose}"]
                row = {"accuracy": float(a.mean()), **paired_difference(a, per_trial[f"plain|{channel}:{dose}"], index)}
                row.update(rho_star(a, plain_clean, index))
                rows.append(row)
            entry[channel] = {"curve": rows, "T90": first_dose(rows, 0.90, "rho"), "T95": first_dose(rows, 0.95, "rho"),
                              "T90_lower": first_dose(rows, 0.90, "rho_low"), "T95_lower": first_dose(rows, 0.95, "rho_low")}
        out["arms"][arm] = entry
    return out


# ------------------------------------------------------------------------------------------ IO


def encoding_path(out_dir: Path, model: int, draw: int) -> Path:
    return out_dir / f"enc_m{model}_s{draw}.json"


def progress(message: str, started: float) -> None:
    print(f"[{time.time() - started:7.0f} s] {message}", file=sys.stderr, flush=True)


def command_evaluate(args: argparse.Namespace) -> None:
    out_dir, started = Path(args.out_dir), time.time()
    for path in sorted(out_dir.glob("rec_m*_s*.pt")):
        model, draw = [int(p[1:]) for p in path.stem[4:].split("_")]
        target = encoding_path(out_dir, model, draw)
        if target.exists() or (args.models and model not in args.models):
            continue
        data = torch.load(path, weights_only=False)
        rec = Recording(data["streams"], data["labels"], data["trials"])
        target.write_text(json.dumps(evaluate_recording(rec, seed=1000 * model + draw)))
        progress(f"encoded model {model} draw {draw}", started)


def load_pooled(out_dir: Path) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    pooled: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    for path in sorted(out_dir.glob("enc_m*_s*.json")):
        model = int(path.stem[5:].split("_")[0])
        for stream, res in json.loads(path.read_text()).items():
            slot = pooled.setdefault(model, {}).setdefault(stream, {})
            for name, values in res["trial_accuracy"].items():
                slot.setdefault(name, []).extend(values)
    return {m: {s: {c: np.array(v) for c, v in conds.items()} for s, conds in streams.items()} for m, streams in pooled.items()}


def command_summarise(args: argparse.Namespace) -> None:
    pooled = load_pooled(Path(args.out_dir))
    rng = np.random.default_rng(0)
    summary = {"status": STATUS, "ladder": list(LADDER), "arms": list(ARMS), "eps_fraction": EPS_FRACTION, "models": {}}
    for model in sorted(pooled):
        summary["models"][str(model)] = {s: summarise_stream(c, rng) for s, c in pooled[model].items()}
    print_summary(summary)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))


def show(value: Optional[int]) -> str:
    return "none" if value is None else str(value)


def print_summary(summary: Dict[str, object]) -> None:
    print(f"variance-equalised encoding, instruments {summary['status']}, doses {summary['ladder']}")
    for model, streams in summary["models"].items():
        for stream, e in streams.items():
            if not e["estimable"]:
                print(f"model {model} {stream}: NOT ESTIMABLE")
                continue
            print(f"model {model} {stream}")
            for arm, a in e["arms"].items():
                for channel in CHANNELS:
                    c = a[channel]
                    accuracy = " ".join(f"{r['accuracy']:.2f}" for r in c["curve"])
                    print(f"   {arm:6s} {channel:8s} clean {a['clean_accuracy']:.3f}  A(T) {accuracy}  "
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
