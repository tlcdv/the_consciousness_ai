"""Events and energy frontier of rank-truncated equalised encodings.

Background. docs/results/thermodynamic_transform_cost_2026_10.md showed that the event-level saving of the equalised
encoding does not carry over to total energy in most settings, because (a) the dense 256 x 256 rotation of the raw
streams costs 65280 multiply-adds and (b) for the larger streams the PCA projection to 256 axes dominates the energy.
Both costs scale with the number of retained axes k, and so does the number of channel events per decision (k x T).
This probe asks how few axes a stream can keep while the readable class information survives, and whether keeping
fewer axes moves the break-even in total energy. It is a software emulation. It says nothing about hardware,
consciousness or Phi.

ARMS, p-bit channel, readout trained on the channel's own noisy outputs (the matched regime of the matched-readout probe):
    plain256         u = z, 256 axes, the encoding of the fidelity probe (the yardstick).
    full{k}          the top k training principal axes, each scaled by (var + eps)^(-1/2), eps = 0.01 x mean var of the
                     retained axes, k in 8, 16, 32, 64, 128, 256. Every arm is rescaled to mean training variance 1.
    pca{k}           the top k axes with no scaling, k in 16, 64, 256. This separates truncation from equalisation.
Channel, doses, folds, readout, penalty grid, REPEATS and COPIES are those of the matched-readout probe. The data are the
30 cached recordings (models 42, 43, 44; stimulus draws 300 to 309; 1200 trials per model). Nothing new is recorded.

COSTS, fixed before the run. Channel events per decision: k x T. Online multiply-adds per decision:
    plain256 raw stream (256-D)    256 (z score) + 256 x 6 (readout)
    k-axis arm, raw stream         256 x k (one dense map folds z score, rotation and scaling) + k x 6
    plain256 or k-axis, larger     d x k (the PCA projection to k axes; the scaling folds into it) + k x 6, where d is the
                                   raw dimension (1024 for obs_map, 16384 for z_state) and k = 256 for plain256
Energy in units of one channel event: k x T + rho x multiply-adds, with rho swept over 1, 10, 100, 1000, 10000 and 2300 (the
labelled reference ratio of the transform-cost study). rho is not measured here. Offline fitting is not counted.

PREDICTIONS, written before the run, with no thresholds.
  F1. For obs_map, a small k (16 to 32) with equalisation keeps most of the class information, needs far fewer events than
      plain256, and lowers the total energy at rho of 10 or more, because the projection cost falls with k.
  F2. For tectum_content and workspace_broadcast the accuracy ceiling falls as k falls. Small k may not reach the target
      yardstick. There may be a k at which the total-energy break-even is better than the full 256 x 256 rotation.
  F3. Model 44 tectum_content and z_state, whose class information sits in low-variance axes, need large k.
  F4. At equal k and dose, equalised arms (full) are at least as accurate as unscaled arms (pca).

PILOT NOTE. A pilot on one new recording (model 43, draw 999, not part of the data, discarded) ran the code end to end in 108 s
per recording. It contradicted F1 for obs_map: the ceiling fell to 0.20 to 0.37 for k of 8 to 32 and recovered only at k of 64.
It supported F2 for tectum_content, where the ceiling fell below k of 64. For workspace_broadcast the ceiling stayed high down to k of 16.
Nothing in the arms, the predictions or the report plan was changed after the pilot. F1 to F4 stand as written.

WHAT IS REPORTED, fixed before the run, no pass line.
  1. The clean accuracy (the ceiling) of each arm, per model and stream.
  2. Accuracy A(T) of each arm at each dose, matched-trained.
  3. The events E* = k x T needed to reach 90% and 95% of the above-chance accuracy of the plain256 arm's clean readout (rho*),
     as a point estimate and as the smallest events value where the lower 95% bound reaches it. "none" means not reached
     within the ladder.
  4. For each arm and rho: the total-energy ratio E_plain256 / E_arm at the two reached events values, where both reach 95%.
     Where plain256 does not reach it, plain256 is taken at T = 16384, so the ratio is a lower bound and is marked.
  5. rho* only where the lower bound of the plain256 clean accuracy exceeds 1/6 + 0.05.
Nothing is pooled across models.

Usage:
    python -m scripts.analysis.probe_thermodynamic_frontier evaluate
    python -m scripts.analysis.probe_thermodynamic_frontier summarise --output-json runs/thermo_fidelity/frontier_summary.json
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
from scripts.analysis.probe_thermodynamic_encoding import (  # noqa: E402
    EPS_FRACTION,
    binomial_estimates,
    resample_index,
    rho_star,
    score_trials,
)
from scripts.analysis.probe_thermodynamic_fidelity import LADDER, N_CLASSES, REPEATS, Readout, estimable, first_dose  # noqa: E402
from scripts.analysis.probe_thermodynamic_matched import COPIES, select_penalty_fast  # noqa: E402

R = 256
STREAM_DIMS = {"tectum_content": 256, "workspace_broadcast": 256, "obs_map": 1024, "z_state": 16384}
ARMS: Dict[str, Tuple[Optional[float], int]] = {"plain256": (None, 256)}
ARMS.update({f"full{k}": (1.0, k) for k in (8, 16, 32, 64, 128, 256)})
ARMS.update({f"pca{k}": (0.0, k) for k in (16, 64, 256)})
RHOS = (1, 10, 100, 1000, 2300, 10000)
T_CAP = 16384
STATUS = "UNPROVEN"


# ------------------------------------------------------------------------------- encodings


def encode_rank(ztr: torch.Tensor, zte: torch.Tensor, alpha: float, k: int):
    """Top-k training principal axes, each scaled by (var + eps)^(-alpha / 2), rescaled to unit mean variance."""
    count = ztr.shape[0]
    _, singular, vh = torch.linalg.svd(ztr, full_matrices=False)
    var, axes = singular[:k] ** 2 / (count - 1), vh[:k]
    scale = (var + EPS_FRACTION * var.mean()) ** (-alpha / 2)
    utr, ute = (ztr @ axes.T) * scale, (zte @ axes.T) * scale
    gain = utr.var(0).mean().rsqrt()
    return utr * gain, ute * gain


def encode(arm: str, ztr: torch.Tensor, zte: torch.Tensor):
    alpha, k = ARMS[arm]
    return (ztr, zte) if alpha is None else encode_rank(ztr, zte, alpha, k)


# ----------------------------------------------------------------------------------- costs


def arm_macs(arm: str, stream_dim: int) -> int:
    """Online multiply-adds per decision under the folding rules of the docstring."""
    alpha, k = ARMS[arm]
    if alpha is None:
        return (stream_dim if stream_dim <= R else stream_dim * R) + R * N_CLASSES
    return stream_dim * k + k * N_CLASSES


def events(arm: str, dose: int) -> int:
    return ARMS[arm][1] * dose


def total_energy(arm: str, stream_dim: int, dose: int, rho: float) -> float:
    return events(arm, dose) + rho * arm_macs(arm, stream_dim)


def energy_ratio(stream_dim: int, arm: str, t_arm: Optional[int], t_plain: Optional[int], rho: float) -> Optional[float]:
    """E_plain256 / E_arm. None when the arm does not reach the target. Plain at the cap when it does not reach it."""
    if t_arm is None:
        return None
    plain = T_CAP if t_plain is None else t_plain
    return total_energy("plain256", stream_dim, plain, rho) / total_energy(arm, stream_dim, t_arm, rho)


# ------------------------------------------------------------------------------ one recording


def evaluate_stream(z: torch.Tensor, labels: torch.Tensor, trials: torch.Tensor, seed: int) -> Dict[str, object]:
    """Per-trial accuracy of every arm: clean ceiling and matched-trained p-bit accuracy at every dose."""
    hits: Dict[str, Dict[int, List[float]]] = {}
    for fold, held in enumerate(make_folds(trials, seed)):
        ztr, zte = normalise(z[~held], z[held])
        ytr, ttr = labels[~held], trials[~held]
        for a, arm in enumerate(ARMS):
            utr, ute = encode(arm, ztr, zte)
            xtr, xte = torch.sigmoid(utr), torch.sigmoid(ute)
            clean = Readout(xtr, ytr, select_penalty_fast(xtr, ytr, ttr, seed + fold))
            score_trials(hits, f"{arm}|clean", clean.predict(xte), held, labels, trials)
            base = 100000 * seed + 1000 * fold + 100 * a
            tests = [binomial_estimates(xte, LADDER, torch.Generator().manual_seed(base + 3 * r), "pbit") for r in range(REPEATS)]
            copies = [binomial_estimates(xtr, LADDER, torch.Generator().manual_seed(base + 10 + c), "pbit") for c in range(COPIES)]
            for i, dose in enumerate(LADDER):
                x_noisy = torch.cat([c[i] for c in copies])
                y_noisy, t_noisy = ytr.repeat(COPIES), ttr.repeat(COPIES)
                readout = Readout(x_noisy, y_noisy, select_penalty_fast(x_noisy, y_noisy, t_noisy, seed + fold))
                for r in range(REPEATS):
                    score_trials(hits, f"{arm}|matched|{dose}", readout.predict(tests[r][i]), held, labels, trials)
    return {"trial_accuracy": {name: [float(np.mean(c[t])) for t in sorted(c)] for name, c in hits.items()}}


def evaluate_recording(rec: Recording, seed: int) -> Dict[str, object]:
    return {name: evaluate_stream(z, rec.labels, rec.trials, seed) for name, z in rec.streams.items()}


# ------------------------------------------------------------------------------------ summary


def summarise_stream(per_trial: Dict[str, np.ndarray], stream: str, rng: np.random.Generator) -> Dict[str, object]:
    plain_clean = per_trial["plain256|clean"]
    index = resample_index(rng, len(plain_clean))
    out: Dict[str, object] = {"estimable": estimable(float(np.quantile(plain_clean[index].mean(1), 0.025))), "arms": {}}
    dim, plain_t95 = STREAM_DIMS.get(stream, R), None
    for arm in ARMS:
        rows = []
        for dose in LADDER:
            row = {"accuracy": float(per_trial[f"{arm}|matched|{dose}"].mean())}
            row.update(rho_star(per_trial[f"{arm}|matched|{dose}"], plain_clean, index))
            rows.append(row)
        entry = {"k": ARMS[arm][1], "macs": arm_macs(arm, dim), "clean_accuracy": float(per_trial[f"{arm}|clean"].mean()), "curve": rows}
        for level, name in ((0.90, "90"), (0.95, "95")):
            entry[f"T{name}"] = first_dose(rows, level, "rho")
            entry[f"T{name}_lower"] = first_dose(rows, level, "rho_low")
        out["arms"][arm] = entry
    plain_t95 = out["arms"]["plain256"]["T95"]
    for arm, entry in out["arms"].items():
        entry["events95"] = None if entry["T95"] is None else ARMS[arm][1] * entry["T95"]
        entry["events90"] = None if entry["T90"] is None else ARMS[arm][1] * entry["T90"]
        entry["energy_ratio"] = {str(rho): energy_ratio(dim, arm, entry["T95"], plain_t95, rho) for rho in RHOS}
    return out


# --------------------------------------------------------------------------------------- IO


def frontier_path(out_dir: Path, model: int, draw: int) -> Path:
    return out_dir / f"fro_m{model}_s{draw}.json"


def progress(message: str, started: float) -> None:
    print(f"[{time.time() - started:7.0f} s] {message}", file=sys.stderr, flush=True)


def command_evaluate(args: argparse.Namespace) -> None:
    out_dir, started = Path(args.out_dir), time.time()
    for path in sorted(out_dir.glob("rec_m*_s*.pt")):
        model, draw = [int(p[1:]) for p in path.stem[4:].split("_")]
        target = frontier_path(out_dir, model, draw)
        if target.exists() or (args.models and model not in args.models):
            continue
        data = torch.load(path, weights_only=False)
        rec = Recording(data["streams"], data["labels"], data["trials"])
        target.write_text(json.dumps(evaluate_recording(rec, seed=1000 * model + draw)))
        progress(f"frontier model {model} draw {draw}", started)


def load_pooled(out_dir: Path) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    pooled: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    for path in sorted(out_dir.glob("fro_m*_s*.json")):
        model = int(path.stem.split("_")[1][1:])
        for stream, res in json.loads(path.read_text()).items():
            slot = pooled.setdefault(model, {}).setdefault(stream, {})
            for name, values in res["trial_accuracy"].items():
                slot.setdefault(name, []).extend(values)
    return {m: {s: {c: np.array(v) for c, v in d.items()} for s, d in st.items()} for m, st in pooled.items()}


def command_summarise(args: argparse.Namespace) -> None:
    pooled = load_pooled(Path(args.out_dir))
    rng = np.random.default_rng(0)
    summary = {"status": STATUS, "ladder": list(LADDER), "rhos": list(RHOS), "arms": list(ARMS), "models": {}}
    for model in sorted(pooled):
        summary["models"][str(model)] = {s: summarise_stream(c, s, rng) for s, c in pooled[model].items()}
    print_summary(summary)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))


def print_summary(summary: Dict[str, object]) -> None:
    print(f"frontier, instruments {summary['status']}, doses {summary['ladder']}, rhos {summary['rhos']}")
    for model, streams in summary["models"].items():
        for stream, e in streams.items():
            if not e["estimable"]:
                print(f"model {model} {stream}: NOT ESTIMABLE")
                continue
            print(f"model {model} {stream}")
            for arm, a in e["arms"].items():
                ratios = " ".join("n/a" if v is None else f"{v:.2f}" for v in a["energy_ratio"].values())
                print(f"   {arm:9s} k {a['k']:3d} ceiling {a['clean_accuracy']:.3f}  T95 {a['T95']} (lower {a['T95_lower']})  "
                      f"events95 {a['events95']}  energy ratio vs plain256 at rho {list(RHOS)}: {ratios}")


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
