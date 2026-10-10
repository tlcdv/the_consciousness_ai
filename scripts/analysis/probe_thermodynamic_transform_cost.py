"""Cost of the equalising transform: does the saving in channel events survive counting the pre-processing?

Background. docs/results/thermodynamic_encoding_2026_10.md and thermodynamic_matched_readout_2026_10.md found that
rotating each vector into its training principal axes and scaling the axes before a stochastic channel cut the events
needed by 16 to 1024 times at equal numbers of dimensions. Those results count channel events only. The transform is a
linear map applied before the channel. This probe counts it. It is arithmetic on operation counts and on the measured
T95* of the matched-readout study. It runs no new channel. It says nothing about any measured hardware.

COUNTING RULES, fixed before the run. One decision uses R = 256 channel dimensions. Online operations per vector:
    common    the same for every arm. A stream of 256 raw dimensions needs a per-dimension z score (256 multiply-adds).
              A larger stream needs the training-fold PCA projection (raw dimension x 256 multiply-adds). The sigmoid
              and the readout (256 x 6 multiply-adds) are also the same for every arm and are counted in `common`.
    extra     the multiply-adds an equalised arm needs beyond plain. Linear maps compose, so all of the arm's linear
              steps fold into ONE matrix applied once.
              Raw 256-D stream: the arm needs a dense 256 x 256 matrix (rotation and scaling) in place of the diagonal
              z score. extra = 256 x 256 - 256 = 65280.
              Larger stream: the PCA scores are already principal axes, so the rotation is a signed permutation and the
              scaling is a diagonal. The scaling folds into the columns of the PCA projection matrix. extra = 0.
              This is checked on the recordings: the mean over rows of max |V| (V from the SVD of the training scores)
              is 1 for a signed permutation. It is reported.
    channel   R x T events for dose T (one p-bit sample or Bernoulli bin per dimension per dose step).
    energy    in units of one channel event: E = R x T + rho x (online multiply-adds), where rho = energy of one
              multiply-add / energy of one channel event. rho is NOT measured here. It is swept over 1, 10, 100, 1000,
              10000. The setup cost per dimension (setting the bias) is the same for every arm and is left out.
Offline costs (fitting the covariance and its SVD) are paid once per stream and are not counted. This is a limit.

REFERENCE SCENARIO, given beside the sweep and labelled. Horowitz, ISSCC 2014 (45 nm): a 32-bit floating point
multiply is 3.7 pJ and an add is 0.9 pJ, so a multiply-add is about 4.6 pJ (as quoted in a Frontiers parameter table,
https://pmc.ncbi.nlm.nih.gov/articles/PMC8934428/table/T3). Extropic's modelled cell energy is about 2 fJ per
Gibbs-sampler cell update (arXiv:2510.23972, section III, a physical model and not a measurement; it includes the random
number generator, the bias, the clock and the communication). The ratio is 4.6 pJ / 2 fJ = 2300. A digital process
below 45 nm or an integer datapath would lower rho. A coupled-cell energy may differ from an independent-sample energy.
I did not verify an 8-bit integer figure.

WHAT IS REPORTED, fixed before the run, no pass line.
  1. Online multiply-add counts per arm and stream, and the diagonality check.
  2. For each model and stream, with T95* of the matched-readout study (p-bit): the break-even ratio
     rho* = R x (T95*_plain - T95*_arm) / extra, the value of rho above which the transform costs more than the
     channel events it saves. "free" means extra = 0. Where plain never reaches T95*, T_plain is set to 16384, so
     rho* is a lower bound and is marked ">=". Where the arm never reaches T95* the entry is "not reached".
  3. The net energy saving factor E_plain / E_arm for each rho in the sweep, and for the reference scenario.
  4. The same with the clean-trained T95* as a sensitivity check.
Nothing is pooled across models.

DIAGONAL-ONLY CHECK, added AFTER the first report. The report showed that the diagonality of obs_map is 0.42, not 1,
while that of z_state is 1.0 and that of the 256-D streams is about 0.25. So for obs_map the tested encoding is NOT a
diagonal scaling of the PCA scores, and the "free" count above rests on an untested variant. The check runs that variant.
    diag_half, diag_full   u_k = s_k / (var_k + eps)^(1/4) and u_k = s_k / (var_k + eps)^(1/2), where s are the
                           training-fold PCA scores of a stream larger than 256 dimensions, with no rotation, and
                           the same gain rescale as the encoding probe. This is the encoding that folds into the PCA
                           matrix at zero extra online cost.
    Streams obs_map and z_state, the 30 cached recordings, p-bit channel, both training regimes, the settings of the
    matched-readout probe. The comparison is T95* of diag_half and diag_full against half and full of the matched-readout
    study (same recordings and folds, new noise draws, so one rung of the ladder is within noise), as a point estimate
    and as the smallest dose where the lower 95% bound reaches it, and accuracy at doses 16, 256 and 4096.
    If the diagonal variants match the rotated ones, the zero-cost count stands for obs_map. If they do not, obs_map needs
    the dense rotation and its extra is 256 x 256 - 256 like the raw streams. No pass line.

Usage:
    python -m scripts.analysis.probe_thermodynamic_transform_cost report --output-json runs/thermo_fidelity/transform_cost.json
    python -m scripts.analysis.probe_thermodynamic_transform_cost evaluate-diag
    python -m scripts.analysis.probe_thermodynamic_transform_cost summarise-diag --output-json runs/thermo_fidelity/diag_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from scripts.analysis.probe_thermodynamic_cost import Recording, make_folds, normalise  # noqa: E402
from scripts.analysis.probe_thermodynamic_encoding import (  # noqa: E402
    EPS_FRACTION,
    encode_arm,
    resample_index,
    rho_star,
    score_trials,
)
from scripts.analysis.probe_thermodynamic_fidelity import LADDER, Readout, estimable, first_dose  # noqa: E402
from scripts.analysis.probe_thermodynamic_matched import REGIMES, score_channel, select_penalty_fast  # noqa: E402

R = 256
N_CLASSES = 6
STREAM_DIMS = {"tectum_content": 256, "workspace_broadcast": 256, "obs_map": 1024, "z_state": 16384}
ARM_NAMES = ("plain", "pca0", "half", "full")
RHO_SWEEP = (1, 10, 100, 1000, 10000)
E_MAC_PJ, E_EVENT_FJ = 4.6, 2.0
REFERENCE_RHO = E_MAC_PJ * 1000.0 / E_EVENT_FJ
T_CAP = 16384
STATUS = "UNPROVEN"


# ------------------------------------------------------------------------------ operation counts


def common_macs(stream_dim: int) -> int:
    """Multiply-adds every arm needs: normalisation (z score or PCA projection) and the readout."""
    normalisation = stream_dim if stream_dim <= R else stream_dim * R
    return normalisation + R * N_CLASSES


def extra_macs(arm: str, stream_dim: int) -> int:
    """Multiply-adds an arm needs beyond plain, with every linear step folded into one matrix."""
    if arm == "plain" or stream_dim > R:
        return 0
    return R * R - R


def online_macs(arm: str, stream_dim: int) -> int:
    return common_macs(stream_dim) + extra_macs(arm, stream_dim)


def energy(arm: str, stream_dim: int, dose: int, rho: float) -> float:
    """Energy of one decision in units of one channel event: R x T events plus rho per multiply-add."""
    return R * dose + rho * online_macs(arm, stream_dim)


def break_even(t_plain: Optional[int], t_arm: Optional[int], stream_dim: int, arm: str) -> Dict[str, object]:
    """The rho above which the transform costs more than the channel events it saves."""
    if t_arm is None:
        return {"status": "not reached"}
    plain_never = t_plain is None
    saved = R * ((T_CAP if plain_never else t_plain) - t_arm)
    extra = extra_macs(arm, stream_dim)
    if extra == 0:
        return {"status": "free", "lower_bound": plain_never}
    return {"status": "finite", "rho_star": saved / extra, "lower_bound": plain_never}


def net_saving(t_plain: Optional[int], t_arm: Optional[int], stream_dim: int, arm: str, rho: float) -> Optional[float]:
    """E_plain / E_arm at the two budgets. None when the arm never reaches the target."""
    if t_arm is None:
        return None
    plain = T_CAP if t_plain is None else t_plain
    return energy("plain", stream_dim, plain, rho) / energy(arm, stream_dim, t_arm, rho)


# ---------------------------------------------------------------------------------- the checks


def diagonality(scores: torch.Tensor) -> float:
    """Mean over rows of max |V| for V from the SVD of centred scores. 1 means a signed permutation."""
    centred = scores.double() - scores.double().mean(0)
    _, _, vh = torch.linalg.svd(centred, full_matrices=False)
    return float(vh.abs().max(dim=1).values.mean())


def composed_matrix(ztr: torch.Tensor, alpha: float) -> torch.Tensor:
    """The single matrix A with encode_arm(ztr, .)(z) = gain * A z for the equalised arms (z centred)."""
    probe = torch.eye(ztr.shape[1], dtype=ztr.dtype)
    _, a = encode_arm(ztr, probe, alpha)
    return a.T  # row i of `a` is encode(e_i), so the map z -> u is a.T applied to z


# --------------------------------------------------------------------------- diagonal-only variant

DIAG_ARMS = {"diag_half": 0.5, "diag_full": 1.0}
LARGE_STREAMS = ("obs_map", "z_state")


def encode_diag(ztr: torch.Tensor, zte: torch.Tensor, alpha: float):
    """Scale the PCA scores by (var + eps)^(-alpha / 2) with no rotation, then rescale to unit mean variance."""
    var = ztr.var(0)
    scale = (var + EPS_FRACTION * var.mean()) ** (-alpha / 2)
    utr, ute = ztr * scale, zte * scale
    gain = utr.var(0).mean().rsqrt()
    return utr * gain, ute * gain


def evaluate_diag_stream(z: torch.Tensor, labels: torch.Tensor, trials: torch.Tensor, seed: int) -> Dict[str, object]:
    """Per-trial accuracy of the diagonal-only arms for one large stream (p-bit channel, both regimes)."""
    hits: Dict[str, Dict[int, List[float]]] = {}
    for k, held in enumerate(make_folds(trials, seed)):
        ztr, zte = normalise(z[~held], z[held])
        ytr, ttr = labels[~held], trials[~held]
        for a, (arm, alpha) in enumerate(DIAG_ARMS.items()):
            utr, ute = encode_diag(ztr, zte, alpha)
            xtr, xte = torch.sigmoid(utr), torch.sigmoid(ute)
            readout = Readout(xtr, ytr, select_penalty_fast(xtr, ytr, ttr, seed + k))
            score_trials(hits, f"{arm}|clean", readout.predict(xte), held, labels, trials)
            score_channel(hits, arm, "pbit", (xtr, xte), (ytr, ttr), readout, held, (labels, trials),
                          100000 * seed + 1000 * k + 100 * a + 7, seed + k)
    return {"trial_accuracy": {name: [float(np.mean(c[t])) for t in sorted(c)] for name, c in hits.items()}}


def diag_path(out_dir: Path, model: int, draw: int) -> Path:
    return out_dir / f"dia_m{model}_s{draw}.json"


def command_evaluate_diag(args: argparse.Namespace) -> None:
    out_dir, started = Path(args.out_dir), time.time()
    for path in sorted(out_dir.glob("rec_m*_s*.pt")):
        model, draw = [int(p[1:]) for p in path.stem[4:].split("_")]
        target = diag_path(out_dir, model, draw)
        if target.exists():
            continue
        data = torch.load(path, weights_only=False)
        rec = Recording(data["streams"], data["labels"], data["trials"])
        result = {s: evaluate_diag_stream(rec.streams[s], rec.labels, rec.trials, seed=1000 * model + draw) for s in LARGE_STREAMS}
        target.write_text(json.dumps(result))
        print(f"[{time.time() - started:7.0f} s] diagonal-only model {model} draw {draw}", file=sys.stderr, flush=True)


def pool_files(out_dir: Path, prefix: str) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
    pooled: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    for path in sorted(out_dir.glob(f"{prefix}_m*_s*.json")):
        model = int(path.stem.split("_")[1][1:])
        for stream, res in json.loads(path.read_text()).items():
            if stream not in LARGE_STREAMS:
                continue
            slot = pooled.setdefault(model, {}).setdefault(stream, {})
            for name, values in res["trial_accuracy"].items():
                slot.setdefault(name, []).extend(values)
    return {m: {s: {c: np.array(v) for c, v in d.items()} for s, d in st.items()} for m, st in pooled.items()}


def diag_entry(diag: Dict[str, np.ndarray], mat: Dict[str, np.ndarray], rng: np.random.Generator) -> Dict[str, object]:
    plain_clean = mat["plain|clean"]
    index = resample_index(rng, len(plain_clean))
    out: Dict[str, object] = {"estimable": estimable(float(np.quantile(plain_clean[index].mean(1), 0.025)))}
    for arm in ("half", "full", "diag_half", "diag_full"):
        source = diag if arm.startswith("diag") else mat
        per_regime = {}
        for regime in REGIMES:
            rows = [{"accuracy": float(source[f"{arm}|{regime}|pbit|{d}"].mean()),
                     **rho_star(source[f"{arm}|{regime}|pbit|{d}"], plain_clean, index)} for d in LADDER]
            per_regime[regime] = {"curve": rows, "T95": first_dose(rows, 0.95, "rho"), "T95_lower": first_dose(rows, 0.95, "rho_low"),
                                  "T90": first_dose(rows, 0.90, "rho")}
        out[arm] = per_regime
    return out


def command_summarise_diag(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    diag, mat = pool_files(out_dir, "dia"), pool_files(out_dir, "mat")
    rng = np.random.default_rng(0)
    report = {"status": STATUS, "models": {str(m): {s: diag_entry(diag[m][s], mat[m][s], rng) for s in diag[m]} for m in sorted(diag)}}
    ix = {t: i for i, t in enumerate(LADDER)}
    for model, streams in report["models"].items():
        for stream, e in streams.items():
            print(f"model {model} {stream}")
            for regime in REGIMES:
                for arm in ("half", "diag_half", "full", "diag_full"):
                    c = e[arm][regime]
                    acc = " ".join(f"{c['curve'][ix[d]]['accuracy']:.2f}" for d in (16, 256, 4096))
                    print(f"   {regime:13s} {arm:9s} T90 {c['T90']} T95 {c['T95']} (lower {c['T95_lower']})  A(16,256,4096) {acc}")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))


# ------------------------------------------------------------------------------------- report


def load_t95(summary: Dict[str, object], regime: str) -> Dict[str, Dict[str, Dict[str, Optional[int]]]]:
    """{model: {stream: {arm: T95 (p-bit)}}} from the matched-readout summary."""
    out: Dict[str, Dict[str, Dict[str, Optional[int]]]] = {}
    for model, streams in summary["models"].items():
        for stream, entry in streams.items():
            out.setdefault(model, {})[stream] = {arm: entry["arms"][arm][regime]["pbit"]["T95"] for arm in ARM_NAMES}
    return out


def build_entry(t95: Dict[str, Optional[int]], stream: str) -> Dict[str, object]:
    dim = STREAM_DIMS[stream]
    out: Dict[str, object] = {}
    for arm in ("pca0", "half", "full"):
        sweep = {str(rho): net_saving(t95["plain"], t95[arm], dim, arm, rho) for rho in RHO_SWEEP}
        sweep["reference"] = net_saving(t95["plain"], t95[arm], dim, arm, REFERENCE_RHO)
        out[arm] = {"T95_plain": t95["plain"], "T95_arm": t95[arm], "break_even": break_even(t95["plain"], t95[arm], dim, arm),
                    "net_saving": sweep}
    return out


def diagonality_check(recordings_dir: Path, models: Sequence[int], draws: Sequence[int]) -> Dict[str, float]:
    values: Dict[str, List[float]] = {s: [] for s in STREAM_DIMS}
    for model in models:
        for draw in draws:
            path = recordings_dir / f"rec_m{model}_s{draw}.pt"
            if not path.exists():
                continue
            data = torch.load(path, weights_only=False)
            for stream, z in data["streams"].items():
                train, _ = normalise(z[:400], z[400:])
                values[stream].append(diagonality(train))
    return {s: (sum(v) / len(v) if v else float("nan")) for s, v in values.items()}


def command_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    summary = json.loads((out_dir / "matched_summary.json").read_text())
    report: Dict[str, object] = {"status": STATUS, "R": R, "rho_sweep": list(RHO_SWEEP), "reference_rho": REFERENCE_RHO,
                                 "macs": {s: {a: online_macs(a, d) for a in ARM_NAMES} for s, d in STREAM_DIMS.items()},
                                 "diagonality": diagonality_check(out_dir, args.models, args.draws), "regimes": {}}
    for regime in ("matched", "clean-trained"):
        t95 = load_t95(summary, regime)
        report["regimes"][regime] = {m: {s: build_entry(t95[m][s], s) for s in t95[m]} for m in t95}
    print_report(report)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))


def fmt_break_even(b: Dict[str, object]) -> str:
    if b["status"] == "free":
        return "free"
    if b["status"] == "not reached":
        return "not reached"
    return (">= " if b["lower_bound"] else "") + f"{b['rho_star']:.0f}"


def print_report(report: Dict[str, object]) -> None:
    print(f"transform cost, instruments {report['status']}, R {report['R']}, reference rho {report['reference_rho']:.0f}")
    print("online multiply-adds:", report["macs"])
    print("diagonality (1 = signed permutation):", {k: round(v, 3) for k, v in report["diagonality"].items()})
    for regime, models in report["regimes"].items():
        print(f"\n[{regime}]")
        for model, streams in models.items():
            for stream, arms in streams.items():
                for arm, e in arms.items():
                    sweep = e["net_saving"]
                    shown = " ".join("n/a" if v is None else f"{v:.2f}" for k, v in sweep.items() if k != "reference")
                    ref = sweep["reference"]
                    print(f"  model {model} {stream:20s} {arm:5s} T95 plain {e['T95_plain']} arm {e['T95_arm']}  "
                          f"break-even {fmt_break_even(e['break_even'])}  net saving at rho {list(RHO_SWEEP)}: {shown}  "
                          f"reference {'n/a' if ref is None else f'{ref:.2f}'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out-dir", default="runs/thermo_fidelity")
    p.add_argument("--models", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--draws", type=int, nargs="+", default=[300, 301, 302])
    p.add_argument("--output-json")
    for name in ("evaluate-diag", "summarise-diag"):
        q = sub.add_parser(name)
        q.add_argument("--out-dir", default="runs/thermo_fidelity")
        if name == "summarise-diag":
            q.add_argument("--output-json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    {"report": command_report, "evaluate-diag": command_evaluate_diag, "summarise-diag": command_summarise_diag}[args.command](args)


if __name__ == "__main__":
    main()
