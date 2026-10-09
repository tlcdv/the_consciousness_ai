"""Offline thermodynamic profile: do recorded representations survive p-bit transduction?

A trained checkpoint is run on DMTS. For each recorded stream (tectum_content, the
workspace broadcast, obs_map, z_state) the probe asks one question. After the vectors pass
through a Poisson spike code and settle in a p-bit memory, can the stimulus class
(sample_shape, 6 classes) still be read out? It also reports the energy and entropy the
settling dissipates, in the emulation. This is a software emulation on a CPU. It measures
nothing about any hardware, and it says nothing about consciousness or Phi. The reported
quantities (thermodynamic_entropy, pbit_energy, fep_free_energy) are UNPROVEN instruments in
docs/instrument_inventory.md.

PIPELINE, per stream, per seed, with 5-fold cross validation grouped by trial:
    z       training-fold normalisation. Raw streams of 256 dimensions: per-dimension z score.
            Larger streams (obs_map, z_state, pooled 4x4 first): training-fold PCA to 256
            components, then one scalar scale.
    memory  one prototype per class, sign of the class mean of z on the training fold.
            Hebbian couplings J = P^T P / n, zero diagonal. 6 patterns in 256 spins is far
            below the Hopfield capacity, which the first version of this probe was not.
    spikes  test vectors: sigmoid(z) -> Poisson rate code (snn_bridge.rate_encode, 64 bins)
            -> rate_decode -> threshold 0.5 -> +-1 spins.
    settle  chromatic block Gibbs on J at BETA for SWEEPS sweeps.
    decode  class with the largest overlap with the final spin state.

ARMS reported: reference (nearest centroid on z, no transduction), quantizer (sign of z, no
noise, no settling), noisy (Poisson spikes, no settling), settled (the full pipeline), and
control (settled at beta = 0.05, where the p-bits output noise), and ridge. The ridge arm is
a linear readout on z. It is INFORMATIONAL and enters no gate. It was added after a 3
permutation smoke run on seed 42 showed nearest centroid scoring near chance on some
streams. It tells whether a stream carries the class in a form the prototype readout misses.

PRE-STATED GATE, written before the first run of this version on a checkpoint. Defaults
BETA = 4, SWEEPS = 30, margin 0.10, label permutations as given by --permutations. Per
stream, with at least 3 seeds:
    G1  reference accuracy is above the p95 of its own label-permutation null.
        If G1 fails at any seed the verdict is UNTESTABLE. The stream does not carry the
        class, so survival of the class cannot be asked.
    G2  settled accuracy >= reference accuracy - 0.10.
    G3  settled accuracy is above the p95 of the permutation null of the settled pipeline.
    G4  control accuracy is not above that same p95. A control that scores high means the
        readout carries the class without the p-bit physics, and the instrument is invalid.
    PASSED if G1 to G4 hold at every seed. FAILED if G1 holds and any of G2 to G4 fails.
    With fewer than 3 seeds the verdict is HYPOTHESIS.
Permutations shuffle the class label between whole trials and rerun the full pipeline.
Efficiency numbers use ASSUMED hardware constants passed on the command line. They are not
measurements. The dense prototype memory needs one color class per spin, so the estimate
counts 256 sequential updates per sweep. That is a cost of this memory, not of the hardware.

Usage:
    python -m scripts.analysis.probe_thermodynamic_cost --synthetic --seed 0 1 2
    python -m scripts.analysis.probe_thermodynamic_cost \\
        --checkpoint runs/capfix_alllevels/tectum.pt runs/capfix_seed43/tectum.pt \\
        runs/capfix_seed44/tectum.pt --seed 42 43 44 --episodes 60 --output-json report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.thermodynamic.energy_minimization import (  # noqa: E402
    LinearGaussianModel,
    relax_free_energy,
    stable_step_bound,
)
from models.thermodynamic.interfaces.snn_bridge import rate_decode, rate_encode  # noqa: E402
from models.thermodynamic.p_bit_emulator import BlockGibbsSampler, ising_energy  # noqa: E402

STATUS = "UNPROVEN"
MIN_SEEDS = 3
F_MAX_HZ, DT_S, N_BINS = 200.0, 1e-3, 64
N_FOLDS, N_COMPONENTS, MARGIN = 5, 256, 0.10
CONTROL_BETA = 0.05
RIDGE_LAMBDA = 100.0


@dataclass
class Recording:
    """Streams {name: [T, D]} with a class label and a trial id per row."""

    streams: Dict[str, torch.Tensor]
    labels: torch.Tensor
    trials: torch.Tensor


# ----------------------------------------------------------------- normalisation and spikes


def normalise(train: torch.Tensor, test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Training-fold normalisation to [rows, <=256]. Larger vectors go through PCA first."""
    train, test = train.double(), test.double()
    mean = train.mean(dim=0)
    if train.shape[1] <= N_COMPONENTS:
        scale = train.std(dim=0).clamp_min(1e-8)
        return (train - mean) / scale, (test - mean) / scale
    basis = torch.linalg.svd(train - mean, full_matrices=False)[2][:N_COMPONENTS]
    a, b = (train - mean) @ basis.T, (test - mean) @ basis.T
    scale = a.std().clamp_min(1e-8)
    return a / scale, b / scale


def quantize(z: torch.Tensor) -> torch.Tensor:
    """Deterministic +-1 spins, sign of z, with 0 mapped to +1."""
    return torch.where(z >= 0, 1.0, -1.0).to(torch.float64)


def to_spins(z: torch.Tensor, seed: int) -> torch.Tensor:
    """Poisson rate transduction of z [rows, n] to +-1 spins."""
    generator = torch.Generator().manual_seed(seed)
    rate = torch.sigmoid(z)
    spikes = rate_encode(rate, N_BINS, F_MAX_HZ, DT_S, generator)
    return torch.where(rate_decode(spikes, F_MAX_HZ, DT_S) > 0.5, 1.0, -1.0).to(torch.float64)


# ------------------------------------------------------------------------- memory and settling


def hebbian_couplings(patterns: torch.Tensor) -> torch.Tensor:
    """J = (1/n) sum_p xi_p xi_p^T with a zero diagonal, for patterns [P, n]."""
    J = patterns.T @ patterns / patterns.shape[1]
    return J - torch.diag(torch.diagonal(J))


def class_means(z: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Mean of z per class [n_classes, n]. A class absent from the rows raises."""
    for c in range(n_classes):
        if not (labels == c).any():
            raise ValueError(f"class {c} has no training rows in this fold")
    return torch.stack([z[labels == c].mean(dim=0) for c in range(n_classes)])


def overlap(spins: torch.Tensor, patterns: torch.Tensor) -> torch.Tensor:
    """Overlap (1/n) sum_i s_i xi_i of spins [rows, n] with patterns [P, n], shape [rows, P]."""
    return spins @ patterns.T / patterns.shape[1]


def predict_by_overlap(spins: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """Class of the prototype with the largest overlap, per row of spins."""
    return overlap(spins, prototypes).argmax(dim=1)


def gibbs_energy_trace(sampler: BlockGibbsSampler, spins: torch.Tensor, n_sweeps: int) -> Dict[str, torch.Tensor]:
    """Energy per sweep [n_sweeps + 1, n_chains] and the settled spins."""
    energies = [ising_energy(spins, sampler.J, sampler.b)]
    for _ in range(n_sweeps):
        sampler.step(spins)
        energies.append(ising_energy(spins, sampler.J, sampler.b))
    return {"energy": torch.stack(energies), "spins": spins}


def entropy_production(energy: torch.Tensor, beta: float) -> torch.Tensor:
    """Heat to the bath per sweep, beta * (E_t - E_{t+1}), in units of k_B. Signed, [n_sweeps, n_chains].

    This is the heat part of the entropy production. It is not the total.
    """
    return beta * (energy[:-1] - energy[1:])


def settle(prototypes: torch.Tensor, spins: torch.Tensor, beta: float, sweeps: int, seed: int):
    """Settle a copy of spins in the Hebbian memory of prototypes. Returns the trace dict."""
    sampler = BlockGibbsSampler(hebbian_couplings(prototypes), torch.zeros(prototypes.shape[1]), beta, seed=seed)
    return gibbs_energy_trace(sampler, spins.clone(), sweeps)


# ----------------------------------------------------------------------------- cross validation


def make_folds(trials: torch.Tensor, seed: int) -> List[torch.Tensor]:
    """Test-row masks, one per fold. Rows of one trial are always in the same fold."""
    ids = trials.unique()
    order = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))]
    return [torch.isin(trials, order[i::N_FOLDS]) for i in range(N_FOLDS)]


def permute_labels(labels: torch.Tensor, trials: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Shuffle the class label between whole trials. Class counts stay the same."""
    ids = trials.unique()
    first = torch.stack([labels[trials == t][0] for t in ids])
    shuffled = first[torch.randperm(len(ids), generator=generator)]
    out = labels.clone()
    for t, lab in zip(ids, shuffled):
        out[trials == t] = lab
    return out


def _cosine(x: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    xn = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cn = centres / centres.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return xn @ cn.T


@dataclass
class FoldData:
    """One cross validation fold, prepared once and reused by every label permutation."""

    test: torch.Tensor
    ztr: torch.Tensor
    zte: torch.Tensor
    spins: torch.Tensor


def prepare_folds(z: torch.Tensor, trials: torch.Tensor, seed: int) -> List[FoldData]:
    """Normalised train and test vectors and the test spins, for each fold. Independent of labels."""
    folds = []
    for k, test in enumerate(make_folds(trials, seed)):
        ztr, zte = normalise(z[~test], z[test])
        folds.append(FoldData(test, ztr, zte, to_spins(zte, seed + 100 * k)))
    return folds


def _ridge_predict(ztr: torch.Tensor, ytr: torch.Tensor, zte: torch.Tensor, n_classes: int) -> torch.Tensor:
    onehot = torch.nn.functional.one_hot(ytr, n_classes).to(ztr.dtype)
    gram = ztr.T @ ztr + RIDGE_LAMBDA * torch.eye(ztr.shape[1], dtype=ztr.dtype)
    return (zte @ torch.linalg.solve(gram, ztr.T @ onehot)).argmax(1)


def _fold_predictions(fold: FoldData, ytr, protos, beta, sweeps, seed, arms, n_classes):
    """{arm: (predicted classes, energy trace or None)} for one fold."""
    out = {}
    if "reference" in arms:
        out["reference"] = (_cosine(fold.zte, class_means(fold.ztr, ytr, n_classes)).argmax(1), None)
    if "ridge" in arms:
        out["ridge"] = (_ridge_predict(fold.ztr, ytr, fold.zte, n_classes), None)
    if "quantizer" in arms:
        out["quantizer"] = (predict_by_overlap(quantize(fold.zte), protos), None)
    if "noisy" in arms:
        out["noisy"] = (predict_by_overlap(fold.spins, protos), None)
    for arm, b in (("settled", beta), ("control", CONTROL_BETA)):
        if arm in arms:
            trace = settle(protos, fold.spins, b, sweeps, seed)
            out[arm] = (predict_by_overlap(trace["spins"], protos), trace["energy"])
    return out


def run_arms(folds: Sequence[FoldData], labels: torch.Tensor, n_classes: int, beta: float, sweeps: int,
             seed: int, arms: Sequence[str]) -> Dict[str, object]:
    """Accuracy per arm, pooled over folds, plus the pooled energy trace of the settled arm."""
    correct = {a: 0 for a in arms}
    traces: List[torch.Tensor] = []
    for k, fold in enumerate(folds):
        ytr = labels[~fold.test]
        protos = quantize(class_means(fold.ztr, ytr, n_classes))
        for arm, (pred, trace) in _fold_predictions(fold, ytr, protos, beta, sweeps, seed + k, arms, n_classes).items():
            correct[arm] += int((pred == labels[fold.test]).sum())
            if arm == "settled":
                traces.append(trace)
    result: Dict[str, object] = {a: correct[a] / len(labels) for a in arms}
    if traces:
        result["energy"] = torch.cat(traces, dim=1)
    return result


def permutation_null(folds, labels, trials, n_classes, beta, sweeps, seed, n_perm: int) -> Dict[str, List[float]]:
    """Reference and settled accuracy under n_perm trial-level label shuffles."""
    generator = torch.Generator().manual_seed(seed + 7919)
    out: Dict[str, List[float]] = {"reference": [], "settled": []}
    for _ in range(n_perm):
        shuffled = permute_labels(labels, trials, generator)
        r = run_arms(folds, shuffled, n_classes, beta, sweeps, seed, ("reference", "settled"))
        out["reference"].append(r["reference"]); out["settled"].append(r["settled"])
    return out


def p95(values: Sequence[float]) -> float:
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), 0.95))


# ------------------------------------------------------------------- free energy and efficiency


def pca_model(batch: torch.Tensor, n_latent: int, obs_weight: float) -> LinearGaussianModel:
    """Linear Gaussian model with the top principal directions of batch [T, D] as weights."""
    centred = batch - batch.mean(dim=0, keepdim=True)
    vh = torch.linalg.svd(centred, full_matrices=False)[2]
    k = min(n_latent, vh.shape[0])
    return LinearGaussianModel(
        weights=vh[:k].T.contiguous(),
        prior_mean=torch.zeros(k, dtype=batch.dtype),
        prior_precision=torch.eye(k, dtype=batch.dtype),
        obs_precision=obs_weight * torch.eye(batch.shape[1], dtype=batch.dtype),
    )


def fep_relaxation_trace(model: LinearGaussianModel, obs: torch.Tensor, n_steps: int) -> Dict[str, object]:
    """Heun relaxation from the prior mean at half the stability bound."""
    step = 0.5 * stable_step_bound(model)
    free = relax_free_energy(model.prior_mean.clone(), obs, model, step, n_steps, "heun").free_energy
    return {"free_energy": free.tolist(), "delta_free_energy": (free[-1] - free[0]).item()}


def energy_estimate(n_spins: int, n_chains: int, n_sweeps: int, n_colors: int,
                    gpu_flops_per_joule: float, pbit_power_w: float, pbit_rate_hz: float) -> Dict[str, float]:
    """Joules for the same sweeps on a GPU and on an assumed p-bit chip. Both inputs are ASSUMED."""
    flops = 2.0 * n_spins * n_spins * n_chains * n_sweeps
    gpu_joules = flops / gpu_flops_per_joule
    pbit_seconds = n_sweeps * n_colors / pbit_rate_hz
    pbit_joules = pbit_power_w * pbit_seconds * n_chains
    return {"flops": flops, "gpu_joules": gpu_joules, "pbit_seconds_per_chain": pbit_seconds,
            "pbit_joules": pbit_joules, "ratio_gpu_over_pbit": gpu_joules / pbit_joules}


# ------------------------------------------------------------------------------ one stream/seed


def profile_stream(z: torch.Tensor, labels: torch.Tensor, trials: torch.Tensor, seed: int, beta: float,
                   sweeps: int, n_perm: int, assumptions: Dict[str, float]) -> Dict[str, object]:
    """Full profile of one stream for one seed."""
    n_classes = int(labels.max()) + 1
    folds = prepare_folds(z, trials, seed)
    arms = run_arms(folds, labels, n_classes, beta, sweeps, seed,
                    ("reference", "ridge", "quantizer", "noisy", "settled", "control"))
    null = permutation_null(folds, labels, trials, n_classes, beta, sweeps, seed, n_perm)
    energy = arms.pop("energy")
    zall = normalise(z, z[:1])[0]
    fep = fep_relaxation_trace(pca_model(zall, 8, 1.0), zall[0], sweeps)
    n_spins = min(z.shape[1], N_COMPONENTS)
    colors = n_spins  # a dense memory needs one color class per spin
    return {
        "seed": seed, "n_rows": len(labels), "n_trials": int(trials.unique().numel()), "n_classes": n_classes,
        "class_counts": torch.bincount(labels, minlength=n_classes).tolist(), "accuracy": arms,
        "null_p95": {k: p95(v) for k, v in null.items()},
        "null_mean": {k: sum(v) / len(v) for k, v in null.items()},
        "pbit_energy": energy.mean(dim=1).tolist(),
        "thermodynamic_entropy": entropy_production(energy, beta).mean(dim=1).tolist(),
        "fep_free_energy": fep["free_energy"], "fep_delta": fep["delta_free_energy"],
        "efficiency": energy_estimate(n_spins, len(labels), sweeps, colors, **assumptions),
    }


def gate(profile: Dict[str, object]) -> Dict[str, bool]:
    """The pre-stated gates G1 to G4 for one stream and one seed."""
    acc, null = profile["accuracy"], profile["null_p95"]
    return {"G1": acc["reference"] > null["reference"],
            "G2": acc["settled"] >= acc["reference"] - MARGIN,
            "G3": acc["settled"] > null["settled"],
            "G4": acc["control"] <= null["settled"]}


def stream_verdict(profiles: Sequence[Dict[str, object]]) -> str:
    """Verdict across seeds for one stream."""
    gates = [gate(p) for p in profiles]
    if len(profiles) < MIN_SEEDS:
        return "HYPOTHESIS (fewer than 3 seeds)"
    if not all(g["G1"] for g in gates):
        return "UNTESTABLE (the stream does not carry the class at every seed)"
    return "PASSED" if all(all(g.values()) for g in gates) else "FAILED"


# ----------------------------------------------------------------------------------- recording


def synthetic_recording(seed: int, n_classes: int = 6, n_trials: int = 60, steps: int = 5, dim: int = 96,
                        separation: float = 1.0) -> Recording:
    """Class centroids plus noise, several rows per trial. A per-trial offset mimics slow drift."""
    gen = torch.Generator().manual_seed(seed)
    centroids = separation * torch.randn(n_classes, dim, generator=gen)
    rows, labels, trials = [], [], []
    for t in range(n_trials):
        offset = 0.3 * torch.randn(dim, generator=gen)
        for _ in range(steps):
            rows.append(centroids[t % n_classes] + offset + 0.5 * torch.randn(dim, generator=gen))
            labels.append(t % n_classes); trials.append(t)
    return Recording({"synthetic": torch.stack(rows)}, torch.tensor(labels), torch.tensor(trials))


def _record_row(streams, tectum, content, broadcast) -> None:
    from torch.nn.functional import avg_pool2d

    streams["tectum_content"].append(torch.tensor(content))
    streams["workspace_broadcast"].append(torch.tensor(broadcast))
    streams["obs_map"].append(avg_pool2d(tectum._last_obs_map, 4).flatten().float())
    z = tectum.z_state
    streams["z_state"].append(avg_pool2d(z.reshape(1, -1, *z.shape[-2:]), 4).flatten().float())


def load_recording(checkpoint: str, n_trials: int, seed: int, phase: str = "sample") -> Recording:
    """Four streams over DMTS rows in `phase`. Needs the full training stack."""
    import numpy as np
    from scripts.analysis.probe_pci import _seed_everything
    from scripts.analysis.probe_perception_decodability import _build_components
    from scripts.analysis.probe_sustained_vs_ignition import _step_states
    from simulations.environments.dmts_env import DMTSEnv

    _seed_everything(seed)
    comps = _build_components("dmts", action_dim=5, seed=seed, mock_semantic=True, load_tectum=checkpoint,
                              latent_mode="continuous", capsule_workspace_source="all_levels")
    env = DMTSEnv(num_trials=n_trials, sample_steps=5, fixation_steps=5, min_delay=12, max_delay=12)
    obs, info = env.reset(seed=seed)
    rng, done = np.random.default_rng(seed), False
    streams: Dict[str, list] = {k: [] for k in ("tectum_content", "workspace_broadcast", "obs_map", "z_state")}
    meta: List[Tuple[str, int]] = []
    with torch.no_grad():
        while not done:
            content, broadcast = _step_states(comps, obs)
            if info["phase"] == phase:
                _record_row(streams, comps[1], content, broadcast)
                meta.append((info["sample_shape"], int(info["trial"])))
            action = int(rng.integers(1, 3)) if info["phase"] == "choice" else 0
            obs, _, done, _, info = env.step(action)
    names = sorted({m[0] for m in meta})
    return Recording({k: torch.stack(v).float() for k, v in streams.items()},
                     torch.tensor([names.index(m[0]) for m in meta]), torch.tensor([m[1] for m in meta]))


# ------------------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", nargs="+", help="tectum.pt per seed (one path is reused for every seed)")
    parser.add_argument("--synthetic", action="store_true", help="use synthetic vectors, no checkpoint")
    parser.add_argument("--episodes", type=int, default=60, help="DMTS trials to record per seed")
    parser.add_argument("--phase", default="sample", help="DMTS phase to record (sample, delay)")
    parser.add_argument("--beta", type=float, default=4.0, help="inverse temperature of the p-bits")
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sweeps", type=int, default=30)
    parser.add_argument("--permutations", type=int, default=100, help="label shuffles for each null")
    parser.add_argument("--streams", nargs="+", help="only these streams (default all recorded)")
    parser.add_argument("--gpu-flops-per-joule", type=float, default=6.7e10, help="ASSUMED GPU efficiency")
    parser.add_argument("--pbit-power-w", type=float, default=1.0, help="ASSUMED p-bit chip power (W)")
    parser.add_argument("--pbit-rate-hz", type=float, default=5e7, help="ASSUMED p-bit update rate (Hz)")
    parser.add_argument("--output-json", help="write the full report here")
    return parser


def _recording(args: argparse.Namespace, index: int, seed: int) -> Recording:
    if args.synthetic:
        return synthetic_recording(seed)
    paths = args.checkpoint
    return load_recording(paths[index % len(paths)], args.episodes, seed, args.phase)


def run(args: argparse.Namespace) -> Dict[str, object]:
    if not args.synthetic and not args.checkpoint:
        raise SystemExit("give --checkpoint or --synthetic")
    if args.checkpoint and len(args.checkpoint) not in (1, len(args.seed)):
        raise SystemExit("give one --checkpoint, or one per --seed")
    assumptions = {"gpu_flops_per_joule": args.gpu_flops_per_joule,
                   "pbit_power_w": args.pbit_power_w, "pbit_rate_hz": args.pbit_rate_hz}
    per_stream: Dict[str, List[Dict[str, object]]] = {}
    for index, seed in enumerate(args.seed):
        rec = _recording(args, index, seed)
        for name, z in rec.streams.items():
            if args.streams and name not in args.streams:
                continue
            per_stream.setdefault(name, []).append(profile_stream(
                z, rec.labels, rec.trials, seed, args.beta, args.sweeps, args.permutations, assumptions))
    return {"status": STATUS, "source": "synthetic" if args.synthetic else args.checkpoint,
            "phase": args.phase, "beta": args.beta, "sweeps": args.sweeps, "permutations": args.permutations,
            "assumptions": assumptions, "seeds": list(args.seed),
            "verdicts": {n: stream_verdict(p) for n, p in per_stream.items()},
            "gates": {n: [gate(p) for p in ps] for n, ps in per_stream.items()}, "streams": per_stream}


def print_report(report: Dict[str, object]) -> None:
    print(f"instruments: {report['status']}   source: {report['source']}   beta {report['beta']}  "
          f"sweeps {report['sweeps']}  permutations {report['permutations']}")
    print(f"assumed constants (not measured): {report['assumptions']}")
    for name, profiles in report["streams"].items():
        print(f"\n{name}: {report['verdicts'][name]}")
        for p, g in zip(profiles, report["gates"][name]):
            a, n = p["accuracy"], p["null_p95"]
            failed = [k for k, ok in g.items() if not ok]
            print(f"  seed {p['seed']}  gates {'all pass' if not failed else 'FAIL ' + ','.join(failed)}  "
                  f"ref {a['reference']:.3f} ridge {a['ridge']:.3f} quant {a['quantizer']:.3f} noisy {a['noisy']:.3f} "
                  f"settled {a['settled']:.3f} control {a['control']:.3f}  "
                  f"null p95 ref {n['reference']:.3f} settled {n['settled']:.3f}  "
                  f"energy {p['pbit_energy'][0]:.1f}->{p['pbit_energy'][-1]:.1f}  rows {p['n_rows']}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    report = run(args)
    print_report(report)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
