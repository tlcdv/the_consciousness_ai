"""
Do traveling waves already exist in the tectum ConvGRU state h_t?

The tectum carries a recurrent state h_t of shape [64, 16, 16] across environment
steps, updated by 3 x 3 convolutional gates (models/core/sensory_tectum.py). Local
recurrent kernels on a map are the condition under which waves appear in cortex
models (Muller, Busch, Davis and Reynolds 2026, Neuron). Nobody has measured
whether this state propagates. This probe measures it before any wave mechanism is
built, per the measure-before-build rule in docs/thalamic_gating_evidence.md
section 6.

Method. One rollout per checkpoint with random actions (the tectum does not read
the policy). h_t is recorded every step, projected onto its first principal
channel direction (models/evaluation/wave_detection.project_channels), and
measured with measure_waves against a spatial-shuffle null. Units are cells per
step. No Hz.

PRE-STATED GATE (written before the first run):
  PRESENT       pgd_mean > null_p95 at all 3 trained seeds.
  ABSENT        pgd_mean <= null_p95 at all 3 trained seeds.
  INCONCLUSIVE  the seeds disagree. Claim nothing.
  The untrained tectum is a control, read first. If the untrained state also sits
  above its null, any wave in the trained state is a property of the architecture,
  not of learning, and must be reported that way.
  Singularity counts are descriptive only. No gate rests on them.

NO-CARRY CONTROL (added after the first run, before its own run, and not part of
the gate above). The shuffle null keeps each cell's own time series, so a spatially
smooth pattern of response latencies passes it even when nothing propagates, for
example one set by the stimulus layout or by zero padding at the grid border.
--no-carry zeros h (not z) before every step. The RSSM builds h_t from the
previous h and the previous latent z, and the observation enters through z, so
zeroing both removes the input as well (a first attempt did that and every cell
was constant, for trained and untrained tectums alike). Zeroing h alone keeps the
input path through z and cuts the multi-step carry in h. Reading, stated before
the control run:
  If the trained PGD stays above its null with no carry and within half of its
  carried value, the gradient comes from the input and is NOT evidence of
  propagation through the recurrent state.
  If it falls to its null, the gradient needs the carried state, which is the
  condition for a traveling wave.

Usage:
    python -m scripts.analysis.probe_tectum_waves --env dmts --steps 512
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.evaluation.wave_detection import measure_waves, project_channels  # noqa: E402
from scripts.analysis.probe_pci import _make_env, _seed_everything  # noqa: E402
from scripts.analysis.probe_perception_decodability import _build_components  # noqa: E402
from scripts.training.train_rlhf import frame_to_tensor  # noqa: E402

TRAINED = {42: "capfix_alllevels/tectum.pt",
           43: "capfix_seed43/tectum.pt",
           44: "capfix_seed44/tectum.pt"}


def record_h_states(env_name: str, seed: int, steps: int,
                    load_tectum: str | None, carry: bool = True) -> np.ndarray:
    """[steps, C, H, W] tectum h_state over one rollout with random actions."""
    _seed_everything(seed)
    action_dim = 5 if env_name == "dmts" else 2
    config, tectum, *_ = _build_components(
        env_name, action_dim=action_dim, seed=seed, mock_semantic=False,
        load_tectum=load_tectum, latent_mode="continuous",
        capsule_workspace_source="all_levels")
    env = _make_env(env_name, seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    device = config["device"]
    trace = []
    with torch.no_grad():
        for _ in range(steps):
            frame = frame_to_tensor(obs, device)
            audio = torch.zeros(1, config["tectum_feature_dim"], 2, device=device)
            if not carry and getattr(tectum, "h_state", None) is not None:
                tectum.h_state = torch.zeros_like(tectum.h_state)
            tectum(frame, audio)
            h = getattr(tectum, "h_state", None)
            if h is None:
                raise RuntimeError("tectum.h_state is None; the RSSM path did not run")
            trace.append(h[0].detach().cpu().numpy().copy())
            obs, _, terminated, truncated, _ = env.step(int(rng.integers(action_dim)))
            if terminated or truncated:
                obs, _ = env.reset()
    return np.stack(trace)


def report(label: str, h: np.ndarray, surrogates: int, seed: int) -> bool:
    field = project_channels(h)
    constant = int(np.sum(field.std(axis=0) == 0))
    if constant:
        print(f"  {label:<22} UNDEFINED, {constant} of {field[0].size} cells never change, "
              f"so no phase exists there. No PGD is reported.")
        return False
    wave = measure_waves(field, n_surrogates=surrogates, seed=seed)
    above = wave.pgd_mean > wave.null_p95
    print(f"  {label:<22} PGD {wave.pgd_mean:.4f}  null mean {wave.null_mean:.4f} "
          f"p95 {wave.null_p95:.4f}  {'ABOVE' if above else 'not above'}   "
          f"singularities {wave.singularities_median:.1f} "
          f"(null {wave.null_singularities_median:.1f})  steps {wave.steps_used}")
    return above


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--env", default="dmts", choices=["dmts", "simple"])
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--surrogates", type=int, default=100)
    parser.add_argument("--runs-dir", default="runs",
                        help="Folder that holds the capfix checkpoints.")
    parser.add_argument("--no-carry", action="store_true",
                        help="Zero h (not z) before every step. The control in the docstring.")
    args = parser.parse_args()
    carry = not args.no_carry
    print(f"carried state: {'yes' if carry else 'NO, h zeroed every step, z kept'}")

    print(f"env {args.env}, {args.steps} steps, {args.surrogates} shuffle surrogates")
    print("CONTROL, untrained tectum")
    for seed in TRAINED:
        report(f"untrained seed {seed}",
               record_h_states(args.env, seed, args.steps, None, carry),
               args.surrogates, seed)
    print("TRAINED")
    verdicts = {}
    for seed, rel in TRAINED.items():
        path = str(Path(args.runs_dir) / rel)
        if not Path(path).exists():
            raise SystemExit(f"missing checkpoint {path}")
        verdicts[seed] = report(f"trained seed {seed}",
                                record_h_states(args.env, seed, args.steps, path, carry),
                                args.surrogates, seed)
    print("PRE-STATED GATE")
    if all(verdicts.values()):
        print("  PRESENT at all 3 seeds. Read the control line before interpreting.")
    elif not any(verdicts.values()):
        print("  FAILED to find waves. ABSENT at all 3 seeds.")
    else:
        print(f"  INCONCLUSIVE, seeds disagree: {verdicts}. Claim nothing.")


if __name__ == "__main__":
    main()
