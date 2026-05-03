"""Run actual training pipeline briefly and inspect iit_metrics state.

After indirect tests, this is the smoking gun: actually run the same
init_components + run_episode pipeline as the verification did, then
peek inside workspace.iit_metrics to see what state_history accumulated
and what phi computes manually.

Compare with --ablate-gate-diversity to see if diversity loss collapses
gate states.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.training.train_rlhf import (
    init_components, run_episode, build_config,
)
from simulations.environments.simple_visual_env import SimpleVisualEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--difficulty", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="runs/diag")
    parser.add_argument("--log-ei-every", type=int, default=0)
    parser.add_argument("--enable-audio", action="store_true")
    parser.add_argument("--env", default="dark_room",
                        choices=["dark_room", "navigation", "dmts", "wcst"])
    parser.add_argument("--ablate-memory-replay", action="store_true")
    parser.add_argument("--ablate-consolidation-fix", action="store_true")
    parser.add_argument("--ablate-rnd-zero-on-reward", action="store_true")
    parser.add_argument("--ablate-gate-diversity", action="store_true")
    parser.add_argument("--ablate-gate-feedback", action="store_true")
    parser.add_argument("--ablate-pad-loop", action="store_true")
    parser.add_argument("--ablate-bptt", action="store_true")
    args = parser.parse_args()

    config = build_config(args)

    torch.manual_seed(0)
    np.random.seed(0)

    (tectum, workspace, reentrant, modulator, emotion_shaper, memory,
     action_core, semantic, gate, tectum_optimizer, reward_predictor,
     reward_optimizer, workspace_optimizer, auditory_specialist,
     self_model, rnd, rnd_optimizer, consolidation_mgr) = init_components(config)

    env = SimpleVisualEnv(width=224, height=224)

    iit = workspace.iit_metrics

    global_step = 0
    for ep in range(args.episodes):
        ep_reward, ep_steps, avg_phi, ratio = run_episode(
            ep, config, tectum, workspace, reentrant,
            modulator, action_core, env,
            gate=gate, memory=memory,
            metrics_logger=None,
            global_step_offset=global_step,
            tectum_optimizer=tectum_optimizer,
            reward_predictor=reward_predictor,
            reward_optimizer=reward_optimizer,
            workspace_optimizer=workspace_optimizer,
            auditory_specialist=auditory_specialist,
            self_model=self_model,
            rnd=rnd,
            rnd_optimizer=rnd_optimizer,
        )
        global_step += ep_steps

    print(f"\n=== After {args.episodes} episodes ({global_step} steps) ===")
    print(f"ablate_gate_diversity = {args.ablate_gate_diversity}")
    print(f"ablate_gate_feedback  = {args.ablate_gate_feedback}")
    print(f"state_history len = {len(iit.state_history)}")
    print(f"raw_history len   = {len(iit._raw_history)}")
    print(f"thresholds = {iit._thresholds}")

    if iit._raw_history:
        raw = np.array(list(iit._raw_history))
        print("\nRaw gate value stats (last {} steps):".format(len(raw)))
        for i, n in enumerate(['attention', 'stability', 'adaptation',
                               'coherence', 'confidence']):
            col = raw[:, i]
            print(f"  {n:12s}: mean={col.mean():.4f} std={col.std():.4f} "
                  f"min={col.min():.4f} max={col.max():.4f}")

    if iit.state_history:
        c = Counter(list(iit.state_history))
        print(f"\nBinarized state diversity:")
        print(f"  unique states: {len(c)}/32")
        print(f"  top-5: {c.most_common(5)}")

        most_common_state = c.most_common(1)[0][0]
        tpm = iit.build_empirical_tpm(5)
        from models.evaluation.iit_phi import GATE_CM
        phi_val = iit.calculate_phi(tpm, most_common_state, cm=GATE_CM)
        print(f"\nDirect pyphi call on most-common state {most_common_state}: "
              f"phi={phi_val:.8f}")

        print("\nphi per unique state (top 5):")
        for state, count in c.most_common(5):
            phi_val = iit.calculate_phi(tpm, state, cm=GATE_CM)
            print(f"  {state} (n={count}): phi={phi_val:.8f}")


if __name__ == "__main__":
    main()
