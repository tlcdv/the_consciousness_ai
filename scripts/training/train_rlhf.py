"""
Training entrypoint for the consciousness agent in the Dark Room environment.

Runs the full cognitive loop: perception (tectum) -> emotion (PAD modulator) ->
consciousness (GNW with reentrant processing) -> action (basal ganglia Go/No-Go).

This script uses the core architecture components directly. It does not require
Qwen2-VL or other large model weights, running instead on the DINOv2 retinotopic
encoder (falls back to a conv stack when weights are unavailable).

Usage:
    python -m scripts.training.train_rlhf
    python -m scripts.training.train_rlhf --episodes 50 --max-steps 200
"""
from __future__ import annotations

import sys
import os
import argparse
import logging

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from simulations.environments.simple_visual_env import SimpleVisualEnv
from models.core.sensory_tectum import SensoryTectum
from models.core.global_workspace import GlobalWorkspace
from models.core.reentrant_processor import ReentrantProcessor
from models.emotion.affective_modulator import AffectiveModulator
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.self_model.action_selection_core import ActionSelectionCore
from models.memory.memory_core import MemoryCore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def build_config(args):
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tectum_feature_dim": 64,
        "tectum_grid_size": 16,
        "workspace_dim": 256,
        "workspace": {
            "broadcast_threshold": 0.6,
            "ignition_gain": 5.0,
            "reverberation_alpha": 0.8,
            "workspace_dim": 256,
        },
        "reentrant": {
            "max_cycles": 5,
            "convergence_threshold": 0.01,
        },
        "emotion": {
            "valence_weight": 0.5,
            "arousal_penalty": 1.0,
        },
        "action_selection": {
            "workspace_dim": 256,
            "action_dim": args.action_dim,
            "context_dim": 128,
            "learning_rate": args.lr,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "memory": {},
        "episodes": args.episodes,
        "max_steps": args.max_steps,
    }


def init_components(config):
    device = config["device"]

    tectum = SensoryTectum({
        "tectum_feature_dim": config["tectum_feature_dim"],
        "tectum_grid_size": config["tectum_grid_size"],
        "workspace_dim": config["workspace_dim"],
    }).to(device)

    workspace = GlobalWorkspace(config["workspace"])

    reentrant = ReentrantProcessor(config["reentrant"])

    modulator = AffectiveModulator()

    emotion_shaper = EmotionalRewardShaper(config["emotion"]).to(device)

    memory = MemoryCore(config["memory"])

    action_core = ActionSelectionCore(
        config["action_selection"],
        emotion_shaper,
        memory,
    )

    return tectum, workspace, reentrant, modulator, emotion_shaper, memory, action_core


def frame_to_tensor(frame: np.ndarray, device: str) -> torch.Tensor:
    """Convert RGB uint8 frame [H, W, 3] to float tensor [1, 3, H, W]."""
    t = torch.from_numpy(frame).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def evaluate_reflex_emotion(frame: np.ndarray) -> dict:
    """Fast heuristic emotion from pixel brightness (amygdala analog)."""
    brightness = frame.mean() / 255.0
    if brightness < 0.3:
        return {"valence": -0.7, "arousal": 0.8, "dominance": -0.4}
    elif brightness > 0.6:
        return {"valence": 0.6, "arousal": -0.2, "dominance": 0.4}
    return {"valence": 0.0, "arousal": 0.1, "dominance": 0.0}


def run_episode(episode_idx, config, tectum, workspace, reentrant,
                modulator, emotion_shaper, action_core, env):
    device = config["device"]
    max_steps = config["max_steps"]

    obs, _ = env.reset()
    total_reward = 0.0
    previous_broadcast = None
    steps_taken = 0

    for step in range(max_steps):
        frame_tensor = frame_to_tensor(obs, device)

        audio_spatial = torch.zeros(1, config["tectum_feature_dim"], 2, device=device)
        tectum_content, vision_bid = tectum(frame_tensor, audio_spatial)

        emotion = evaluate_reflex_emotion(obs)

        bids = {
            "vision": max(0.0, min(1.0, vision_bid)),
            "audio": 0.0,
            "memory": 0.1,
            "body": 0.05,
        }
        payloads = {
            "vision": {"tensor": tectum_content, "source": "tectum"},
        }

        specialists = {"vision": tectum}
        settle_result = reentrant.settle(
            workspace=workspace,
            specialists=specialists,
            initial_bids=bids,
            payloads=payloads,
            goal_vector=torch.tensor([1.0, -1.0, 1.0], device=device),
        )

        broadcast = settle_result.broadcast_content
        is_conscious = settle_result.is_conscious
        phi = settle_result.phi

        if not isinstance(broadcast, torch.Tensor):
            broadcast = torch.zeros(1, config["workspace_dim"], device=device)

        arousal = emotion["arousal"]
        action, value = action_core.select_action(
            broadcast, emotion_arousal=arousal, rpe_cache=0.0
        )

        next_obs, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if hasattr(emotion_shaper, 'compute_emotional_reward'):
            emotion_values = {
                "valence": emotion["valence"],
                "arousal": emotion["arousal"],
                "dominance": emotion["dominance"],
            }
            shaped_reward = emotion_shaper.compute_emotional_reward(
                emotion_values=emotion_values,
                base_reward=env_reward,
            )
        else:
            shaped_reward = env_reward

        prev = previous_broadcast if previous_broadcast is not None else broadcast
        action_core.step(
            workspace_broadcast=prev,
            action=action,
            raw_reward=shaped_reward,
            next_broadcast=broadcast,
            done=done,
            emotion_state=emotion,
            attention_level=phi,
            narrative="",
        )

        if step > 0 and step % 10 == 0:
            action_core.update_policy()

        previous_broadcast = broadcast.detach().clone()
        reward_val = shaped_reward if isinstance(shaped_reward, (int, float)) else shaped_reward.item()
        total_reward += reward_val
        obs = next_obs
        steps_taken = step + 1

        if step % 20 == 0:
            logger.info(
                f"  step {step:3d} | phi={phi:.3f} | conscious={is_conscious} | "
                f"arousal={arousal:.2f} | reward={reward_val:.3f}"
            )

        if done:
            break

    return total_reward, steps_taken


def main():
    parser = argparse.ArgumentParser(description="Train consciousness agent in the Dark Room")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--render", action="store_true", help="Render the environment in a window")
    args = parser.parse_args()

    config = build_config(args)
    device = config["device"]
    logger.info(f"Device: {device}")

    tectum, workspace, reentrant, modulator, emotion_shaper, memory, action_core = init_components(config)

    render_mode = "human" if args.render else "rgb_array"
    env = SimpleVisualEnv(render_mode=render_mode, width=224, height=224)

    logger.info(f"Starting training: {args.episodes} episodes, {args.max_steps} max steps each")

    rewards_history = []
    for ep in range(args.episodes):
        logger.info(f"Episode {ep + 1}/{args.episodes}")
        ep_reward, ep_steps = run_episode(
            ep, config, tectum, workspace, reentrant,
            modulator, emotion_shaper, action_core, env,
        )
        rewards_history.append(ep_reward)
        avg_last_5 = np.mean(rewards_history[-5:])
        logger.info(
            f"Episode {ep + 1} done | steps={ep_steps} | "
            f"reward={ep_reward:.2f} | avg(last5)={avg_last_5:.2f}"
        )

    env.close()
    logger.info("Training complete.")
    logger.info(f"Final avg reward (last 5): {np.mean(rewards_history[-5:]):.2f}")


if __name__ == "__main__":
    main()
