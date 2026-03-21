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
from models.core.semantic_pathway import SemanticPathway
from scripts.training.metrics_logger import ConsciousnessMetricsLogger, StepMetrics

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

    semantic = SemanticPathway(
        input_dim=config.get("semantic_input_dim", 1536),
        workspace_dim=config["workspace_dim"],
    ).to(device)

    return tectum, workspace, reentrant, modulator, emotion_shaper, memory, action_core, semantic


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
                modulator, emotion_shaper, action_core, env,
                semantic=None, metrics_logger=None, global_step_offset=0):
    device = config["device"]
    max_steps = config["max_steps"]

    obs, _ = env.reset()
    total_reward = 0.0
    previous_broadcast = None
    steps_taken = 0
    phi_accum = 0.0
    conscious_steps = 0

    if metrics_logger is not None:
        metrics_logger.reset_episode_state()

    for step in range(max_steps):
        global_step = global_step_offset + step
        frame_tensor = frame_to_tensor(obs, device)

        audio_spatial = torch.zeros(1, config["tectum_feature_dim"], 2, device=device)
        tectum_content, vision_bid = tectum(frame_tensor, audio_spatial)

        emotion = evaluate_reflex_emotion(obs)

        # Semantic pathway: zero embedding when Qwen2-VL unavailable (bid degrades to 0)
        semantic_embedding = torch.zeros(1, 1536, device=device)
        if semantic is not None:
            semantic_content, semantic_bid = semantic(semantic_embedding)
        else:
            semantic_content = torch.zeros(1, config["workspace_dim"], device=device)
            semantic_bid = 0.0

        bids = {
            "vision": max(0.0, min(1.0, vision_bid)),
            "audio": 0.0,
            "memory": 0.1,
            "body": 0.05,
            "semantic": max(0.0, min(1.0, semantic_bid)),
        }
        payloads = {
            "vision": {"tensor": tectum_content, "source": "tectum"},
            "semantic": {"tensor": semantic_content, "source": "semantic"},
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
        sync_r = getattr(workspace, 'last_sync_R', 0.0)

        if not isinstance(broadcast, torch.Tensor):
            broadcast = torch.zeros(1, config["workspace_dim"], device=device)

        broadcast_mag = float(broadcast.norm().item())

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
        phi_accum += phi
        if is_conscious:
            conscious_steps += 1
        obs = next_obs
        steps_taken = step + 1

        # --- Metrics logging ---
        if metrics_logger is not None:
            # Build gate state tuple from workspace competition results
            comp = workspace.state.competition_results
            gate_state = tuple(comp.get(k, 0.0) for k in sorted(comp.keys())) if comp else None
            # Workspace state from broadcast magnitude bins
            ws_state = (broadcast_mag, phi, sync_r)

            metrics_logger.log_step(StepMetrics(
                global_step=global_step,
                phi=phi,
                sync_r=sync_r,
                is_conscious=bool(is_conscious),
                reward=reward_val,
                broadcast_mag=broadcast_mag,
                valence=emotion["valence"],
                arousal=emotion["arousal"],
                dominance=emotion["dominance"],
                gate_state=gate_state,
                workspace_state=ws_state,
            ))

            # Insight detection
            state_hash = f"{int(obs.mean())}_{int(obs.std())}"
            action_key = action if isinstance(action, (int, str)) else int(action)
            is_insight = metrics_logger.detect_insight_moment(
                state_hash=state_hash,
                action=action_key,
                reward=reward_val,
                broadcast_mag=broadcast_mag,
            )
            if is_insight:
                logger.info(f"  ** INSIGHT MOMENT at step {step} (phi={phi:.3f}, R={sync_r:.3f}) **")

        if step % 20 == 0:
            logger.info(
                f"  step {step:3d} | phi={phi:.3f} | R={sync_r:.3f} | conscious={is_conscious} | "
                f"arousal={arousal:.2f} | reward={reward_val:.3f}"
            )

        if done:
            break

    avg_phi = phi_accum / max(steps_taken, 1)
    consciousness_ratio = conscious_steps / max(steps_taken, 1)
    return total_reward, steps_taken, avg_phi, consciousness_ratio


def main():
    parser = argparse.ArgumentParser(description="Train consciousness agent in the Dark Room")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--render", action="store_true", help="Render the environment in a window")
    parser.add_argument("--env", type=str, default="dark_room", choices=["dark_room", "navigation"],
                        help="Environment to train in")
    parser.add_argument("--log-dir", type=str, default="runs", help="Directory for metrics logs")
    parser.add_argument("--log-ei-every", type=int, default=50,
                        help="Compute EI every N episodes (0 to disable)")
    args = parser.parse_args()

    config = build_config(args)
    device = config["device"]
    logger.info(f"Device: {device}")

    tectum, workspace, reentrant, modulator, emotion_shaper, memory, action_core, semantic = init_components(config)

    render_mode = "human" if args.render else "rgb_array"
    if args.env == "navigation":
        from simulations.environments.navigation_env import NavigationEnv
        env = NavigationEnv(render_mode=render_mode, width=224, height=224)
    else:
        env = SimpleVisualEnv(render_mode=render_mode, width=224, height=224)

    metrics_logger = ConsciousnessMetricsLogger(
        log_dir=args.log_dir, use_tensorboard=True
    )

    logger.info(f"Starting training: {args.episodes} episodes, {args.max_steps} max steps each")
    logger.info(f"Metrics logging to: {args.log_dir}")

    rewards_history = []
    global_step = 0
    for ep in range(args.episodes):
        logger.info(f"Episode {ep + 1}/{args.episodes}")
        ep_reward, ep_steps, avg_phi, consciousness_ratio = run_episode(
            ep, config, tectum, workspace, reentrant,
            modulator, emotion_shaper, action_core, env, semantic,
            metrics_logger=metrics_logger, global_step_offset=global_step,
        )
        global_step += ep_steps
        rewards_history.append(ep_reward)
        avg_last_5 = np.mean(rewards_history[-5:])

        # EI computation at configured interval
        ei_gates, ei_workspace, ei_ratio = 0.0, 0.0, 0.0
        if args.log_ei_every > 0 and (ep + 1) % args.log_ei_every == 0:
            ei_result = metrics_logger.compute_and_log_ei(ep)
            ei_gates = ei_result["ei_gates"]
            ei_workspace = ei_result["ei_workspace"]
            ei_ratio = ei_result["ratio"]
            logger.info(
                f"  EI: gates={ei_gates:.4f} workspace={ei_workspace:.4f} "
                f"ratio={ei_ratio:.2f} emergent={ei_result['emergent']}"
            )

        metrics_logger.log_episode(
            episode=ep, total_reward=ep_reward, steps=ep_steps,
            avg_phi=avg_phi, consciousness_ratio=consciousness_ratio,
            ei_gates=ei_gates, ei_workspace=ei_workspace, ei_ratio=ei_ratio,
        )

        logger.info(
            f"Episode {ep + 1} done | steps={ep_steps} | "
            f"reward={ep_reward:.2f} | avg(last5)={avg_last_5:.2f} | "
            f"phi={avg_phi:.3f} | conscious={consciousness_ratio:.1%}"
        )

    metrics_logger.close()
    env.close()
    logger.info("Training complete.")
    logger.info(f"Final avg reward (last 5): {np.mean(rewards_history[-5:]):.2f}")


if __name__ == "__main__":
    main()
