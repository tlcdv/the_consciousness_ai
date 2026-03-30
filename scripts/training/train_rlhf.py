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
from models.core.consciousness_gating import ConsciousnessGate
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
        "enable_audio": getattr(args, "enable_audio", False),
        "audio_sample_rate": 16000,
        "audio_num_bands": 64,
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

    # ConsciousnessGate: produces 5 continuous gate values (attention, stability,
    # adaptation, coherence, confidence) from workspace broadcast. These are the
    # causal nodes for IIT Phi computation and EI measurement.
    gate = ConsciousnessGate({
        "hidden_size": config["workspace_dim"],
        "gating": {
            "attention_threshold": 0.5,
            "stability_threshold": 0.6,
            "base_adaptation_rate": 0.01,
        },
    }).to(device)

    # Optimizer for tectum + gate parameters (retinotopic encoder, RSSM, capsules,
    # attention/stability networks) so that phi and sync_R evolve during training
    tectum_optimizer = torch.optim.Adam(
        list(tectum.parameters()) + list(gate.parameters()),
        lr=config.get("tectum_lr", 3e-4),
    )

    # Auxiliary reward predictor: maps tectum content to scalar reward estimate
    reward_predictor = torch.nn.Sequential(
        torch.nn.Linear(config["workspace_dim"], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    ).to(device)
    reward_optimizer = torch.optim.Adam(reward_predictor.parameters(), lr=1e-3)

    # Workspace binding optimizer: optimizes KuramotoLayer coupling_weights and
    # natural_frequencies so sync_R becomes reward-correlated (dopamine modulates
    # gamma synchrony). Without this, sync_R stays static at ~0.22.
    workspace_optimizer = torch.optim.Adam(
        workspace.binding_system.parameters(), lr=config.get("workspace_lr", 1e-4)
    )

    # Auditory specialist: cochlear pipeline (gammatone -> hair cell -> tonotopic
    # encoder -> workspace projection). Only instantiated when --enable-audio is set.
    auditory_specialist = None
    if config.get("enable_audio", False):
        from models.audio.auditory_specialist import AuditorySpecialist
        auditory_specialist = AuditorySpecialist(config).to(device)
        logger.info("Auditory specialist enabled (cochlear pipeline)")

    return (tectum, workspace, reentrant, modulator, emotion_shaper, memory,
            action_core, semantic, gate, tectum_optimizer, reward_predictor,
            reward_optimizer, workspace_optimizer, auditory_specialist)


def frame_to_tensor(frame: np.ndarray, device: str) -> torch.Tensor:
    """Convert RGB uint8 frame [H, W, 3] to float tensor [1, 3, H, W]."""
    t = torch.from_numpy(frame).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def evaluate_emotion(vision_bid: float, env_reward: float, prev_reward: float,
                     broadcast: torch.Tensor | None = None,
                     qualia_mapper=None) -> dict:
    """Two-stage emotion: reflex (pre-conscious) + appraisal (post-broadcast).

    Stage 1 (reflex): surprise from tectum bid and reward prediction error
    drive arousal and valence. This is fast and content-independent.

    Stage 2 (appraisal): if a workspace broadcast exists, the phenomenological
    mapper extracts valence and intensity from the broadcast content, blending
    them into the reflex estimate. This is slower and content-specific.
    """
    # Reflex: tectum surprise and reward delta
    # Arousal requires bid > 0.5 to activate (baseline bids are ~0.2-0.5)
    surprise = max(0.0, vision_bid - 0.5)
    reward_delta = env_reward - prev_reward
    valence = float(np.clip(reward_delta * 2.0, -1.0, 1.0))
    arousal = float(np.clip(surprise + abs(reward_delta) * 0.5, 0.0, 1.0))
    dominance = 0.0

    # Appraisal: phenomenological state from broadcast content
    if broadcast is not None and qualia_mapper is not None:
        try:
            phenom = qualia_mapper.map_state(broadcast)
            valence = 0.6 * valence + 0.4 * phenom.valence
            dominance = phenom.intensity * 0.3
        except Exception:
            pass  # graceful fallback to reflex-only

    return {"valence": valence, "arousal": arousal, "dominance": dominance}


def run_episode(episode_idx, config, tectum, workspace, reentrant,
                modulator, emotion_shaper, action_core, env,
                semantic=None, gate=None, metrics_logger=None, global_step_offset=0,
                tectum_optimizer=None, reward_predictor=None, reward_optimizer=None,
                workspace_optimizer=None, auditory_specialist=None):
    device = config["device"]
    max_steps = config["max_steps"]

    obs, info = env.reset()
    total_reward = 0.0
    previous_broadcast = None
    prev_reward = 0.0
    steps_taken = 0
    phi_accum = 0.0
    conscious_steps = 0

    # Reset recurrent state between episodes
    tectum.h_state = None
    tectum.z_state = None
    if hasattr(action_core, 'pfc_hidden'):
        action_core.pfc_hidden = None

    if metrics_logger is not None:
        metrics_logger.reset_episode_state()

    for step in range(max_steps):
        global_step = global_step_offset + step
        frame_tensor = frame_to_tensor(obs, device)

        # Audio processing: cochlear pipeline when enabled, zero stub otherwise
        audio_affect = None
        if auditory_specialist is not None and isinstance(obs, np.ndarray):
            audio_waveform = info.get("audio_waveform") if isinstance(info, dict) else None
            if audio_waveform is not None:
                waveform_t = torch.from_numpy(audio_waveform).float().unsqueeze(0).unsqueeze(0).to(device)
                audio_content, audio_bid_raw = auditory_specialist(waveform_t)
                audio_spatial = auditory_specialist.get_spatial_for_tectum()
                audio_affect = auditory_specialist.get_affect_output()
            else:
                audio_content = torch.zeros(1, config["workspace_dim"], device=device)
                audio_bid_raw = 0.0
                audio_spatial = torch.zeros(1, config["tectum_feature_dim"], 2, device=device)
        else:
            audio_content = torch.zeros(1, config["workspace_dim"], device=device)
            audio_bid_raw = 0.0
            audio_spatial = torch.zeros(1, config["tectum_feature_dim"], 2, device=device)

        tectum_content, vision_bid = tectum(frame_tensor, audio_spatial)

        # Stage 1: reflex emotion (pre-workspace, drives affective bid modulation)
        # Audio startle reflex: spectral flux and roughness drive arousal spike
        emotion = evaluate_emotion(vision_bid, 0.0, prev_reward)
        if audio_affect is not None:
            af = audio_affect["acoustic_features"]
            spectral_flux = af[0, 4].item() if af.shape[1] > 4 else 0.0
            roughness = af[0, 2].item() if af.shape[1] > 2 else 0.0
            emotion["arousal"] = float(np.clip(emotion["arousal"] + spectral_flux * 0.3, 0, 1))
            emotion["valence"] = float(np.clip(emotion["valence"] - roughness * 0.2, -1, 1))

        # Semantic pathway: zero embedding when Qwen2-VL unavailable (bid degrades to 0)
        semantic_embedding = torch.zeros(1, 1536, device=device)
        if semantic is not None:
            semantic_content, semantic_bid = semantic(semantic_embedding)
        else:
            semantic_content = torch.zeros(1, config["workspace_dim"], device=device)
            semantic_bid = 0.0

        bids = {
            "vision": max(0.0, min(1.0, vision_bid)),
            "audio": max(0.0, min(1.0, audio_bid_raw)),
            "memory": 0.1,
            "body": 0.05,
            "semantic": max(0.0, min(1.0, semantic_bid)),
        }
        # Include capsule structured payload so GNW broadcast preserves compositional info
        vision_payload = {"tensor": tectum_content, "source": "tectum"}
        capsule_data = tectum.get_capsule_payload()
        if capsule_data:
            vision_payload.update(capsule_data)
        payloads = {
            "vision": vision_payload,
            "audio": {"tensor": audio_content, "source": "audio"},
            "semantic": {"tensor": semantic_content, "source": "semantic"},
        }

        specialists = {"vision": tectum}
        if auditory_specialist is not None and audio_bid_raw > 0.0:
            specialists["audio"] = auditory_specialist
        settle_result = reentrant.settle(
            workspace=workspace,
            specialists=specialists,
            initial_bids=bids,
            payloads=payloads,
            goal_vector=torch.tensor([1.0, -1.0, 1.0], device=device),
        )

        broadcast = settle_result.broadcast_content
        is_conscious = settle_result.is_conscious
        sync_r = getattr(workspace, 'last_sync_R', 0.0)

        if not isinstance(broadcast, torch.Tensor):
            broadcast = torch.zeros(1, config["workspace_dim"], device=device)

        broadcast_mag = float(broadcast.norm().item())

        # ConsciousnessGate: compute 5 causal gate values from broadcast
        # These drive IIT Phi (via empirical TPM) and EI measurement
        if gate is not None:
            gate_input = broadcast.detach().view(-1)[:config["workspace_dim"]]
            if gate_input.shape[0] < config["workspace_dim"]:
                gate_input = torch.nn.functional.pad(
                    gate_input, (0, config["workspace_dim"] - gate_input.shape[0])
                )
            _, gate_state_obj = gate(gate_input.unsqueeze(0))
            phi_result = workspace.iit_metrics.compute_phi_from_gate_state(gate_state_obj)
            phi = phi_result.phi + (sync_r * 0.1)
        else:
            phi = settle_result.phi

        # Stage 2: appraisal emotion (post-broadcast, content-specific)
        emotion = evaluate_emotion(
            vision_bid, 0.0, prev_reward,
            broadcast=broadcast, qualia_mapper=workspace.qualia_mapper,
        )

        arousal = emotion["arousal"]
        action, value = action_core.select_action(
            broadcast, emotion_arousal=arousal, rpe_cache=0.0
        )

        # Discrete environments (DMTS, WCST): convert continuous action to int
        env_action = action
        if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
            env_action = int(np.argmax(action[:env.action_space.n]))

        next_obs, env_reward, terminated, truncated, info = env.step(env_action)
        done = terminated or truncated

        # --- Tectum + reward predictor auxiliary training ---
        # Gives gradient signal to tectum parameters so phi/sync_R can evolve.
        # Reward prediction loss backprops through tectum_content into the
        # retinotopic encoder, RSSM, and capsule parameters.
        if tectum_optimizer is not None and reward_predictor is not None and step % 5 == 0:
            pred_reward = reward_predictor(tectum_content)
            reward_target = torch.tensor([[env_reward]], device=device)
            pred_loss = torch.nn.functional.mse_loss(pred_reward, reward_target)

            tectum_optimizer.zero_grad()
            reward_optimizer.zero_grad()
            pred_loss.backward(retain_graph=True)
            tectum_optimizer.step()
            reward_optimizer.step()

        # --- Workspace binding optimizer ---
        # Reward-correlated sync: maximize sync_R when reward is positive,
        # penalize when negative. Biologically grounded: dopamine modulates
        # gamma-band synchronization (Benchenane et al. 2010).
        if workspace_optimizer is not None and step % 10 == 0:
            sync_tensor = getattr(workspace, 'last_sync_R_tensor', None)
            if sync_tensor is not None and sync_tensor.requires_grad:
                # Loss: -reward_signal * sync_R (maximize sync when reward positive)
                reward_signal = float(np.clip(env_reward, -1.0, 1.0))
                sync_loss = -reward_signal * sync_tensor.squeeze()
                workspace_optimizer.zero_grad()
                sync_loss.backward(retain_graph=True)
                workspace_optimizer.step()

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
        prev_reward = reward_val
        total_reward += reward_val
        phi_accum += phi
        if is_conscious:
            conscious_steps += 1
        obs = next_obs
        steps_taken = step + 1

        # --- Metrics logging ---
        if metrics_logger is not None:
            # Gate state from ConsciousnessGate (5 causal node values)
            if gate is not None:
                gs = gate.state
                gate_state = (
                    gs.attention_level, gs.stability_score,
                    gs.adaptation_rate, gs.meta_memory_coherence,
                    gs.narrator_confidence,
                )
            else:
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

            # Insight detection (coarse hashing to prevent every step being "novel")
            # Use downsampled pixel sum for cheap hashing (avoids float64 alloc from np.std)
            state_hash = f"{int(obs[::4, ::4].sum() / 1000)}"
            if isinstance(action, (int, str)):
                action_key = action
            elif hasattr(action, '__len__'):
                # Round continuous actions to nearest integer for coarse binning
                action_key = "_".join(str(round(float(a))) for a in action)
            else:
                action_key = int(action)
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
    parser.add_argument("--env", type=str, default="dark_room",
                        choices=["dark_room", "navigation", "dmts", "wcst"],
                        help="Environment to train in")
    parser.add_argument("--difficulty", type=int, default=0,
                        help="Distractor overlap level for DMTS (0-3)")
    parser.add_argument("--log-dir", type=str, default="runs", help="Directory for metrics logs")
    parser.add_argument("--log-ei-every", type=int, default=50,
                        help="Compute EI every N episodes (0 to disable)")
    parser.add_argument("--enable-audio", action="store_true",
                        help="Enable cochlear auditory pipeline")
    args = parser.parse_args()

    config = build_config(args)
    device = config["device"]
    logger.info(f"Device: {device}")

    # Override action dim for discrete environments BEFORE init_components
    # so ActionSelectionCore is built with the correct output dimension
    render_mode = "human" if args.render else "rgb_array"
    if args.env == "dmts":
        config["action_selection"]["action_dim"] = 5
    elif args.env == "wcst":
        config["action_selection"]["action_dim"] = 4

    (tectum, workspace, reentrant, modulator, emotion_shaper, memory,
     action_core, semantic, gate, tectum_optimizer, reward_predictor,
     reward_optimizer, workspace_optimizer, auditory_specialist) = init_components(config)

    if args.env == "navigation":
        from simulations.environments.navigation_env import NavigationEnv
        env = NavigationEnv(render_mode=render_mode, width=224, height=224)
    elif args.env == "dmts":
        from simulations.environments.dmts_env import DMTSEnv
        env = DMTSEnv(render_mode=render_mode, width=224, height=224,
                      distractor_overlap=args.difficulty)
    elif args.env == "wcst":
        from simulations.environments.wcst_env import WCSTEnv
        env = WCSTEnv(render_mode=render_mode, width=224, height=224)
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
            gate=gate,
            metrics_logger=metrics_logger, global_step_offset=global_step,
            tectum_optimizer=tectum_optimizer,
            reward_predictor=reward_predictor,
            reward_optimizer=reward_optimizer,
            workspace_optimizer=workspace_optimizer,
            auditory_specialist=auditory_specialist,
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
