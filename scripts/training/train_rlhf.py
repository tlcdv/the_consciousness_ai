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
import random

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
from models.self_model.self_representation_core import SelfRepresentationCore
from models.memory.memory_core import MemoryCore
from models.core.semantic_pathway import SemanticPathway
from models.core.topographic_loss import topographic_spatial_loss
from models.core.rnd_curiosity import RNDCuriosity
from models.evaluation.phi_riiu import RIIUPhi
from models.memory.optimized_store import MemoryConsolidationManager
from scripts.training.metrics_logger import ConsciousnessMetricsLogger, StepMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def _set_global_seed(seed: int) -> None:
    """Seed all RNG sources used by the training loop.

    Sets python `random`, `numpy.random`, `torch` (CPU and CUDA), and
    `PYTHONHASHSEED`. Also enables deterministic algorithms with a warn-only
    fallback so the run does not crash if a non-deterministic op is used in
    a path we haven't audited. Exact bit-for-bit reproducibility across
    PyTorch versions is not promised; the goal is "two runs with seed=42
    should produce statistically comparable metrics", not byte equivalence.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass


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
            # Phase A of 2026-05-17 plan: broadcast assembly mode.
            # 'winner_take_all' (legacy) iterates winners and .update()s
            # their payloads, decoupling broadcast from sync_R.
            # 'attention_weighted' computes softmax-weighted fusion of all
            # eligible module payloads, making broadcast structurally
            # downstream of AKOrN sync_R.
            "broadcast_mode": getattr(args, "broadcast_mode", "winner_take_all"),
            "attention_temperature": getattr(args, "attention_temperature", 0.5),
            "attention_floor": getattr(args, "attention_floor", 0.05),
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
        # Ablation flags (off by default). Each reverts exactly one Phase 3
        # or 2026-04-27 change so the cause of any regression can be isolated.
        "ablate_memory_replay": getattr(args, "ablate_memory_replay", False),
        "ablate_consolidation_fix": getattr(args, "ablate_consolidation_fix", False),
        "ablate_rnd_zero_on_reward": getattr(args, "ablate_rnd_zero_on_reward", False),
        "ablate_gate_diversity": getattr(args, "ablate_gate_diversity", False),
        "ablate_gate_entropy": getattr(args, "ablate_gate_entropy", False),
        "ablate_gate_feedback": getattr(args, "ablate_gate_feedback", False),
        "ablate_pad_loop": getattr(args, "ablate_pad_loop", False),
        "ablate_bptt": getattr(args, "ablate_bptt", False),
        # pyphi sampling cadence: compute phi every Nth step instead of
        # every step. State history still accumulates every step so the
        # TPM stays warm. Cuts pyphi MIP calls N-fold to avoid the ~91k
        # call segfault threshold observed in pyphi 1.x Cython internals.
        "phi_sample_every": getattr(args, "phi_sample_every", 5),
        # RIIU parallel phi pathway (sliding-window SVD residual on broadcast).
        # Runs alongside pyphi each step, logged as phi_riiu. When enabled,
        # the phi-delta reward source switches from pyphi to RIIU so the
        # reward signal tracks the variance phi pathway. See plan
        # let-s-plan-the-next-misty-parasol.md and the upstream paper
        # arxiv:2506.13825 (Apache-2.0).
        "riiu_enabled": getattr(args, "enable_riiu", False),
        "riiu_rank": getattr(args, "riiu_rank", 16),
        "riiu_window": getattr(args, "riiu_window", 64),
        # Substrate selection for the RIIU reward source. When riiu_probe_all
        # is True, three RIIUPhi pipelines run in parallel and all three phi
        # values are logged (phi_riiu, phi_riiu_tectum, phi_riiu_audio); the
        # reward source picks the one named by riiu_source.
        "riiu_source": getattr(args, "riiu_source", "broadcast"),
        "riiu_probe_all": getattr(args, "riiu_probe_all", False),
        # Global seed for reproducibility. None means inherit ambient RNG state
        # (matches pre-RIIU behavior). Setting an int seeds python/numpy/torch
        # and the env.reset call on the first episode.
        "seed": getattr(args, "seed", None),
    }


def init_components(config):
    device = config["device"]

    tectum_config = {
        "tectum_feature_dim": config["tectum_feature_dim"],
        "tectum_grid_size": config["tectum_grid_size"],
        "workspace_dim": config["workspace_dim"],
    }
    if config.get("ablate_bptt"):
        tectum_config["bptt_window"] = 1
    tectum = SensoryTectum(tectum_config).to(device)

    workspace = GlobalWorkspace(config["workspace"])

    reentrant = ReentrantProcessor(config["reentrant"])

    modulator = AffectiveModulator()
    # Attach so workspace.run_competition can apply the modulator on every
    # reentrant cycle when pad_state is passed through. Without this attachment
    # the new explicit-arg modulation path is silently inert.
    workspace.affective_modulator = modulator

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
        "ablate_feedback": config.get("ablate_gate_feedback", False),
        "gating": {
            "attention_threshold": 0.5,
            "stability_threshold": 0.6,
            "base_adaptation_rate": 0.01,
        },
    }).to(device)

    # workspace.consciousness_gate is intentionally NOT attached. With the
    # gate attached, workspace.run_competition() runs pyphi on every
    # reentrant cycle (~5x per training step), and over a 200-episode run
    # that hits the ~91k-call segfault threshold in pyphi's Cython
    # internals around episode 76. The 4-tuple-pollution bug that the
    # earlier attachment was meant to prevent is no longer reachable: the
    # legacy compute_phi_proxy fallback in global_workspace.py was removed
    # in commit 8a322f9, so an unattached gate now means phi=0 with no
    # state_history write (instead of the proxy writing 4-tuples). Phi is
    # computed once per step in run_episode below, and sampled every Nth
    # step via --phi-sample-every (see argparse).

    # Self-model: tracks body schema, interoceptive state, capability model.
    # Provides internal drive signals (energy/fatigue/damage) that feed the
    # affective modulator, closing the embodiment-affect loop.
    self_model = SelfRepresentationCore(config.get("self_model", {}))

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

    # RND curiosity: intrinsic reward from prediction error on workspace broadcast.
    # Operates on broadcast (not raw pixels), so consciousness quality drives
    # exploration quality. Target network frozen, predictor trained.
    rnd = RNDCuriosity(input_dim=config["workspace_dim"], feature_dim=64).to(device)
    rnd_optimizer = torch.optim.Adam(rnd.predictor_network.parameters(), lr=1e-3)

    # Auditory specialist: cochlear pipeline (gammatone -> hair cell -> tonotopic
    # encoder -> workspace projection). Only instantiated when --enable-audio is set.
    auditory_specialist = None
    if config.get("enable_audio", False):
        from models.audio.auditory_specialist import AuditorySpecialist
        auditory_specialist = AuditorySpecialist(config).to(device)
        logger.info("Auditory specialist enabled (cochlear pipeline)")

    consolidation_mgr = MemoryConsolidationManager({
        "merge_threshold": 0.9,
        "prune_threshold": 0.05,
        "relevance_decay": 0.99,
        "use_legacy_merge": config.get("ablate_consolidation_fix", False),
    })

    # RIIU parallel phi pathway. Consumes one or more 256-D activation
    # substrates and computes a sliding-window SVD residual phi per substrate.
    # Returned as a dict {substrate_name: RIIUPhi} so the run_episode loop can
    # iterate substrates uniformly. In single-substrate mode the dict has one
    # entry; in --riiu-probe-all mode it has three.
    riiu_phis: dict | None = None
    if config.get("riiu_enabled", False):
        substrates = ["broadcast", "tectum", "audio"] if config.get("riiu_probe_all", False) \
            else [config.get("riiu_source", "broadcast")]
        if config.get("riiu_probe_all", False) and not config.get("enable_audio", False):
            import warnings as _warnings
            _warnings.warn(
                "--riiu-probe-all is on but --enable-audio is off. The 'audio' "
                "substrate will see a zero-vector input every step and report "
                "phi_riiu_audio=0.0. Re-run with --enable-audio for a real "
                "audio-substrate measurement.",
                RuntimeWarning,
                stacklevel=2,
            )
        riiu_phis = {
            name: RIIUPhi(
                dim=config["workspace_dim"],
                rank=config.get("riiu_rank", 16),
                window=config.get("riiu_window", 64),
                device=device,
            )
            for name in substrates
        }
        logger.info(
            f"RIIU phi pathway enabled (substrates={substrates}, "
            f"reward_source={config.get('riiu_source', 'broadcast')}, "
            f"dim={config['workspace_dim']}, rank={config.get('riiu_rank', 16)}, "
            f"window={config.get('riiu_window', 64)})"
        )

    return (tectum, workspace, reentrant, modulator, emotion_shaper, memory,
            action_core, semantic, gate, tectum_optimizer, reward_predictor,
            reward_optimizer, workspace_optimizer, auditory_specialist, self_model,
            rnd, rnd_optimizer, consolidation_mgr, riiu_phis)


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
                modulator, action_core, env,
                gate=None, memory=None,
                metrics_logger=None, global_step_offset=0,
                tectum_optimizer=None, reward_predictor=None, reward_optimizer=None,
                workspace_optimizer=None, auditory_specialist=None,
                self_model=None, rnd=None, rnd_optimizer=None,
                riiu_phis: dict | None = None,
                riiu_source: str = "broadcast"):
    device = config["device"]
    max_steps = config["max_steps"]

    obs, info = env.reset()
    total_reward = 0.0
    previous_broadcast = None
    steps_taken = 0
    phi_accum = 0.0
    conscious_steps = 0

    # Reset recurrent state between episodes
    tectum.h_state = None
    tectum.z_state = None
    if hasattr(action_core, 'pfc_hidden'):
        action_core.pfc_hidden = None

    # Clear cross-episode gate state so feedback into attention doesn't
    # carry the previous episode's last gate values.
    if gate is not None and hasattr(gate, 'reset_episode'):
        gate.reset_episode()

    if metrics_logger is not None:
        metrics_logger.reset_episode_state()

    # Reset TPM at episode start so phi tracks within-episode dynamics
    if hasattr(workspace, 'iit_metrics'):
        workspace.iit_metrics.reset_tpm()

    prev_action = None
    prev_phi = 0.0
    prev_phi_riiu = 0.0  # parallel pathway, separate from pyphi prev_phi
    prev_env_reward = 0.0  # actual env reward (not shaped) for emotion delta
    # Phi-sampling state: pyphi runs only every Nth step; in between we
    # carry forward the last sampled value so logging stays well-defined.
    # phi_method is set explicitly in both branches ("pyphi"/"skipped") so
    # it does not need to be carried.
    last_phi = 0.0

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
        # Uses prev_env_reward (actual env reward, not shaped) for reward delta
        emotion = evaluate_emotion(vision_bid, 0.0, prev_env_reward)
        if audio_affect is not None:
            af = audio_affect["acoustic_features"]
            spectral_flux = af[0, 4].item() if af.shape[1] > 4 else 0.0
            roughness = af[0, 2].item() if af.shape[1] > 2 else 0.0
            emotion["arousal"] = float(np.clip(emotion["arousal"] + spectral_flux * 0.3, 0, 1))
            emotion["valence"] = float(np.clip(emotion["valence"] - roughness * 0.2, -1, 1))

        # Semantic pathway: requires Qwen2-VL embeddings to produce a meaningful
        # signal. Without Qwen2-VL loaded, the semantic channel bids 0 and does
        # not participate in workspace competition. Padding tectum content to
        # 1536-D would be circular (competing against itself).
        semantic_content = torch.zeros(1, config["workspace_dim"], device=device)
        semantic_bid = 0.0

        # Memory retrieval: use previous broadcast to find similar past experiences
        # Memory bid scales with retrieval relevance (more relevant = higher bid)
        memory_bid = 0.1
        if memory is not None and previous_broadcast is not None:
            try:
                query = previous_broadcast.detach().view(-1)
                similar = memory.get_similar_experiences(query, emotion, k=1)
                if similar and similar[0].get("score", 0.0) > 0.0:
                    memory_bid = min(0.6, 0.1 + similar[0]["score"] * 0.5)
            except Exception:
                pass  # graceful fallback to default bid

        raw_bids = {
            "vision": max(0.0, min(1.0, vision_bid)),
            "audio": max(0.0, min(1.0, audio_bid_raw)),
            "memory": memory_bid,
            "body": 0.15 if (self_model is not None and
                              self_model.state.interoceptive_state.get("energy", 1.0) < 0.4) else 0.05,
            "semantic": max(0.0, min(1.0, semantic_bid)),
        }

        # --- Affective modulation: emotion shapes which modules win ---
        # Valence field biases bids (positive -> approach, negative -> threat)
        # Arousal-threshold coupling adjusts GNW ignition threshold
        # Build interoceptive state: prefer self-model (tracks homeostatic dynamics
        # across steps), fall back to env info for environments that report battery.
        interoceptive_state = None
        if self_model is not None:
            action_np = np.array(prev_action) if prev_action is not None and not isinstance(prev_action, np.ndarray) else prev_action
            self_model.update_self_model(
                current_state={},
                attention_level=prev_phi,
                action=action_np,
                emotional_state=emotion,
            )
            interoceptive_state = dict(self_model.state.interoceptive_state)
            # Sync env battery into self-model energy when available
            if isinstance(info, dict) and "battery" in info:
                interoceptive_state["energy"] = float(info["battery"])
                self_model.state.interoceptive_state["energy"] = float(info["battery"])
        elif isinstance(info, dict):
            if "interoceptive_state" in info:
                interoceptive_state = info["interoceptive_state"]
            elif "battery" in info:
                battery = float(info["battery"])
                interoceptive_state = {
                    "energy": battery,
                    "fatigue": max(0.0, 1.0 - battery) * 0.5,
                    "damage": 0.0,
                }

        # Affective modulation now happens inside the workspace's
        # run_competition (called per cycle from reentrant.settle), driven by
        # the explicit pad_state and interoceptive_state we pass through.
        # raw_bids is passed straight in; workspace handles the modulation.

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
        # Ablation: pass None for both PAD signals so the workspace's
        # affective modulator and reentrant cycles run without embodiment
        # input. Tests whether the embodiment-affect loop wired in 2026-04-27
        # has measurable effect on phi/EI/reward dynamics.
        ablate_pad = config.get("ablate_pad_loop", False)
        settle_result = reentrant.settle(
            workspace=workspace,
            specialists=specialists,
            initial_bids=raw_bids,
            payloads=payloads,
            goal_vector=torch.tensor([1.0, -1.0, 1.0], device=device),
            pad_state=None if ablate_pad else emotion,
            interoceptive_state=None if ablate_pad else interoceptive_state,
        )

        broadcast_content = settle_result.broadcast_content
        is_conscious = settle_result.is_conscious
        sync_r = getattr(workspace, 'last_sync_R', 0.0)

        # GlobalWorkspace.run_competition returns broadcast_content as a dict
        # built from each winning module's payload (see global_workspace.py
        # lines 205-219). Vision payload is {"tensor": tectum_content,
        # "source": "tectum", "capsule_poses": ..., "capsule_activities": ...},
        # so the actual content tensor lives under the "tensor" key.
        # Subconscious cycles return {} -> zero broadcast.
        #
        # Detach: the extracted tensor still carries gradient history back into
        # the RSSM via tectum_content. Without detach, the tectum optimizer's
        # in-place parameter step (every 5 steps) invalidates the BPTT graph
        # for the gate diversity backward that runs after it, raising
        # "variable needed for gradient computation has been modified by an
        # inplace operation". The gate is trained by its own diversity loss
        # (operating on gate_values_tensor, which has its own clean graph
        # rooted at gate.parameters()), and the reward predictor trains the
        # tectum directly via tectum_content. So the broadcast does not need
        # to carry gradient into downstream consumers.
        if isinstance(broadcast_content, torch.Tensor):
            broadcast = broadcast_content.detach()
        elif (isinstance(broadcast_content, dict)
              and isinstance(broadcast_content.get("_fused"), torch.Tensor)):
            # Phase A path: attention-weighted fusion produces a single
            # _fused tensor that integrates all eligible module payloads.
            fused = broadcast_content["_fused"]
            if fused.dim() == 1:
                fused = fused.unsqueeze(0)
            broadcast = fused.detach().to(device)
        elif (isinstance(broadcast_content, dict)
              and isinstance(broadcast_content.get("tensor"), torch.Tensor)):
            broadcast = broadcast_content["tensor"].detach()
        else:
            broadcast = torch.zeros(1, config["workspace_dim"], device=device)

        broadcast_mag = float(broadcast.norm().item())

        # RIIU parallel phi: push broadcast into the sliding window and compute
        # the SVD-residual phi. Zero before the buffer is warm (first `rank+1`
        # pushes). Detached, used as a logged metric and (optionally) the
        # phi-delta reward source. When riiu_phis has multiple substrates
        # (--riiu-probe-all), each gets pushed independently from a
        # different activation source: broadcast (post-GNW), tectum_content
        # (pre-binding sensory), or audio_content (cochlear, requires
        # --enable-audio).
        phi_riiu_vals: dict[str, float] = {}
        if riiu_phis is not None:
            substrate_tensors = {
                "broadcast": broadcast,
                "tectum": tectum_content,
                "audio": audio_content,
            }
            for name, riiu_instance in riiu_phis.items():
                src_tensor = substrate_tensors.get(name)
                if src_tensor is None:
                    phi_riiu_vals[name] = 0.0
                    continue
                riiu_instance.push(src_tensor.view(-1))
                phi_riiu_vals[name] = (
                    riiu_instance.compute_value() if riiu_instance.is_warm else 0.0
                )
        # Reward source: whichever substrate the CLI flag selected.
        # Fall back to 0.0 if that substrate is not active (e.g. requested
        # 'tectum' without --riiu-probe-all and not as the sole source).
        phi_riiu_val = phi_riiu_vals.get(riiu_source, 0.0)

        # ConsciousnessGate: compute 5 causal gate values from broadcast.
        # These drive IIT Phi (via empirical TPM) and EI measurement.
        # gate.last_gate_values_tensor preserves gradients so action loss
        # can backprop through the gate networks.
        gate_values_tensor = None
        if gate is not None:
            gate_input = broadcast.view(-1)[:config["workspace_dim"]]
            if gate_input.shape[0] < config["workspace_dim"]:
                gate_input = torch.nn.functional.pad(
                    gate_input, (0, config["workspace_dim"] - gate_input.shape[0])
                )
            gate_input_batched = gate_input.unsqueeze(0)
            _, gate_state_obj = gate(gate_input_batched)
            # Sampled pyphi: run the expensive Big Phi computation every
            # Nth step only. The TPM still updates every step via
            # update_from_gate_state. On non-sampled steps phi carries
            # forward; phi_method = "skipped" marks the row so analysis
            # can filter. See plan i-need-you-first-goofy-church for
            # the segfault-threshold rationale.
            sample_every = max(1, config.get("phi_sample_every", 5))
            if step % sample_every == 0:
                phi_result = workspace.iit_metrics.compute_phi_from_gate_state(gate_state_obj)
                # Report phi as the actual IIT result, not phi + sync_R * 0.1.
                # The old additive term made phi tautologically correlate
                # with sync_R (Phi-1 r=1.000 was a trivial identity, not a
                # scientific finding). Phi and sync_R are now independent.
                phi = phi_result.phi
                phi_method = phi_result.method
                last_phi = phi
            else:
                # Keep TPM warm without paying the pyphi MIP cost.
                workspace.iit_metrics.update_from_gate_state(gate_state_obj)
                phi = last_phi
                phi_method = "skipped"
            # Use the differentiable tensor directly from the gate (preserves grads)
            gate_values_tensor = gate.last_gate_values_tensor
        else:
            phi = settle_result.phi
            phi_method = "no_gate"

        # Stage 2: appraisal emotion (post-broadcast, content-specific).
        # Runs BEFORE action selection so the full conscious emotion drives the action.
        # env_reward=0.0 is correct here: reward is unknown before acting.
        emotion = evaluate_emotion(
            vision_bid, 0.0, prev_env_reward,
            broadcast=broadcast, qualia_mapper=workspace.qualia_mapper,
        )

        # Phi-weighted exploration: high phi = more exploitation, low phi = more exploration.
        # This creates a causal pathway: phi -> behavior -> reward -> phi changes.
        # phi is typically in [0, 0.1] range, so we scale by 10x to get a meaningful effect.
        arousal = emotion["arousal"]
        phi_exploration_scale = max(0.2, 1.0 - phi * 10.0)  # phi=0.05 -> scale=0.5
        effective_arousal = arousal * phi_exploration_scale
        action, value = action_core.select_action(
            broadcast, emotion_arousal=effective_arousal, rpe_cache=0.0
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

            # TDANN topographic loss: enforce spatial self-organization on tectum features
            topo_loss = torch.tensor(0.0, device=device)
            obs_map = getattr(tectum, '_last_obs_map', None)
            if obs_map is not None and obs_map.shape[-1] >= 4:
                topo_loss = topographic_spatial_loss(obs_map, alpha=0.25)

            total_tectum_loss = pred_loss + topo_loss

            tectum_optimizer.zero_grad()
            reward_optimizer.zero_grad()
            total_tectum_loss.backward(retain_graph=True)
            tectum_optimizer.step()
            reward_optimizer.step()
            # Force-detach recurrent state immediately after the optimizer
            # mutates tectum parameters in-place. Without this, a later
            # backward call through the BPTT-retained graph would reference
            # the pre-step parameter version and raise:
            # "variable needed for gradient computation has been modified
            # by an inplace operation". This effectively closes the BPTT
            # window at every tectum optimizer step regardless of bptt_window.
            if tectum.h_state is not None:
                tectum.h_state = tectum.h_state.detach()
            if tectum.z_state is not None:
                tectum.z_state = tectum.z_state.detach()
            tectum._steps_since_detach = 0

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

        # --- Gate diversity loss ---
        # Penalize gate outputs near 0.5 (encourage decisiveness).
        # Uses the differentiable gate_values_tensor from the forward pass
        # (preserves causal structure, no need to re-forward individual nets).
        # Ablation: skip the loss entirely so the gate is shaped only by
        # task gradients (reward predictor + RND), testing whether the
        # diversity prior helps or hurts emergent gate dynamics.
        if (gate is not None and gate_values_tensor is not None
                and step % 5 == 0
                and not config.get("ablate_gate_diversity", False)):
            # Per-neuron binarization (weight 0.05). Pushes each gate output
            # toward {0, 1}. Necessary because IIT operates on binary states
            # and the empirical TPM needs sharp transitions; a gate that
            # stays near 0.5 produces low-entropy binarization with no
            # useful structure.
            #
            # 2026-05-13 attempt to AUGMENT this with a temporal diversity
            # term failed smoke testing. Two designs were tried, both at
            # 5-episode dark_room scale:
            #   1. -var(current vs buffer mean): 42 unique vs 49 ablated.
            #   2. Per-neuron firing-rate entropy: 42 unique vs 50 ablated.
            # Both designs reduce diversity by ~15% at smoke scale because
            # they fight the binarization push. Reverting to per-neuron
            # alone. The IITMetrics._gate_buffer infrastructure is kept in
            # iit_phi.py for future experiments. The --ablate-gate-entropy
            # CLI flag is also kept (no-op currently) so the ablation
            # campaign command lines stay forward-compatible.
            #
            # The 200-episode phi collapse to 6.5e-05 observed in
            # runs/baseline_sampled_200 must be caused by something other
            # than the loss design. The ablation campaign will test which
            # of {memory replay, consolidation fix, gate feedback, RND
            # zero-on-reward} is the load-bearing component.
            gate_diversity_loss = -torch.log(
                torch.abs(gate_values_tensor - 0.5).clamp(min=0.01)
            ).mean()
            gate_loss = 0.05 * gate_diversity_loss
            tectum_optimizer.zero_grad()
            gate_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            tectum_optimizer.step()

        # --- Post-action emotion update (C5 fix) ---
        # Re-evaluate emotion with actual env_reward for memory storage
        # and next step's prev_reward. The pre-action emotion (computed above)
        # correctly drove action selection without future knowledge.
        emotion_post = evaluate_emotion(
            vision_bid, env_reward, prev_env_reward,
            broadcast=broadcast, qualia_mapper=workspace.qualia_mapper,
        )

        # --- Reward shaping (single pass, C4 fix) ---
        # Phi-delta intrinsic reward: consciousness integration change.
        # When RIIU is enabled, the reward source switches from pyphi-phi to
        # RIIU-phi so the policy learns from the high-variance pathway.
        # pyphi-phi continues to be computed and logged unchanged.
        delta_phi = phi - prev_phi
        if riiu_phis is not None:
            delta_phi_reward = phi_riiu_val - prev_phi_riiu
        else:
            delta_phi_reward = delta_phi
        # RND curiosity: novel broadcast states drive exploration
        curiosity_score = 0.0
        if rnd is not None:
            curiosity_score, rnd_loss = rnd(broadcast.detach())
            if rnd_optimizer is not None and step % 5 == 0:
                rnd_optimizer.zero_grad()
                rnd_loss.backward()
                rnd_optimizer.step()
            # Zero curiosity when agent receives positive external reward so
            # exploration drive doesn't pull agent away from rewarding states.
            # Ablation: let RND fire even on rewarding states, restoring the
            # pre-eeaec28 behavior so the impact of the suppression can be
            # measured.
            if env_reward > 0 and not config.get("ablate_rnd_zero_on_reward", False):
                curiosity_score = 0.0

        # Additive intrinsic bonuses applied to env_reward. These go into
        # action_core.step() which applies emotion shaping ONCE internally.
        # delta_phi_reward equals delta_phi when RIIU is disabled, otherwise
        # it equals the RIIU phi delta.
        intrinsic_bonus = 0.5 * delta_phi_reward + 0.1 * curiosity_score
        reward_for_action_core = env_reward + intrinsic_bonus

        # Store ALL experiences with post-action emotion and phi priority
        if memory is not None:
            action_t = torch.tensor(action, dtype=torch.float, device=device) if not isinstance(action, torch.Tensor) else action
            memory.store_experience(
                state=broadcast.detach().view(-1),
                action=action_t.view(-1),
                reward=float(reward_for_action_core),
                emotion_values=emotion_post,
                attention_level=phi,
                priority=phi,
            )

        # Pass env_reward + intrinsic bonuses to action_core, which applies
        # emotion shaping internally (single pass, no double shaping).
        prev = previous_broadcast if previous_broadcast is not None else broadcast
        action_core.step(
            workspace_broadcast=prev,
            action=action,
            raw_reward=reward_for_action_core,
            next_broadcast=broadcast,
            done=done,
            emotion_state=emotion_post,
            attention_level=phi,
            narrative="",
        )

        if step > 0 and step % 10 == 0:
            action_core.update_policy()

        previous_broadcast = broadcast.detach().clone()
        prev_action = action
        prev_phi = phi
        prev_phi_riiu = phi_riiu_val
        # Track actual env_reward for next step's emotion delta (not shaped)
        prev_env_reward = env_reward
        reward_val = reward_for_action_core if isinstance(reward_for_action_core, (int, float)) else reward_for_action_core.item()
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
                phi_method=phi_method,
                phi_riiu=phi_riiu_val,
                phi_riiu_broadcast=phi_riiu_vals.get("broadcast", 0.0),
                phi_riiu_tectum=phi_riiu_vals.get("tectum", 0.0),
                phi_riiu_audio=phi_riiu_vals.get("audio", 0.0),
            ))

            # Insight detection: hash broadcast embedding for meaningful novelty signal
            # round(1) prevents float noise from making every state unique
            state_hash = str(hash(tuple(
                broadcast.detach().cpu().numpy().flatten().round(1)
            )))
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

    # Ablation flags. Each reverts exactly one Phase 3 or 2026-04-27 change.
    parser.add_argument("--ablate-memory-replay", action="store_true",
                        help="Skip the memory consolidation + replay block")
    parser.add_argument("--ablate-consolidation-fix", action="store_true",
                        help="Use legacy _merge_similar that drops state/action/emotion fields")
    parser.add_argument("--ablate-rnd-zero-on-reward", action="store_true",
                        help="Let RND curiosity fire even when env_reward > 0")
    parser.add_argument("--ablate-gate-diversity", action="store_true",
                        help="Skip ALL gate diversity losses (binarization + variance)")
    parser.add_argument("--ablate-gate-entropy", action="store_true",
                        help="Skip only the temporal variance term, keep the "
                             "per-neuron binarization push. Use to isolate "
                             "the contribution of the new variance loss "
                             "from the existing -log(|g-0.5|) term.")
    parser.add_argument("--ablate-gate-feedback", action="store_true",
                        help="Zero the gate_feedback projection in ConsciousnessGate.forward")
    parser.add_argument("--ablate-pad-loop", action="store_true",
                        help="Pass None for pad_state and interoceptive_state into reentrant.settle")
    parser.add_argument("--ablate-bptt", action="store_true",
                        help="Set tectum bptt_window=1 (one-step encoder, no truncated BPTT)")
    parser.add_argument("--phi-sample-every", type=int, default=5,
                        help="Run pyphi only every Nth step. State history "
                             "still updated every step so the TPM stays warm. "
                             "Cuts pyphi MIP calls N-fold to avoid the ~91k-call "
                             "segfault threshold in pyphi 1.x. Default 5.")

    # RIIU parallel phi pathway flags
    parser.add_argument("--enable-riiu", action="store_true",
                        help="Enable RIIU phi pathway (sliding-window SVD residual "
                             "on broadcast). When on, the phi-delta reward source "
                             "switches from pyphi to RIIU; pyphi-phi is still "
                             "logged for comparison. See docs/decisions/"
                             "2026_05_16_riiu_license.md.")
    parser.add_argument("--riiu-rank", type=int, default=16,
                        help="Truncated-SVD rank for RIIU. Default 16.")
    parser.add_argument("--riiu-window", type=int, default=64,
                        help="Sliding-window length for RIIU. Default 64. "
                             "Must exceed --riiu-rank.")
    parser.add_argument("--riiu-source", type=str, default="broadcast",
                        choices=["broadcast", "tectum", "audio"],
                        help="Which activation substrate drives the RIIU "
                             "phi-delta reward. Default 'broadcast' (the "
                             "post-GNW workspace broadcast tensor). 'tectum' "
                             "uses pre-binding sensory features; 'audio' "
                             "uses cochlear features and requires "
                             "--enable-audio.")
    parser.add_argument("--riiu-probe-all", action="store_true",
                        help="Instantiate three RIIUPhi pipelines (broadcast, "
                             "tectum, audio) in parallel and log all three "
                             "phi values per step. The --riiu-source flag "
                             "still selects which one drives reward. Used "
                             "for the 2026-05-17 substrate-probe experiment "
                             "(see ~/.claude/plans/let-s-plan-the-next-misty-parasol.md).")

    parser.add_argument("--seed", type=int, default=None,
                        help="Global RNG seed. None inherits ambient state. "
                             "Setting an int seeds python/numpy/torch and the "
                             "env.reset call on episode 0.")

    # Phase A of 2026-05-17 Phi-1 retest plan: attention-weighted broadcast
    # fusion. Default 'winner_take_all' preserves all existing test outputs
    # and reproducibility of prior runs.
    parser.add_argument("--broadcast-mode", type=str, default="winner_take_all",
                        choices=["winner_take_all", "attention_weighted"],
                        help="Workspace broadcast assembly mode. "
                             "'winner_take_all' (legacy default) iterates ignition "
                             "winners and merges their payloads. "
                             "'attention_weighted' computes a softmax-weighted "
                             "sum of all eligible module payloads, with weights "
                             "from AKOrN bound_bids. Makes phi-on-broadcast "
                             "structurally downstream of sync_R.")
    parser.add_argument("--attention-temperature", type=float, default=0.5,
                        help="Softmax temperature for attention-weighted fusion. "
                             "Lower = sharper, higher = more uniform. Default 0.5.")
    parser.add_argument("--attention-floor", type=float, default=0.05,
                        help="Minimum bound_bid for a module to be eligible "
                             "for fusion. Default 0.05.")

    args = parser.parse_args()

    if args.seed is not None:
        _set_global_seed(args.seed)
        logger.info(f"Global seed set to {args.seed}")

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
     reward_optimizer, workspace_optimizer, auditory_specialist,
     self_model, rnd, rnd_optimizer, consolidation_mgr,
     riiu_phis) = init_components(config)

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
    active_ablations = [k for k in (
        "ablate_memory_replay", "ablate_consolidation_fix",
        "ablate_rnd_zero_on_reward", "ablate_gate_diversity",
        "ablate_gate_feedback", "ablate_pad_loop", "ablate_bptt",
    ) if config.get(k)]
    logger.info(f"Active ablations: {active_ablations if active_ablations else 'none'}")

    rewards_history = []
    global_step = 0
    for ep in range(args.episodes):
        logger.info(f"Episode {ep + 1}/{args.episodes}")
        ep_reward, ep_steps, avg_phi, consciousness_ratio = run_episode(
            ep, config, tectum, workspace, reentrant,
            modulator, action_core, env,
            gate=gate, memory=memory,
            metrics_logger=metrics_logger, global_step_offset=global_step,
            tectum_optimizer=tectum_optimizer,
            reward_predictor=reward_predictor,
            reward_optimizer=reward_optimizer,
            workspace_optimizer=workspace_optimizer,
            auditory_specialist=auditory_specialist,
            self_model=self_model,
            rnd=rnd,
            rnd_optimizer=rnd_optimizer,
            riiu_phis=riiu_phis,
            riiu_source=config.get("riiu_source", "broadcast"),
        )
        global_step += ep_steps

        # Ablation: skip consolidation + replay entirely so the policy only
        # learns from the online action_core.update_policy() calls. Tests
        # whether the memory replay loop is load-bearing for reward and
        # phi dynamics.
        if not config.get("ablate_memory_replay", False):
            # --- Memory consolidation: decay relevance, merge similar, prune ---
            if memory is not None and memory.recent_experiences:
                for exp in memory.recent_experiences:
                    exp.setdefault("id", f"exp_{id(exp)}")
                    exp.setdefault("relevance", max(0.2, exp.get("priority", 1.0)))
                memory.recent_experiences = consolidation_mgr.consolidate(
                    memory.recent_experiences
                )

            # --- Memory replay: phi-prioritized policy update every 10 episodes ---
            if (ep + 1) % 10 == 0 and memory is not None and memory.recent_experiences:
                batch = consolidation_mgr.get_replay_batch(
                    memory.recent_experiences, k=16
                )
                if batch:
                    replay_metrics = action_core.replay_update(batch)
                    if replay_metrics:
                        logger.info(
                            f"  Replay update: loss={replay_metrics.get('replay_total_loss', 0):.4f}"
                        )

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
