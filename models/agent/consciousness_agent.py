import torch
import numpy as np
import logging
import time
from typing import Dict, Any, Tuple, Optional

# Components
from models.vision_language.qwen2.qwen2_integration import Qwen2VLIntegration
from models.core.global_workspace import GlobalWorkspace
from models.self_model.action_selection_core import ActionSelectionCore
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore
from models.narrative.narrative_generator import NarrativeGenerator
from models.self_model.self_representation_core import SelfRepresentationCore

logger = logging.getLogger(__name__)

class ConsciousnessAgent:
    """
    The Central Controller (The Self).
    Orchestrates the loop between Perception, Emotion, Consciousness, and Action.
    
    Architecture:
    1. Senses (Qwen2-VL) -> Percepts
    2. Emotion (Homeostasis) -> Affect
    3. Workspace (GNW) -> Conscious State (Ignition)
    4. Action (PPO) -> Behavior
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Initializing Consciousness Agent...")
        
        # 1. Perception (The Senses)
        # We use Qwen2-VL to turn raw pixels into semantic descriptions.
        self.vision_system = Qwen2VLIntegration(config.get("vision", {}))
        
        # Midbrain sensory integration (Fuses vision and spatial audio)
        from models.core.sensory_tectum import SensoryTectum
        self.sensory_tectum = SensoryTectum({
            "tectum_feature_dim": 1536, # Matches Qwen2-VL ViT dim
            "tectum_grid_size": 14,     # Default patch grid
            "workspace_dim": 256
        }).to(self.device)
        
        # Somatosensory Mapping (Body Schema & Proprioception)
        from models.self_model.embodiment_core import ProprioceptiveProcessor
        self.raw_state_dim = config.get("proprioception", {}).get("raw_dim", 40)
        self.body_processor = ProprioceptiveProcessor(
            raw_state_dim=self.raw_state_dim, 
            num_parts=10, 
            feature_per_part=8
        ).to(self.device)
        
        # 2. Memory & Emotion (The Self)
        self.memory = MemoryCore(config.get("memory", {}))
        self.emotion_shaper = EmotionalRewardShaper(config.get("emotion", {}))
        
        # 6. Action Selection (Prefrontal Cortex & Basal Ganglia)
        # Replaces generic RL with biological Go/No-Go pathways modulated by Dopamine
        self.action_core = ActionSelectionCore(
            config.get("action_selection", {}), 
            self.emotion_shaper, 
            self.memory
        )
        
        # 4. Consciousness (The Workspace)
        self.global_workspace = GlobalWorkspace(config.get("workspace", {}))
        
        # 5. Self-Model & Narrative (Phase 5)
        self.narrative_gen = NarrativeGenerator(config.get("narrative", {}))
        self.self_model = SelfRepresentationCore(config.get("self_model", {}))
        
        # Internal State
        self.current_emotion = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        self.anxiety_level = 0.0
        self.step_count = 0
        self.last_rpe = 0.0  # Dopamine RPE from previous step, fed back into BG
        self.previous_broadcast = None  # Track previous workspace broadcast for temporal differentiation
        
        # Simple Text Encoder for PPO State (Placeholder for a better semantic encoder)
        # Maps text descriptions to the state_dim expected by PPO
        self.state_dim = config.get("reinforcement", {}).get("state_dim", 128)
        self.text_encoder = torch.nn.Sequential(
            torch.nn.Linear(768, self.state_dim), # Assuming we get 768 dim embeddings (e.g. BERT-like)
            torch.nn.ReLU()
        ).to(self.device)
        # We'll use a random projection if no real text encoder is loaded for this prototype
        self.text_projection = torch.randn(768, self.state_dim).to(self.device)

    def step(self, observation: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main Cognitive Cycle.
        """
        self.step_count += 1
        start_time = time.time()
        
        # --- 1. Perception ---
        # Analyze scene logically with Qwen2-VL (for broadcast payload)
        try:
            visual_description = self.vision_system.analyze_scene(observation, prompt="Describe the light level and safety.")
            # Map raw pixel patches to a 2D grid for the Sensory Tectum
            vision_grid = self.vision_system.get_visual_embeddings(observation, return_spatial_grid=True)
            # Add batch dimension [1, C, H, W]
            vision_grid = vision_grid.unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Vision failure: {e}")
            visual_description = "darkness and uncertainty"
            vision_grid = torch.zeros(1, 1536, 14, 14, device=self.device)

        # Generate a dummy audio spatial vector for now (e.g., straight ahead)
        audio_spatial = torch.zeros(1, 1536, 2, device=self.device)
        
        # Process through Sensory Tectum (RSSM) to get surprise-based bidding payload
        tectum_content, vision_surprise_bid = self.sensory_tectum(vision_grid, audio_spatial)
        
        # --- 1b. Somatosensory Processing ---
        # Extract proprioception from observation dict if available, else dummy
        if isinstance(observation, dict) and 'proprioception' in observation:
            raw_proprioception = observation['proprioception'].to(self.device)
            collision_flags = observation.get('collisions', torch.zeros(1, 10)).to(self.device)
        else:
            raw_proprioception = torch.zeros(1, self.raw_state_dim, device=self.device)
            collision_flags = torch.zeros(1, 10, device=self.device)
            
        body_schema, body_bid = self.body_processor(raw_proprioception, collision_flags)
        
        # --- 2. Emotion (Fast Path) ---
        # Evaluate "Reflexive" emotional response to the percept
        # In "Dark Room", darkness = high arousal (anxiety)
        # This logic mimics the Amygdala (fast, pattern-matched)
        reflex_emotion = self._evaluate_reflex_emotion(visual_description)
        self.current_emotion = reflex_emotion
        
        # Ensure we have valid bids between 0.0 and 1.0
        vision_bid = max(0.0, min(1.0, vision_surprise_bid))
        emotion_bid = abs(reflex_emotion["arousal"]) # High arousal = high bid
        
        # --- 3. Consciousness (Global Workspace) ---
        # Submit bids and semantic payloads to the workspace
        # Bids dict defines who gets access based on scalar values
        bids = {
            "vision": vision_bid,
            "emotion": emotion_bid,
            "memory": 0.1, # Low baseline bid
            "audio": 0.0, # No pure audio semantic stream yet
            "body": body_bid
        }
        
        # Payloads are the complex tensors/strings broadcasted if that module wins
        payloads = {
            "vision": visual_description,
            "emotion": reflex_emotion,
            "memory": "No active recall", # Placeholder
            "body": "Physical state updated" # In reality, we'd pass the schema tensor, using string for logging prototype
        }
        
        # Calculate Goal Vector (Homeostasis) - Agent wants High Valence, Low Arousal
        goal_vector = torch.tensor([1.0, -1.0, 1.0], device=self.device) # Target: [Valence=1, Arousal=-1, Dominance=1]
        
        # Run GNW Competition
        # Pass explicit bids and payloads (bypasses legacy evaluate_salience polling)
        broadcast_content, winners = self.global_workspace.run_competition(
            inputs={},  # Legacy param (unused when explicit bids provided)
            goal_vector=goal_vector, 
            bids=bids, 
            payloads=payloads
        )
        
        # Check Ignition
        is_conscious = self.global_workspace.state.is_conscious
        phi = self.global_workspace.state.phi_value
        
        # --- 4. Action Selection (PFC & Basal Ganglia) ---
        # State Construction:
        # We need to feed a vector to the PFC. 
        # If Conscious: Use the Broadcast Content (Integrated)
        # If Zombie: Use the Raw Percept (Reflex/Heuristic fallback)
        
        # Currently, broadcast_content is a tensor [B, workspace_dim] from Sensory Tectum
        if not is_conscious or not isinstance(broadcast_content, torch.Tensor):
            # Fallback to a zero tensor if no active broadcast (Zombie mode or error)
            workspace_dim = self.config.get("workspace", {}).get("workspace_dim", 256) # Corrected path for workspace_dim
            broadcast_tensor = torch.zeros(1, workspace_dim, device=self.device)
        else:
            broadcast_tensor = broadcast_content
            
        # Add arousal to scale exploration in Basal Ganglia (Panicked vs Calm)
        arousal = self.emotion_shaper.state.arousal # Use emotion_shaper
        action, value = self.action_core.select_action(broadcast_tensor, emotion_arousal=arousal, rpe_cache=self.last_rpe)
        
        # Determine reward (Placeholder: in a real env, this comes from the step)
        # We simulate a reward based on satisfying the homeostasis goal
        distance_to_goal = torch.norm(goal_vector - torch.tensor([
            self.emotion_shaper.state.valence, # Use emotion_shaper
            self.emotion_shaper.state.arousal, # Use emotion_shaper
            self.emotion_shaper.state.dominance # Use emotion_shaper
        ], device=self.device))
        
        simulated_reward = -distance_to_goal.item()
        
        # Use previous broadcast for temporal difference (RPE = r + gamma*V(s') - V(s))
        # On first step, fall back to current broadcast
        prev_broadcast = self.previous_broadcast if self.previous_broadcast is not None else broadcast_tensor
        
        # --- 4.5. Self-Model & Narrative (Phase 5) ---
        # Generate conscious narrative from the active broadcast and emotional state
        narrative_text = self.narrative_gen.generate_from_workspace(
            broadcast=visual_description if not is_conscious else "Conscious of " + visual_description,  # Simplified for payload
            emotional_state=self.current_emotion,
            action=action
        )
        
        # Update the autobiographical / capability self-model
        self_update_info = self.self_model.update_self_model(
            current_state={"prediction_outcomes": {}},  # Will be expanded with real prediction outcomes
            attention_level=phi,
            action=action,
            emotional_state=self.current_emotion,
            rpe=self.last_rpe,
            social_feedback=None
        )
        
        # Update Core Tracking
        step_metrics = self.action_core.step(
            workspace_broadcast=prev_broadcast,
            action=action,
            raw_reward=simulated_reward,
            next_broadcast=broadcast_tensor,  # Current is the "next" relative to previous
            done=False,
            emotion_state=self.current_emotion,
            attention_level=phi,
            narrative=narrative_text
        )
        
        # Track for next iteration
        self.previous_broadcast = broadcast_tensor.detach().clone()
        self.last_rpe = step_metrics.get("dopamine_rpe", 0.0)
        
        # Periodic Policy Update
        if self.memory.episodic.size > 10:
            training_metrics = self.action_core.update_policy()
        else:
            training_metrics = {}
        
        # --- 5. Return ---
        info = {
            "description": visual_description,
            "narrative": narrative_text,
            "emotion": self.current_emotion,
            "is_conscious": is_conscious,
            "phi": phi,
            "qualia": self.global_workspace.get_unity_metrics()[3], # Qualia Vector
            "action_value": value,
            "latency": time.time() - start_time,
            "self_model_id": self.self_model.state.id
        }
        
        return action, info

    def update(self, 
               state: np.ndarray, 
               action: np.ndarray, 
               reward: float, 
               next_state: np.ndarray, 
               done: bool, 
               info: Dict[str, Any]):
        """
        Learning Step (Post-Action).
        Feeds result back to Reinforcement Core.
        """
        # Convert raw numpy inputs to tensors/embeddings matching step() logic
        # Note: In a real loop, we'd cache the tensors from step() to avoid re-encoding
        # For this prototype, we re-encode or assume caller handles it.
        # But wait, PPO update needs tensors.
        
        # We'll trust the RL Core to handle the buffering if we pass the right data.
        # rl_core.step() takes (state, action, reward, next_state...)
        
        # Re-encode for consistency (Optimization: pass tensors from step return)
        state_vec = self._encode_text_to_state(info["description"]) # Approximation
        # Next state needs encoding too? 
        # In "Dark Room", next_state image comes from environment *after* step.
        # We might skip re-encoding 'next_state' here and let RL Core handle sparse rewards 
        # or require the Training Loop to call agent.encode(next_obs).
        
        # Simplified: We just pass the values to RL Core's step function
        # The RL Core stores them.
        
        # We pass the *shaped* emotional state
        self.rl_core.step(
            state=state_vec,
            action=action,
            raw_reward=reward,
            next_state=state_vec, # Placeholder: PPO needs real next state, usually handled in training loop
            done=done,
            emotion_state=self.current_emotion,
            attention_level=self.global_workspace.state.broadcast_strength,
            narrative=info["description"]
        )
        
        # Trigger PPO Update if buffer is full
        metrics = self.rl_core.update_policy()
        return metrics

    def _evaluate_reflex_emotion(self, description: str) -> Dict[str, float]:
        """
        Heuristic 'Amygdala'.
        Decodes text to basic PAD (Pleasure-Arousal-Dominance) values.
        """
        description = description.lower()
        valence = 0.0
        arousal = 0.0
        dominance = 0.0
        
        # Dark Room Heuristics
        if "dark" in description or "black" in description or "nothing" in description:
            valence = -0.8
            arousal = 0.8  # Anxiety
            dominance = -0.5 # Helpless
        elif "light" in description or "bright" in description:
            valence = 0.8
            arousal = -0.2 # Relief
            dominance = 0.5 # Control
            
        return {"valence": valence, "arousal": arousal, "dominance": dominance}

    def _encode_text_to_state(self, text: str) -> torch.Tensor:
        """
        Dummy Semantic Encoder. 
        In production, use a frozen BERT/CLIP text encoder.
        Here, we hash the string to a random but consistent vector for 'Blind' testing.
        """
        # Deterministic hash seed
        seed = hash(text) % (2**32)
        torch.manual_seed(seed)
        return torch.randn(self.state_dim, device=self.device)
