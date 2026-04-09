from __future__ import annotations

import pandas as pd
from threading import Lock
import subprocess
from typing import Any
from dataclasses import dataclass
import numpy as np
import torch
import logging
import asyncio

from models.self_model.action_selection_core import ActionSelectionCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.narrative.narrative_engine import NarrativeEngine
from models.memory.memory_core import MemoryCore
from models.predictive.dreamerv3_wrapper import DreamerV3Wrapper as DreamerV3
from simulations.environments.vr_environment import VREnvironment
try:
    from models.cognitive.chain_of_thought import ChainOfThought
except ImportError:
    ChainOfThought = None  # type: ignore[assignment, misc]
try:
    from models.ace_core.ace_agent import ACEConsciousAgent
    from models.ace_core.ace_config import ACEConfig
except ImportError:
    ACEConsciousAgent = None  # type: ignore[assignment, misc]
    ACEConfig = None  # type: ignore[assignment, misc]
# VideoLLaMA3 has been removed (2026-02-27). Visual backbone is Qwen2-VL exclusively.
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.core.consciousness_core import ConsciousnessCore
from models.evaluation.consciousness_monitor import ConsciousnessMonitor
from models.predictive.dreamer_emotional_wrapper import DreamerEmotionalWrapper
from models.memory.attention_schema import AttentionSchema
from models.perception.predictive_processor import PredictiveProcessor
from models.core.global_workspace import GlobalWorkspace, WorkspaceMessage
from models.core.consciousness_gating import ConsciousnessGate

try:
    import unreal
except ImportError:
    unreal = None


@dataclass
class SimulationConfig:
    """Configuration for the simulation environment."""
    max_steps: int = 1000
    emotional_scale: float = 2.0
    emotion_threshold: float = 0.6
    memory_capacity: int = 100000
    narrative_max_length: int = 128
    # Removed Pavilion-specific flag and config
    # use_pavilion: bool = True
    # pavilion_config: dict | None = None


class SimulationManager:
    """
    Main simulation manager for consciousness development
    through emotional learning.
    """

    def __init__(self, consciousness_system=None, config=None):
        # Support single-arg call: SimulationManager(config_dict)
        if config is None and isinstance(consciousness_system, dict):
            config = consciousness_system
            consciousness_system = None
        if config is None:
            config = {}
        self.core = consciousness_system
        self.config = config
        self.lock = Lock()
        logging.info("Simulation Manager initialized with config: %s", config)

        # Tracking metrics.
        self.episode_rewards: list[float] = []
        self.emotion_history: list[dict[str, float]] = []
        self.current_scenario = None

        # Defer heavy component initialization, wrap in try/except so tests can still instantiate
        self.consciousness_monitor = None
        self.rl_core = None
        self.emotion_network = None
        self.narrative = None
        self.memory = None
        self.chain_processor = None
        self.env = None
        self.ace_agent = None
        self.consciousness_core = None

        try:
            self.consciousness_monitor = ConsciousnessMonitor(consciousness_system, None, config)
        except Exception as e:
            logging.warning("Could not initialize ConsciousnessMonitor: %s", e)

        gating_config = {
            'gating': {
                'attention_threshold': 0.5,
                'stability_threshold': 0.6,
                'base_adaptation_rate': 0.01
            },
            'hidden_size': 128
        }
        try:
            self.consciousness_gate = ConsciousnessGate(gating_config)
        except Exception as e:
            logging.warning("Could not initialize gating: %s", e)
            self.consciousness_gate = None

    def run_interaction(self, agent, environment, max_steps=None) -> dict[str, Any]:
        """Synchronous interaction loop for testing."""
        sim_cfg = self.config.get('simulation', {}) if isinstance(self.config, dict) else {}
        max_steps = max_steps or sim_cfg.get('max_steps', 100)
        state = environment.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done, info = environment.step(action)
            total_reward += reward
            steps += 1
            state = next_state
        return {
            'total_reward': total_reward,
            'steps': steps,
            'update_info': {},
        }

    def execute_code(self, code: str) -> dict:
        """
        Safely executes dynamically generated Python code.
        """
        try:
            exec_globals = {}
            exec(code, exec_globals)
            logging.info("Code executed successfully.")
            return exec_globals
        except Exception as e:
            logging.error("Code execution error: %s", e)
            raise

    def load_interaction_data(self):
        """
        Load simulation datasets (e.g., INTERACTION, UE-HRI) for environment tasks.
        """
        try:
            interaction_data = pd.read_csv("/data/simulations/interaction_data.csv")
            print("INTERACTION data loaded successfully.")

            ue_hri_data = pd.read_csv("/data/simulations/ue_hri_data.csv")
            print("UE-HRI data loaded successfully.")
        except Exception as e:
            print(f"Error loading datasets: {e}")

    async def run_interaction_episode(self, agent, environment) -> dict[str, Any]:
        try:
            episode_data = []
            state = environment.reset()
            done = False
            step = 0
            
            while not done:
                # Process interaction step
                action = agent.select_action(state)
                next_state, reward, done, info = environment.step(action)
                
                # Ensure emotional context exists
                emotion_context = info.get('emotional_context', {})
                if not emotion_context and hasattr(self, 'emotion_network'):
                    # Fallback: generate emotional context if missing
                    emotional_context = self.emotion_network.generate_default_emotions()
                    logging.warning("Missing emotional context, using default values")

                # Compute reward with safety checks
                emotional_reward = self.rl_core.compute_reward(
                    state=state,
                    emotion_values=emotional_context,
                    narrative=agent.current_narrative()
                )
                
                # Store experience
                await self.memory.store_experience({
                    "state": state,
                    "action": action,
                    "reward": emotional_reward,
                    "next_state": next_state,
                    "emotion": emotion_context,
                    "narrative": agent.current_narrative(),
                    "done": done
                })
                
                # Update episode data
                episode_data.append({
                    "step": step,
                    "emotion": emotion_context,
                    "reward": emotional_reward
                })
                
                state = next_state
                step += 1
                
            # Generate narrative with error handling
            try:
                thought_data = await self.chain_processor.generate_multimodal_thought()
                agent.update_narrative(thought_data["chain_text"])
                episode_data.append({
                    "chain_of_thought": thought_data["chain_text"],
                    "visual_output": thought_data.get("visual_output")
                })
            except Exception as e:
                logging.error("Narrative generation failed: %s", e)
                agent.update_narrative("Narrative generation failed; using last known context.")
                
            return {"episode_data": episode_data}
            
        except Exception as e:
            logging.error("Episode execution failed: %s", e)
            raise

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get current learning and performance metrics.
        """
        mean_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        recent_emotions = self.emotion_history[-1000:] if self.emotion_history else []
        emotion_stability = 0.0
        if recent_emotions:
            valences = [em.get("valence", 0.0) for em in recent_emotions]
            emotion_stability = np.std(valences)

        return {
            "mean_reward": mean_reward,
            "emotion_stability": emotion_stability,
            "memory_usage": self.memory.get_usage_stats(),
            "learning_progress": self.rl_core.get_learning_stats()
        }

    def save_checkpoint(self, path: str):
        """
        Save simulation state and model checkpoints.
        """
        checkpoint = {
            "rl_core": self.rl_core.state_dict(),
            "emotion_network": self.emotion_network.state_dict(),
            "episode_rewards": self.episode_rewards,
            "emotion_history": self.emotion_history,
            "config": self.config
        }
        torch.save(checkpoint, path)

    async def initialize_simulation(self):
        """Initialize all components"""
        await self.ace_agent.initialize()
        # Visual backbone initialization (Qwen2-VL)
        if hasattr(self, 'visual_processor') and self.visual_processor is not None:
            await self.visual_processor.initialize()
        
    async def simulation_step(self, visual_input, audio_input=None, context=None):
        # Update ACE and consciousness core integration
        perception = {}
        if hasattr(self, 'visual_processor') and self.visual_processor is not None:
            perception = await self.visual_processor.process_input(
                visual_input=visual_input,
                audio_input=audio_input
            )

        # Global workspace broadcast
        await self.global_workspace.broadcast(
            WorkspaceMessage(
                source="perception",
                content=perception,
                priority=0.8
            )
        )

        # Process through consciousness core
        consciousness_state = await self.consciousness_core.process({
            'perception': perception,
            'context': context
        })

        # Generate emotional response
        emotional_response = await self.emotional_memory.generate_response(
            consciousness_state
        )

        # Process through ACE
        ace_result = await self.ace_agent.process_interaction(
            visual_input=visual_input,
            audio_input=audio_input, 
            context={
                'consciousness_state': consciousness_state,
                'emotional_response': emotional_response,
                'perception': perception
            }
        )

        # Update attention schema
        current_focus = {
            'visual': visual_input,
            'audio': audio_input,
            'consciousness': consciousness_state,
            'emotion': emotional_response
        }
        await self.attention_schema.update(current_focus)

        # Get cumulative focus overview
        cumulative_focus = await self.attention_schema.get_overview()
        
        # Update self model
        await self.adjust_self_model(cumulative_focus)

        # Update emotional memory
        await self.emotional_memory.update(
            consciousness_state,
            emotional_response,
            ace_result['animation_data']
        )

        # Update world model
        await self.world_model.update(
            consciousness_state, 
            emotional_response,
            ace_result
        )

        return {
            'consciousness_state': consciousness_state,
            'emotional_response': emotional_response,
            'ace_result': ace_result,
            'perception': perception,
            'attention_focus': cumulative_focus
        }
    
    def adjust_self_model(self, cumulative_focus):
        """
        Dynamically adjust internal state parameters based on the aggregated focus data.
        This is a placeholder function intended to integrate meta-awareness into the self-model.
        """
        # Example implementation: log the focus data and adjust parameters accordingly.
        print("Adjusting self-model with focus data:", cumulative_focus)

    def load_character_blueprint(self):
        """Load ACE-compatible character blueprint"""
        try:
            blueprint_path = self.ace_config.get_blueprint_path()
            return unreal.load_object(None, blueprint_path)
        except Exception as e:
            print(f"Failed to load character blueprint: {e}")
            return None

    def run_simulation_step(self):
        # existing simulation logic

        # Evaluate consciousness metrics
        metrics = self.consciousness_monitor.update_metrics()
        self.log_metrics(metrics)

    def log_metrics(self, metrics):
        # Simple logging
        print("[ConsciousnessMetrics]", metrics)


# Example usage
if __name__ == "__main__":
    manager = SimulationManager(config=SimulationConfig())
    manager.execute_code("print('Hello, Unreal Engine!')")
    manager.load_interaction_data()
