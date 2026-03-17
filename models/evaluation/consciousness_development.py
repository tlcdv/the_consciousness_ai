from __future__ import annotations

# models/evaluation/consciousness_development.py
from dataclasses import dataclass, field
import torch
import time
import logging
import numpy as np # Added for metric calculations

# Assuming reward_shaping is directly under emotion
from ..emotion.reward_shaping import EmotionalRewardShaper
# Assuming memory_core is directly under memory
from ..memory.memory_core import MemoryCore
# Assuming consciousness_metrics is in the same directory (evaluation)
from .consciousness_metrics import ConsciousnessMetrics
# Assuming dreamer_emotional_wrapper is directly under predictive
from ..predictive.dreamer_emotional_wrapper import DreamerEmotionalWrapper
# Correct import based on user feedback and likely structure
from ..self_model.self_representation_core import SelfRepresentationCore
# Import ConsciousnessCore type hint if possible, or use 'object'/'Any'
# from ..core.consciousness_core import ConsciousnessCore # Ideal, if import works
from typing import Any # Use Any if direct import is problematic

@dataclass
class DevelopmentMetrics:
    """Tracks consciousness development metrics"""
    emotional_awareness: float = 0.0
    memory_coherence: float = 0.0
    attention_level: float = 0.0 # Assumed to be provided externally
    behavioral_adaptation: float = 0.0 # Based on learning progress
    survival_success: float = 0.0 # Based on episode outcomes
    last_updated: float = field(default_factory=time.time)

class ConsciousnessDevelopment:
    """
    Manages and evaluates consciousness development through interaction
    with core components like emotion, memory, and reinforcement learning.
    """

    def __init__(self, config: dict, core_module: Any | None = None): # Added core_module dependency
        """
        Initializes the ConsciousnessDevelopment module.

        Args:
            config: A dictionary containing configuration for all sub-modules.
            core_module: An instance of ConsciousnessCore (or similar) needed by some metrics.
        """
        self.config = config
        self.core_module = core_module # Store the core module instance

        # --- Core components ---
        try:
            self.dreamer = DreamerEmotionalWrapper(config.get('dreamer_config', {}))
            logging.info("DreamerEmotionalWrapper initialized in ConsciousnessDevelopment.")
        except Exception as e:
            logging.error(f"Failed to initialize DreamerEmotionalWrapper: {e}", exc_info=True)
            self.dreamer = None

        try:
            self.reward_shaper = EmotionalRewardShaper(config.get('reward_config', {}))
            logging.info("EmotionalRewardShaper initialized in ConsciousnessDevelopment.")
        except Exception as e:
            logging.error(f"Failed to initialize EmotionalRewardShaper: {e}", exc_info=True)
            self.reward_shaper = None

        try:
            self.memory = MemoryCore(config.get('memory_config', {}))
            logging.info("MemoryCore initialized in ConsciousnessDevelopment.")
        except Exception as e:
            logging.error(f"Failed to initialize MemoryCore: {e}", exc_info=True)
            self.memory = None

        try:
            self.self_model = SelfRepresentationCore(config.get('self_model_config', {}))
            logging.info("SelfRepresentationCore initialized in ConsciousnessDevelopment.")
        except Exception as e:
            logging.error(f"Failed to initialize SelfRepresentationCore: {e}", exc_info=True)
            self.self_model = None

        # ConsciousnessMetrics initialization requires core_module and memory system
        if self.core_module is None:
             logging.warning("ConsciousnessCore instance not provided to ConsciousnessDevelopment; some metrics (PCI, SelfAwareness) might be limited.")
        if self.memory is None:
             logging.warning("MemoryCore instance not available for ConsciousnessMetrics initialization.")

        try:
            self.consciousness_metrics = ConsciousnessMetrics(
                config.get('metrics_config', {}),
                core_module=self.core_module,
                memory_system=self.memory # Pass the initialized memory system
            )
            logging.info("ConsciousnessMetrics initialized in ConsciousnessDevelopment.")
        except Exception as e:
            logging.error(f"Failed to initialize ConsciousnessMetrics: {e}", exc_info=True)
            self.consciousness_metrics = None


        # Initialize metrics tracking
        self.metrics = DevelopmentMetrics()
        logging.info("ConsciousnessDevelopment initialized.")


    def process_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        emotion_values: dict[str, float],
        attention_level: float, # Assuming this comes from ConsciousnessCore or attention module
        done: bool,
        timestamp: float = None,
    ) -> dict:
        """
        Processes a single step of experience, updates models, shapes reward, and calculates metrics.
        """
        if timestamp is None:
            timestamp = time.time()
        logging.debug(f"Processing experience at timestamp {timestamp}")

        # 1. Shape the reward using emotional context
        shaped_reward = reward # Default to raw reward if shaper fails
        if self.reward_shaper:
            try:
                shaped_reward = self.reward_shaper.shape_reward(
                    raw_reward=reward,
                    emotion_values=emotion_values,
                    state=state,
                    action=action,
                    next_state=next_state
                )
                logging.debug(f"Reward shaped: {reward} -> {shaped_reward}")
            except Exception as e:
                logging.error(f"Error during reward shaping: {e}", exc_info=True)
        else:
            logging.warning("Reward shaper not available.")


        # 2. Store experience for RL/memory
        experience = {
            "state": state, "action": action, "raw_reward": reward,
            "shaped_reward": shaped_reward, "next_state": next_state,
            "emotion_values": emotion_values, "attention_level": attention_level,
            "done": done, "timestamp": timestamp
        }
        self.store_experience(**experience) # Handles memory storage

        # 3. Update RL model (Dreamer)
        learning_info = {} # Default empty info
        if self.dreamer and hasattr(self.dreamer, 'update') and callable(self.dreamer.update):
             try:
                  # Assuming dreamer.update takes the experience dict and returns learning info
                  # Note: Dreamer often updates based on sequences, not single steps.
                  # This might need adjustment based on DreamerEmotionalWrapper's actual API.
                  # For now, assume it can handle single steps or buffers internally.
                  learning_info = self.dreamer.update(experience)
                  if learning_info is None: learning_info = {} # Ensure it's a dict
                  logging.debug(f"Dreamer updated. Learning info: {learning_info}")
             except Exception as e:
                  logging.error(f"Error calling dreamer.update: {e}", exc_info=True)
                  learning_info = {"error": str(e)}
        else:
             logging.warning("Dreamer component or its 'update' method not available.")
             # Provide mock loss if dreamer isn't working, for metric calculation
             learning_info = {"loss": np.random.rand() * 2.0 + 0.1} # Mock loss


        # 4. Update internal development metrics based on this step
        try:
            self.update_metrics(emotion_values, attention_level, learning_info, done)
            logging.debug(f"Development metrics updated: {self.metrics}")
        except Exception as e:
            logging.error(f"Error updating development metrics: {e}", exc_info=True)

        # 5. Return relevant info
        return {"learning_info": learning_info, "current_metrics": self.get_metrics()}


    def store_experience(self, **kwargs):
        """Stores experience in memory."""
        if self.memory and hasattr(self.memory, 'store') and callable(self.memory.store):
            try:
                # Ensure MemoryCore has an appropriate method like store
                self.memory.store(timestamp=kwargs['timestamp'], data=kwargs)
                logging.debug(f"Experience stored in memory at timestamp {kwargs['timestamp']}")
            except Exception as e:
                logging.error(f"Error storing experience in memory: {e}", exc_info=True)
        else:
            logging.warning("Memory component or its 'store' method not available.")


    def update_metrics(
        self,
        emotion_values: dict[str, float],
        attention_level: float,
        learning_info: dict,
        done: bool # Include 'done' flag for survival tracking
    ):
        """Updates the internal DevelopmentMetrics based on recent experience."""
        self.metrics.emotional_awareness = self._calculate_emotional_awareness(emotion_values)

        # Memory coherence might be expensive, update less often or in evaluate_development
        # For now, check if method exists
        if self.memory and hasattr(self.memory, 'check_coherence') and callable(self.memory.check_coherence):
             # Placeholder: Call might be needed less frequently
             # self.metrics.memory_coherence = self.memory.check_coherence()
             pass # Defer expensive checks
        else:
             # logging.warning("MemoryCore.check_coherence method not found.") # Logged once at init
             self.metrics.memory_coherence = 0.0 # Assume 0 if cannot check

        self.metrics.attention_level = attention_level # Directly use provided value
        self.metrics.behavioral_adaptation = self._calculate_behavioral_adaptation(learning_info)

        # Update survival success based on 'done' flag (simplified)
        # A more robust approach would track episode outcomes over time in memory
        if done:
             # Simple update: Assume non-failure if 'done' is True (needs refinement based on actual reward/state)
             # This is a very basic approximation. Real tracking needs episode outcome analysis.
             current_success_rate = self.metrics.survival_success
             # Simple moving average (e.g., weight 0.05 for new episode outcome)
             # Assume success = 1.0 if done (needs better condition), failure = 0.0
             episode_outcome = 1.0 # Placeholder: Determine actual outcome (e.g., based on final reward)
             self.metrics.survival_success = (current_success_rate * 0.95) + (episode_outcome * 0.05)
             logging.debug(f"Survival success updated based on episode end: {self.metrics.survival_success}")


        self.metrics.last_updated = time.time()


    def _calculate_emotional_awareness(self, emotion_values: dict[str, float]) -> float:
        """Calculates a score based on emotion complexity/differentiation."""
        if not emotion_values: return 0.0
        try:
            values = np.array(list(emotion_values.values()))
            # Use standard deviation as a measure of spread/differentiation
            std_dev = np.std(values)
            # Use mean absolute value as a measure of overall intensity
            mean_abs_intensity = np.mean(np.abs(values))
            # Combine metrics (example weights) - normalize std_dev assuming emotions range roughly -1 to 1
            awareness_score = np.clip((std_dev * 0.6) + (mean_abs_intensity * 0.4), 0.0, 1.0)
            return float(awareness_score)
        except Exception as e:
            logging.error(f"Error calculating emotional awareness: {e}", exc_info=True)
            return 0.0

    def _calculate_behavioral_adaptation(self, learning_info: dict) -> float:
        """Calculates adaptation based on RL learning progress (e.g., loss reduction)."""
        try:
            loss = learning_info.get("loss")
            if loss is not None:
                # Lower loss indicates better adaptation. Use inverse relationship.
                # Add small epsilon to avoid division by zero, clip to [0, 1]
                adaptation_score = np.clip(1.0 / (1.0 + max(0, float(loss))), 0.0, 1.0)
                return float(adaptation_score)
            else:
                # logging.warning("No 'loss' found in learning_info for behavioral adaptation calculation.")
                return 0.0 # Default if no loss info
        except (ValueError, TypeError, Exception) as e:
            logging.error(f"Error calculating behavioral adaptation from loss '{learning_info.get('loss')}': {e}", exc_info=True)
            return 0.0

    def calculate_survival_success(self) -> float:
        """Calculates survival success rate based on historical data (e.g., from memory)."""
        # This method is less useful if updated per-step via 'done'.
        # Kept for potential batch analysis via memory query.
        if self.memory and hasattr(self.memory, 'query_survival_rate') and callable(self.memory.query_survival_rate):
             try:
                  # Example: Query last N episodes
                  survival_rate = self.memory.query_survival_rate(last_n=100) # Needs MemoryCore method
                  logging.debug(f"Queried survival rate from memory: {survival_rate}")
                  # Update the metric if queried successfully
                  self.metrics.survival_success = survival_rate
                  return survival_rate
             except Exception as e:
                  logging.error(f"Error calling memory.query_survival_rate: {e}", exc_info=True)
                  # Return current stored value on error
                  return self.metrics.survival_success
        else:
             # logging.warning("MemoryCore.query_survival_rate method not found.") # Logged at init
             # Return current stored value if cannot query
             return self.metrics.survival_success


    def get_metrics(self) -> dict:
        """Returns the current development metrics as a dictionary."""
        # Ensure metrics are reasonably up-to-date
        # self.metrics.survival_success = self.calculate_survival_success() # Optional: Force query here if needed
        return self.metrics.__dict__


    def evaluate_development(
        self,
        # Removed current_state, activity_log etc. as these should come from the core_module if needed
        attention_metrics: dict[str, float] | None = None
    ) -> dict:
        """
        Performs a more comprehensive evaluation using ConsciousnessMetrics.
        Relies on the core_module instance provided during initialization.

        Args:
            attention_metrics: Optional dictionary with detailed attention data.

        Returns:
            A dictionary containing a snapshot of the agent's developmental status.
        """
        logging.info("Performing comprehensive development evaluation...")
        report = {
            "timestamp": time.time(),
            "development_metrics": self.get_metrics(), # Get latest basic metrics
            "consciousness_metrics": {}, # Placeholder for complex metrics
            "attention_focus": attention_metrics if attention_metrics else {}
        }

        if self.consciousness_metrics is None:
            logging.error("ConsciousnessMetrics not available for evaluation.")
            report["consciousness_metrics"]["error"] = "Metrics module not initialized"
            return report

        # Get necessary state/log from the core module
        current_state = {}
        activity_log = []
        if self.core_module:
             if hasattr(self.core_module, 'get_current_state') and callable(self.core_module.get_current_state):
                  try:
                       current_state = self.core_module.get_current_state()
                  except Exception as e:
                       logging.error(f"Error getting current state from core module: {e}", exc_info=True)
                       report["consciousness_metrics"]["error"] = "Failed to get current state"

             if hasattr(self.core_module, 'get_recent_activity_log') and callable(self.core_module.get_recent_activity_log):
                  try:
                       activity_log = self.core_module.get_recent_activity_log()
                  except Exception as e:
                       logging.error(f"Error getting activity log from core module: {e}", exc_info=True)
                       # Continue evaluation even if log is missing
        else:
             logging.warning("Core module not available, cannot retrieve state/log for complex metrics.")
             report["consciousness_metrics"]["error"] = "Core module not available"


        # Calculate complex metrics if state is available
        if current_state or activity_log: # Proceed if we have at least some info
             try:
                  phi_approx = self.consciousness_metrics.phi_calculator.calculate_phi_approximation(current_state, activity_log)
             except Exception as e:
                  logging.error(f"Error calculating Phi approximation: {e}", exc_info=True)
                  phi_approx = {"error": str(e)}

             try:
                  ignition_detected, ignition_details = self.consciousness_metrics.gwt_tracker.detect_ignition(current_state, activity_log)
             except Exception as e:
                  logging.error(f"Error detecting GWT ignition: {e}", exc_info=True)
                  ignition_detected, ignition_details = False, {"error": str(e)}

             try:
                  pci_approx = self.consciousness_metrics.pci_tester.calculate_pci_approximation(current_state)
             except Exception as e:
                  logging.error(f"Error calculating PCI approximation: {e}", exc_info=True)
                  pci_approx = {"error": str(e)}

             try:
                  self_awareness_scores = self.consciousness_metrics.self_awareness_monitor.evaluate_self_awareness(current_state)
             except Exception as e:
                  logging.error(f"Error evaluating self-awareness: {e}", exc_info=True)
                  self_awareness_scores = {"error": str(e)}

             report["consciousness_metrics"] = {
                 "phi_approx": phi_approx,
                 "gwt_ignition": ignition_detected,
                 "gwt_details": ignition_details,
                 "pci_approx": pci_approx,
                 "self_awareness": self_awareness_scores
             }
        elif "error" not in report["consciousness_metrics"]: # Add error if not already present
             report["consciousness_metrics"]["error"] = "Insufficient state/log data for calculation"


        logging.info(f"Development evaluation complete. Report: {report}")
        return report