from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

Observation = Dict[str, Any] # Input from environment
State = Dict[str, Any] # Integrated internal state from ConsciousnessCore
Action = Dict[str, Any] # Action to be executed
LearningInfo = Dict[str, Any] # Information about the learning process (e.g., loss)
Experience = Dict[str, Any] # Single step transition data

class WorldModelInterface(ABC):
    """
    Abstract Base Class for world models and reinforcement learning agents (like Dreamer).
    Defines methods for observing, acting, and potentially learning.
    """

    def __init__(self, config: dict):
        """
        Initializes the world model/RL agent with its specific configuration.
        """
        self.config = config
        super().__init__()

    @abstractmethod
    def observe(self, observation: Observation) -> Any:
        """
        Processes a new observation from the environment to update the internal state
        of the world model or agent's belief state.

        Args:
            observation: The raw observation from the environment.

        Returns:
            An internal representation or summary of the model's updated state (optional).
        """
        pass

    @abstractmethod
    def get_action(self, internal_state: State) -> Action:
        """
        Selects an action based on the agent's current internal state or belief state.

        Args:
            internal_state: The integrated state provided by ConsciousnessCore.

        Returns:
            The action dictionary to be executed.
        """
        pass

    # Optional: Explicit update method if learning is separate from observe/get_action
    # def update(self, experience_or_batch: Any) -> LearningInfo:
    #     """
    #     Updates the model's parameters based on experience.
    #     """
    #     logging.warning("update not implemented in base interface.")
    #     return {}