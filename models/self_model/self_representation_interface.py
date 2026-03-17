from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

# Context for self-model update (e.g., perception, emotion, action feedback)
UpdateContext = Dict[str, Any]
# Representation of the self-model's state
SelfModelState = Dict[str, Any]
# Agent status derived from the self-model
AgentStatus = Dict[str, Any]

class SelfRepresentationInterface(ABC):
    """
    Abstract Base Class for self-representation modules.
    Defines methods for updating and querying the agent's self-model.
    """

    def __init__(self, config: dict):
        """
        Initializes the self-representation module with its specific configuration.
        """
        self.config = config
        super().__init__()

    @abstractmethod
    def update(self, context: UpdateContext) -> SelfModelState:
        """
        Updates the internal self-model based on the provided context.

        Args:
            context: A dictionary containing relevant information.

        Returns:
            The updated state of the self-model.
        """
        pass

    @abstractmethod
    def get_state(self) -> SelfModelState:
        """
        Returns the current internal state of the self-model.

        Returns:
            The current self-model state dictionary.
        """
        pass

    @abstractmethod
    def get_status(self) -> AgentStatus:
        """
        Returns a summary of the agent's status derived from the self-model.

        Returns:
            A dictionary representing the agent's status (e.g., health, position, energy).
        """
        pass