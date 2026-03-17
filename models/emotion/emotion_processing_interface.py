from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

# Context for emotion update (e.g., perception, memory, internal state)
UpdateContext = Dict[str, Any]
# Emotional state representation (e.g., vector of intensities)
EmotionalState = Dict[str, float]

class EmotionProcessingInterface(ABC):
    """
    Abstract Base Class for emotion processing modules.
    Defines the method for updating the internal emotional state.
    """

    def __init__(self, config: dict):
        """
        Initializes the emotion processing module with its specific configuration.
        """
        self.config = config
        super().__init__()

    @abstractmethod
    def update(self, context: UpdateContext) -> EmotionalState:
        """
        Updates the internal emotional state based on the provided context.

        Args:
            context: A dictionary containing relevant information.

        Returns:
            The updated emotional state dictionary.
        """
        pass

    @abstractmethod
    def get_state(self) -> EmotionalState:
        """
        Returns the current internal emotional state.

        Returns:
            The current emotional state dictionary.
        """
        pass

    # Optional method needed for PCI perturbation example in ConsciousnessCore
    # def add_noise(self, magnitude: float):
    #      logging.warning("add_noise not implemented in base interface.")
    #      pass