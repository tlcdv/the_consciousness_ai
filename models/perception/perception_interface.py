from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

Observation = Dict[str, Any] # Raw input (e.g., image, audio tensors, sensor data)
PerceptionSummary = Dict[str, Any] # Processed output (e.g., object list, text transcript, scene description)

class PerceptionInterface(ABC):
    """
    Abstract Base Class for all perception modules.
    Defines the standard method for processing raw observations.
    """

    def __init__(self, config: dict):
        """
        Initializes the perception module with its specific configuration.
        """
        self.config = config
        super().__init__()

    @abstractmethod
    def process(self, observation: Observation) -> PerceptionSummary:
        """
        Processes raw sensory observation data into a structured summary.

        Args:
            observation: A dictionary containing raw sensory data (e.g., {'image': tensor, 'audio': tensor}).

        Returns:
            A dictionary summarizing the perceived information (e.g., {'objects': [...], 'transcript': "..."}).
        """
        pass

    # Optional: Add methods for specific modalities if needed
    # @abstractmethod
    # def process_visual(self, image_data: Any) -> dict:
    #     pass
    #
    # @abstractmethod
    # def process_audio(self, audio_data: Any) -> dict:
    #     pass