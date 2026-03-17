from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any

# Context for retrieval (e.g., current perception, emotion)
QueryContext = Dict[str, Any]
# Data to be stored (e.g., full experience dict)
MemoryData = Dict[str, Any]
# Retrieved memory entry
RetrievedMemory = Dict[str, Any]

class MemoryInterface(ABC):
    """
    Abstract Base Class for memory systems.
    Defines methods for storing and retrieving experiences/information.
    """

    def __init__(self, config: dict):
        """
        Initializes the memory system with its specific configuration.
        """
        self.config = config
        super().__init__()

    @abstractmethod
    def store(self, timestamp: float, data: MemoryData):
        """
        Stores data associated with a specific timestamp.

        Args:
            timestamp: The time the data occurred.
            data: The dictionary containing the information to store.
        """
        pass

    @abstractmethod
    def retrieve(self, query_context: QueryContext, top_k: int = 5) -> list[RetrievedMemory]:
        """
        Retrieves relevant memories based on the provided context.

        Args:
            query_context: A dictionary containing cues for retrieval.
            top_k: The maximum number of memories to retrieve.

        Returns:
            A list of retrieved memory dictionaries, potentially ordered by relevance.
        """
        pass

    # Optional common methods used elsewhere (add if needed by concrete implementations)
    # def check_coherence(self) -> float:
    #     logging.warning("check_coherence not implemented in base interface.")
    #     return 0.0
    #
    # def query_survival_rate(self, last_n: int) -> float:
    #     logging.warning("query_survival_rate not implemented in base interface.")
    #     return 0.0