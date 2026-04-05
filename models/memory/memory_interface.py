"""Minimal memory interface: type aliases and ABC for memory subsystems."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

QueryContext = Dict[str, Any]
MemoryData = Dict[str, Any]
RetrievedMemory = Dict[str, Any]


class MemoryInterface(ABC):
    """Abstract base class for memory systems."""

    def __init__(self, config: dict):
        self.config = config
        super().__init__()

    @abstractmethod
    def store(self, timestamp: float, data: MemoryData):
        ...

    @abstractmethod
    def retrieve(self, query_context: QueryContext, top_k: int = 5) -> list[RetrievedMemory]:
        ...
