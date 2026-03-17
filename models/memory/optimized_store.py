"""
Memory Optimization Module

Implements efficient memory storage and retrieval through:
1. Hierarchical memory indexing
2. Emotional context-based partitioning
3. Attention-weighted storage
4. Dynamic memory consolidation

Based on MANN architecture for cognitive self-representation.
"""
from __future__ import annotations

import time

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class MemoryMetrics:
    """Unified memory system metrics"""
    retrieval_latency: float = 0.0
    index_balance: float = 0.0
    partition_efficiency: float = 0.0
    memory_utilization: float = 0.0
    consolidation_rate: float = 0.0
    cache_hit_rate: float = 0.0


# Alias for backwards compatibility
MemoryOptimizationMetrics = MemoryMetrics


class EmotionalHierarchicalIndex:
    """Hierarchical index partitioned by emotional context."""

    def __init__(self, config: dict):
        self.config = config
        self._partitions: dict[str, list] = {"neutral": [], "positive": [], "negative": []}

    def get_optimal_partition(self, emotional_context: dict[str, float]) -> str:
        valence = emotional_context.get("valence", 0.0)
        if valence > 0.3:
            return "positive"
        elif valence < -0.3:
            return "negative"
        return "neutral"

    def get_relevant_partitions(self, emotional_context: dict[str, float] | None = None) -> list[str]:
        if emotional_context is None:
            return list(self._partitions.keys())
        primary = self.get_optimal_partition(emotional_context)
        return [primary, "neutral"] if primary != "neutral" else ["neutral"]

    def store(self, partition: str, memory_id: str, vector, metadata: dict | None = None):
        if partition not in self._partitions:
            self._partitions[partition] = []
        self._partitions[partition].append({"id": memory_id, "vector": vector, "metadata": metadata})

    def search(self, partition: str, query_vector, k: int = 5) -> list[dict]:
        entries = self._partitions.get(partition, [])
        if not entries:
            return []
        results = []
        q = query_vector.detach().cpu().numpy().flatten() if isinstance(query_vector, torch.Tensor) else np.array(query_vector).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm
        for entry in entries:
            v = entry["vector"]
            v_np = v.detach().cpu().numpy().flatten() if isinstance(v, torch.Tensor) else np.array(v).flatten()
            v_norm = np.linalg.norm(v_np)
            if v_norm == 0:
                continue
            sim = float(np.dot(q, v_np / v_norm))
            results.append({"id": entry["id"], "similarity": sim, "metadata": entry.get("metadata")})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]


class TemporalHierarchicalIndex:
    """Temporal index for time based memory retrieval."""

    def __init__(self, config: dict):
        self._entries: list[dict] = []

    def store(self, memory_id: str, timestamp: float):
        self._entries.append({"id": memory_id, "timestamp": timestamp})


class MemoryConsolidationManager:
    """Manages memory consolidation across partitions."""

    def __init__(self, config: dict):
        self.config = config
        self.threshold = config.get("consolidation_threshold", 0.8)

    def check_consolidation(self, partition: str):
        pass  # Placeholder

    def consolidate_partition(self, partition: str):
        pass  # Placeholder


class OptimizedMemoryStore:
    """
    Implements optimized memory storage with emotional indexing.
    Uses hierarchical structure for fast retrieval.
    """

    def __init__(self, config: dict):
        self.config = config

        # Initialize optimized storage components
        self.emotional_index = EmotionalHierarchicalIndex(config)
        self.temporal_index = TemporalHierarchicalIndex(config)
        self.consolidation_manager = MemoryConsolidationManager(config)

        self.metrics = MemoryOptimizationMetrics()

    def store_optimized(
        self,
        memory_vector: torch.Tensor,
        emotional_context: dict[str, float],
        attention_level: float,
        metadata: dict | None = None
    ) -> str:
        """Store memory with optimized indexing and consolidation."""
        # Apply attention-based gating
        if attention_level < self.config.get('attention_threshold', 0.5):
            return None

        # Get optimal partition based on emotional context
        partition = self.emotional_index.get_optimal_partition(emotional_context)

        # Store in hierarchical indices
        memory_id = self._store_in_indices(
            memory_vector=memory_vector,
            partition=partition,
            emotional_context=emotional_context,
            metadata=metadata
        )

        # Trigger consolidation if needed
        self.consolidation_manager.check_consolidation(partition)

        return memory_id

    def retrieve_optimized(
        self,
        query_vector: torch.Tensor,
        emotional_context: dict[str, float] | None = None,
        k: int = 5
    ) -> list[dict]:
        """Retrieve memories using optimized indices."""
        start_time = time.time()

        # Get relevant emotional partitions
        partitions = self.emotional_index.get_relevant_partitions(emotional_context)

        # Search within partitions
        results = []
        for partition in partitions:
            partition_results = self._search_partition(
                partition=partition,
                query_vector=query_vector,
                k=k
            )
            results.extend(partition_results)

        # Update latency metrics
        self.metrics.retrieval_latency = time.time() - start_time

        # Sort by relevance and return top k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:k]

    def consolidate_memories(self, partition: str):
        """Consolidate memories within partition for optimization."""
        self.consolidation_manager.consolidate_partition(partition)
        self._update_optimization_metrics()

    def _store_in_indices(self, memory_vector, partition: str, emotional_context: dict, metadata: dict | None) -> str:
        memory_id = f"mem_{time.time()}_{partition}"
        self.emotional_index.store(partition, memory_id, memory_vector, metadata)
        self.temporal_index.store(memory_id, time.time())
        return memory_id

    def _search_partition(self, partition: str, query_vector, k: int = 5) -> list[dict]:
        return self.emotional_index.search(partition, query_vector, k)

    def _update_optimization_metrics(self):
        total = sum(len(v) for v in self.emotional_index._partitions.values())
        self.metrics.memory_utilization = min(1.0, total / 10000.0)
