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
    """
    Manages memory consolidation: pruning low-relevance entries, merging
    similar ones (cosine > merge_threshold), and providing replay batches.

    Each memory entry gets a `relevance` score that increments on retrieval
    and decays by `decay_rate` per consolidation cycle.
    """

    def __init__(self, config: dict):
        self.config = config
        self.consolidation_threshold = config.get("consolidation_threshold", 100)
        self.merge_threshold = config.get("merge_threshold", 0.9)
        self.prune_threshold = config.get("prune_threshold", 0.1)
        self.decay_rate = config.get("relevance_decay", 0.99)
        # Legacy merge: only preserve id/vector/relevance/metadata on merge,
        # dropping state/action/emotion fields. Reproduces the bug fixed in
        # commit deef672 so the impact of that fix can be measured.
        self.use_legacy_merge = config.get("use_legacy_merge", False)
        self._consolidation_count = 0

    def check_consolidation(self, partition: str, entries: list[dict] | None = None) -> bool:
        """Check if a partition needs consolidation (entry count exceeds threshold)."""
        if entries is None:
            return False
        return len(entries) >= self.consolidation_threshold

    def consolidate(self, entries: list[dict]) -> list[dict]:
        """
        Run a full consolidation cycle on a list of memory entries.

        Each entry must have: "vector", "id", and optionally "relevance", "metadata".

        Steps:
          1. Decay all relevance scores by decay_rate
          2. Merge entries with cosine similarity > merge_threshold
          3. Prune entries with relevance < prune_threshold

        Returns the consolidated list.
        """
        if not entries:
            return entries

        # 1. Decay relevance
        for entry in entries:
            entry.setdefault("relevance", 1.0)
            entry["relevance"] *= self.decay_rate

        # 2. Merge similar entries
        if self.use_legacy_merge:
            entries = self._merge_similar_legacy(entries)
        else:
            entries = self._merge_similar(entries)

        # 3. Prune low-relevance
        entries = [e for e in entries if e.get("relevance", 0.0) >= self.prune_threshold]

        self._consolidation_count += 1
        return entries

    def _merge_similar(self, entries: list[dict]) -> list[dict]:
        """Merge entries whose vectors have cosine similarity > merge_threshold."""
        if len(entries) < 2:
            return entries

        # Extract vectors
        vectors = []
        for e in entries:
            v = e["vector"]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy().flatten()
            else:
                v = np.array(v).flatten()
            vectors.append(v)

        norms = [np.linalg.norm(v) for v in vectors]
        normed = [v / max(n, 1e-8) for v, n in zip(vectors, norms)]

        merged_mask = [False] * len(entries)
        result = []

        for i in range(len(entries)):
            if merged_mask[i]:
                continue
            group_indices = [i]
            for j in range(i + 1, len(entries)):
                if merged_mask[j]:
                    continue
                sim = float(np.dot(normed[i], normed[j]))
                if sim > self.merge_threshold:
                    group_indices.append(j)
                    merged_mask[j] = True

            if len(group_indices) == 1:
                result.append(entries[i])
            else:
                # Merge: average vectors, sum relevance, preserve all fields from best entry
                avg_vec = np.mean([vectors[idx] for idx in group_indices], axis=0)
                total_relevance = sum(entries[idx].get("relevance", 1.0) for idx in group_indices)
                best_idx = max(group_indices, key=lambda idx: entries[idx].get("relevance", 1.0))

                # Start with all fields from the highest-relevance entry
                merged_entry = dict(entries[best_idx])
                merged_entry["vector"] = avg_vec
                merged_entry["relevance"] = total_relevance
                merged_entry["merged_count"] = len(group_indices)
                result.append(merged_entry)

        return result

    def _merge_similar_legacy(self, entries: list[dict]) -> list[dict]:
        """Pre-deef672 _merge_similar: only preserves id/vector/relevance/metadata.

        Other fields (state, action, reward, emotion_values, ...) are dropped
        from the merged entry, which broke action_core.replay_update because
        it expected those fields. Used only under the ablate-consolidation-fix
        flag to measure the impact of the deef672 fix.
        """
        if len(entries) < 2:
            return entries

        vectors = []
        for e in entries:
            v = e["vector"]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy().flatten()
            else:
                v = np.array(v).flatten()
            vectors.append(v)

        norms = [np.linalg.norm(v) for v in vectors]
        normed = [v / max(n, 1e-8) for v, n in zip(vectors, norms)]

        merged_mask = [False] * len(entries)
        result = []

        for i in range(len(entries)):
            if merged_mask[i]:
                continue
            group_indices = [i]
            for j in range(i + 1, len(entries)):
                if merged_mask[j]:
                    continue
                sim = float(np.dot(normed[i], normed[j]))
                if sim > self.merge_threshold:
                    group_indices.append(j)
                    merged_mask[j] = True

            if len(group_indices) == 1:
                result.append(entries[i])
            else:
                avg_vec = np.mean([vectors[idx] for idx in group_indices], axis=0)
                total_relevance = sum(entries[idx].get("relevance", 1.0) for idx in group_indices)
                best_idx = max(group_indices, key=lambda idx: entries[idx].get("relevance", 1.0))
                merged_entry = {
                    "id": entries[best_idx].get("id"),
                    "vector": avg_vec,
                    "relevance": total_relevance,
                    "metadata": entries[best_idx].get("metadata"),
                    "merged_count": len(group_indices),
                }
                result.append(merged_entry)

        return result

    def get_replay_batch(self, entries: list[dict], k: int = 16) -> list[dict]:
        """Return top-K entries by relevance for experience replay."""
        if not entries:
            return []
        sorted_entries = sorted(entries, key=lambda e: e.get("relevance", 0.0), reverse=True)
        return sorted_entries[:k]

    def increment_relevance(self, entry: dict, amount: float = 0.1):
        """Increment relevance on retrieval."""
        entry["relevance"] = entry.get("relevance", 1.0) + amount

    @property
    def consolidation_count(self) -> int:
        return self._consolidation_count


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
    ) -> str | None:
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
        entries = self.emotional_index._partitions.get(partition, [])
        if self.consolidation_manager.check_consolidation(partition, entries):
            self.consolidate_memories(partition)

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

    def consolidate_memories(self, partition: str | None = None):
        """
        Consolidate memories within one or all partitions.

        Prunes low-relevance entries, merges similar ones, and updates metrics.
        """
        partitions = [partition] if partition else list(self.emotional_index._partitions.keys())
        for p in partitions:
            entries = self.emotional_index._partitions.get(p, [])
            if not entries:
                continue
            # Ensure all entries have relevance
            for e in entries:
                e.setdefault("relevance", 1.0)
            consolidated = self.consolidation_manager.consolidate(entries)
            self.emotional_index._partitions[p] = consolidated
        self._update_optimization_metrics()

    def get_replay_batch(self, k: int = 16) -> list[dict]:
        """Return top-K entries by relevance across all partitions."""
        all_entries = []
        for entries in self.emotional_index._partitions.values():
            all_entries.extend(entries)
        return self.consolidation_manager.get_replay_batch(all_entries, k)

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
