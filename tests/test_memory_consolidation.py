"""Tests for MemoryConsolidationManager and OptimizedMemoryStore consolidation."""
from __future__ import annotations

import unittest
import numpy as np
import torch

from models.memory.optimized_store import (
    MemoryConsolidationManager,
    OptimizedMemoryStore,
)


class TestConsolidationManager(unittest.TestCase):

    def setUp(self):
        self.manager = MemoryConsolidationManager({
            "merge_threshold": 0.9,
            "prune_threshold": 0.1,
            "relevance_decay": 0.99,
        })

    def _make_entry(self, vec, relevance=1.0, entry_id="e"):
        return {"id": entry_id, "vector": np.array(vec, dtype=np.float32), "relevance": relevance}

    def test_empty_consolidation(self):
        """Consolidating empty list returns empty list."""
        result = self.manager.consolidate([])
        self.assertEqual(result, [])

    def test_relevance_decay(self):
        """Consolidation should decay relevance by decay_rate."""
        entries = [self._make_entry([1.0, 0.0], relevance=1.0)]
        result = self.manager.consolidate(entries)
        self.assertAlmostEqual(result[0]["relevance"], 0.99, places=5)

    def test_prune_low_relevance(self):
        """Entries below prune_threshold should be removed."""
        entries = [
            self._make_entry([1.0, 0.0], relevance=0.5, entry_id="keep"),
            self._make_entry([0.0, 1.0], relevance=0.05, entry_id="prune"),
        ]
        result = self.manager.consolidate(entries)
        ids = [e["id"] for e in result]
        self.assertIn("keep", ids)
        self.assertNotIn("prune", ids)

    def test_merge_similar_vectors(self):
        """Vectors with cosine > merge_threshold should be merged."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.99, 0.01, 0.0], dtype=np.float32)  # cosine ~ 0.9999
        entries = [
            {"id": "a", "vector": v1, "relevance": 0.5},
            {"id": "b", "vector": v2, "relevance": 0.8},
        ]
        result = self.manager.consolidate(entries)
        self.assertEqual(len(result), 1)
        # Relevance should be summed (after decay)
        self.assertGreater(result[0]["relevance"], 0.5)

    def test_no_merge_dissimilar_vectors(self):
        """Orthogonal vectors should not be merged."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        entries = [
            {"id": "a", "vector": v1, "relevance": 1.0},
            {"id": "b", "vector": v2, "relevance": 1.0},
        ]
        result = self.manager.consolidate(entries)
        self.assertEqual(len(result), 2)

    def test_consolidation_count(self):
        """consolidation_count should increment per consolidation."""
        self.assertEqual(self.manager.consolidation_count, 0)
        self.manager.consolidate([self._make_entry([1, 0])])
        self.assertEqual(self.manager.consolidation_count, 1)
        self.manager.consolidate([self._make_entry([1, 0])])
        self.assertEqual(self.manager.consolidation_count, 2)

    def test_increment_relevance(self):
        """increment_relevance should increase the relevance field."""
        entry = {"id": "e", "relevance": 0.5}
        self.manager.increment_relevance(entry, amount=0.2)
        self.assertAlmostEqual(entry["relevance"], 0.7, places=5)

    def test_increment_relevance_default(self):
        """Entry without relevance key should get default + amount."""
        entry = {"id": "e"}
        self.manager.increment_relevance(entry)
        self.assertAlmostEqual(entry["relevance"], 1.1, places=5)


class TestReplayBatch(unittest.TestCase):

    def setUp(self):
        self.manager = MemoryConsolidationManager({})

    def test_replay_batch_returns_top_k(self):
        """get_replay_batch should return K entries sorted by relevance."""
        entries = [
            {"id": "low", "vector": [0], "relevance": 0.1},
            {"id": "mid", "vector": [0], "relevance": 0.5},
            {"id": "high", "vector": [0], "relevance": 0.9},
        ]
        batch = self.manager.get_replay_batch(entries, k=2)
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0]["id"], "high")
        self.assertEqual(batch[1]["id"], "mid")

    def test_replay_batch_empty(self):
        self.assertEqual(self.manager.get_replay_batch([], k=5), [])


class TestOptimizedStoreConsolidation(unittest.TestCase):

    def setUp(self):
        self.store = OptimizedMemoryStore({
            "attention_threshold": 0.0,  # Accept all
            "merge_threshold": 0.95,
            "prune_threshold": 0.05,
            "relevance_decay": 0.99,
        })

    def test_store_and_consolidate(self):
        """Store several memories, consolidate, verify count reduced or stable."""
        for i in range(10):
            vec = torch.randn(16)
            self.store.store_optimized(
                memory_vector=vec,
                emotional_context={"valence": 0.5},
                attention_level=0.8,
            )
        total_before = sum(len(v) for v in self.store.emotional_index._partitions.values())
        self.assertEqual(total_before, 10)

        self.store.consolidate_memories()
        total_after = sum(len(v) for v in self.store.emotional_index._partitions.values())
        # After one consolidation with random vectors, no merging expected but decay happens
        self.assertGreater(total_after, 0)
        self.assertLessEqual(total_after, total_before)

    def test_store_below_attention_threshold(self):
        """store_optimized should return None when attention is below threshold."""
        store = OptimizedMemoryStore({"attention_threshold": 0.5})
        result = store.store_optimized(
            memory_vector=torch.randn(16),
            emotional_context={"valence": 0.5},
            attention_level=0.3,
        )
        self.assertIsNone(result)
        total = sum(len(v) for v in store.emotional_index._partitions.values())
        self.assertEqual(total, 0)

    def test_replay_batch_from_store(self):
        """get_replay_batch should work on the store."""
        for i in range(5):
            self.store.store_optimized(
                memory_vector=torch.randn(16),
                emotional_context={"valence": 0.0},
                attention_level=0.8,
            )
        batch = self.store.get_replay_batch(k=3)
        self.assertLessEqual(len(batch), 3)

    def test_consolidate_all_partitions(self):
        """consolidate_memories(None) should process all partitions."""
        self.store.store_optimized(torch.randn(16), {"valence": 0.5}, 0.8)
        self.store.store_optimized(torch.randn(16), {"valence": -0.5}, 0.8)
        self.store.store_optimized(torch.randn(16), {"valence": 0.0}, 0.8)
        self.store.consolidate_memories()  # No partition arg -> all
        # Should not crash, metrics updated
        self.assertIsNotNone(self.store.metrics.memory_utilization)


if __name__ == "__main__":
    unittest.main()
