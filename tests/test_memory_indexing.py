import unittest
import torch
import numpy as np
from models.memory.emotional_indexing import EmotionalMemoryIndex
from models.memory.emotional_memory_core import EmotionalMemoryCore
# EmotionalGraphNetwork not used directly in these tests

class TestMemoryIndexing(unittest.TestCase):
    """Test suite for emotional memory indexing and retrieval system."""

    def setUp(self):
        # Example dictionary-based config that EmotionalMemoryIndex might expect.
        self.config = {
            'vector_dimension': 768,
            'index_name': 'test_emotional_memories',
            'embedding_batch_size': 32,
            'consciousness_thresholds': {
                'emotional_awareness': 0.7,
                'memory_coherence': 0.6,
                'attention_level': 0.8
            },
            # Additional placeholders if needed.
        }

        # Initialize core components.
        # Adjust as necessary if EmotionalMemoryIndex uses a MemoryConfig dataclass, etc.
        self.memory_index = EmotionalMemoryIndex(self.config)
        self.memory_core = EmotionalMemoryCore(self.config)
        # EmotionalGraphNetwork requires heavy config; not needed for index tests

    def test_memory_storage_and_retrieval(self):
        """Test basic memory storage and retrieval functionality."""
        test_memory = {
            'state': torch.randn(32),
            'emotion_values': {
                'valence': 0.8,  # Positive emotion
                'arousal': 0.7,
                'dominance': 0.6
            },
            'attention_level': 0.9,
            'narrative': "Successfully completed challenging task with positive outcome"
        }

        # Store memory in index.
        memory_id = self.memory_index.store_memory(
            state=test_memory['state'],
            emotion_values=test_memory['emotion_values'],
            attention_level=test_memory['attention_level'],
            narrative=test_memory['narrative']
        )

        self.assertIsNotNone(memory_id, "Memory ID should not be None after storing.")

        # Retrieve similar memories.
        retrieved_memories = self.memory_index.retrieve_similar_memories(
            emotion_query=test_memory['emotion_values'],
            k=1
        )

        # Verify retrieval is non-empty.
        self.assertGreater(len(retrieved_memories), 0)
        # For a robust test, check that the similarity is above some threshold (dummy logic).
        self.assertGreater(retrieved_memories[0].get('similarity', 0.0), 0.8)

    def test_emotional_coherence(self):
        """Test emotional coherence in memory sequences."""
        # Create a sequence of memories with gradually improving valence.
        memories = []
        base_valence = 0.3

        for i in range(5):
            memory = {
                'state': torch.randn(32),
                'emotion_values': {
                    'valence': min(1.0, base_valence + 0.1 * i),
                    'arousal': 0.7,
                    'dominance': 0.5 + 0.05 * i
                },
                'attention_level': 0.8 + 0.02 * i,
                'narrative': f"Memory {i} in emotional sequence"
            }
            memories.append(memory)

        # Store each memory.
        for mem in memories:
            self.memory_index.store_memory(
                state=mem['state'],
                emotion_values=mem['emotion_values'],
                attention_level=mem['attention_level'],
                narrative=mem['narrative']
            )

        # Retrieve them as a temporal sequence (placeholder method).
        # If your code tracks timestamps, pass actual start/end times.
        temporal_sequence = self.memory_index.get_temporal_sequence(
            start_time=0.0, 
            end_time=float('inf')
        )
        self.assertEqual(len(temporal_sequence), 5)

        # Check that valence is non-decreasing.
        valences = [item['emotion_values']['valence'] for item in temporal_sequence]
        self.assertTrue(all(x <= y for x, y in zip(valences, valences[1:])),
                        "Valence should be non-decreasing in the stored sequence.")

    def test_consciousness_relevant_retrieval(self):
        """Test retrieval based on consciousness relevance."""
        high_consciousness_memory = {
            'state': torch.randn(32),
            'emotion_values': {'valence': 0.8, 'arousal': 0.9, 'dominance': 0.7},
            'attention_level': 0.95,
            'narrative': "Highly conscious experience with deep emotional impact"
        }
        low_consciousness_memory = {
            'state': torch.randn(32),
            'emotion_values': {'valence': 0.4, 'arousal': 0.3, 'dominance': 0.4},
            'attention_level': 0.5,
            'narrative': "Low consciousness routine experience"
        }

        # Store both.
        self.memory_index.store_memory(
            state=high_consciousness_memory['state'],
            emotion_values=high_consciousness_memory['emotion_values'],
            attention_level=high_consciousness_memory['attention_level'],
            narrative=high_consciousness_memory['narrative']
        )
        self.memory_index.store_memory(
            state=low_consciousness_memory['state'],
            emotion_values=low_consciousness_memory['emotion_values'],
            attention_level=low_consciousness_memory['attention_level'],
            narrative=low_consciousness_memory['narrative']
        )

        # Retrieve with a consciousness threshold (dummy logic).
        retrieved = self.memory_index.retrieve_similar_memories(
            emotion_query={'valence': 0.6, 'arousal': 0.6, 'dominance': 0.6},
            min_consciousness_score=0.8
        )

        # Verify only high consciousness memory is retrieved.
        self.assertEqual(len(retrieved), 1)
        self.assertGreater(retrieved[0].get('consciousness_score', 0.0), 0.8)

    def test_memory_statistics(self):
        """Test memory statistics and metrics tracking."""
        # Store a series of memories.
        for i in range(10):
            mem = {
                'state': torch.randn(32),
                'emotion_values': {
                    'valence': np.random.random(),
                    'arousal': np.random.random(),
                    'dominance': np.random.random()
                },
                'attention_level': 0.7 + 0.02 * i,
                'narrative': f"Test memory {i}"
            }
            self.memory_index.store_memory(
                state=mem['state'],
                emotion_values=mem['emotion_values'],
                attention_level=mem['attention_level'],
                narrative=mem['narrative']
            )

        # Check memory stats (dummy property in EmotionalMemoryIndex).
        stats = self.memory_index.memory_stats
        self.assertIn('emotional_coherence', stats)
        self.assertIn('temporal_consistency', stats)
        self.assertIn('consciousness_relevance', stats)

        # Example check: stats should be between 0 and 1 if they represent normalized metrics.
        for key, val in stats.items():
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0)

if __name__ == '__main__':
    unittest.main()
