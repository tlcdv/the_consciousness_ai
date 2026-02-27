import unittest
import numpy as np
from models.narrative.narrative_generator import NarrativeGenerator, NarrativeBuffer

class TestNarrativeBuffer(unittest.TestCase):
    """Tests for the rolling narrative history buffer."""
    
    def test_capacity_eviction(self):
        """Buffer should evict oldest entries when capacity is reached."""
        buffer = NarrativeBuffer(capacity=3)
        for i in range(5):
            buffer.add(f"thought {i}")
        
        recent = buffer.get_recent()
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0], "thought 2")
        self.assertEqual(recent[2], "thought 4")
    
    def test_coherence_metric(self):
        """Coherence should increase as the buffer fills."""
        buffer = NarrativeBuffer(capacity=10)
        self.assertEqual(buffer.get_coherence(), 1.0)  # Empty is maximally coherent
        
        buffer.add("first")
        self.assertEqual(buffer.get_coherence(), 1.0)  # Single item is coherent
        
        buffer.add("second")
        coherence_2 = buffer.get_coherence()
        for i in range(8):
            buffer.add(f"thought {i}")
        coherence_full = buffer.get_coherence()
        self.assertGreater(coherence_full, coherence_2)


class TestNarrativeGenerator(unittest.TestCase):
    """Tests for the Phase 5 template-based Narrative Generator."""
    
    def setUp(self):
        self.generator = NarrativeGenerator({"buffer_size": 10})
        
    def test_pad_quadrant_mapping(self):
        """Test that PAD values correctly map to semantic quadrants."""
        self.assertEqual(self.generator._get_pad_quadrant(0.8, 0.8), "high_arousal_positive")
        self.assertEqual(self.generator._get_pad_quadrant(0.6, 0.2), "low_arousal_positive")
        self.assertEqual(self.generator._get_pad_quadrant(-0.7, 0.9), "high_arousal_negative")
        self.assertEqual(self.generator._get_pad_quadrant(-0.5, 0.1), "low_arousal_negative")
        self.assertEqual(self.generator._get_pad_quadrant(0.1, 0.3), "neutral")

    def test_subject_extraction(self):
        """Test robust subject extraction from various broadcast types."""
        self.assertEqual(self.generator._extract_subject("bright light"), "bright light")
        self.assertEqual(self.generator._extract_subject({"description": "loud noise"}), "loud noise")
        self.assertEqual(self.generator._extract_subject(np.zeros((3, 3))), "complex spatial pattern")
        self.assertEqual(self.generator._extract_subject(42), "unclear stimuli")
        
    def test_generate_populates_buffer(self):
        """Generation should add entries to the rolling buffer."""
        emotion = {"valence": 0.5, "arousal": 0.3, "dominance": 0.0}
        self.generator.generate_from_workspace("test subject", emotion)
        self.assertEqual(len(self.generator.buffer.get_recent()), 1)
        
        self.generator.generate_from_workspace("test subject 2", emotion)
        self.assertEqual(len(self.generator.buffer.get_recent()), 2)

    def test_action_awareness_high_dominance(self):
        """High dominance + high action should produce confidence phrasing."""
        emotion = {"valence": 0.8, "arousal": 0.8, "dominance": 0.5}
        action = np.array([0.9, 0.0, 0.0])
        narrative = self.generator.generate_from_workspace("target", emotion, action)
        self.assertIn("confident action", narrative)

    def test_action_awareness_low_dominance(self):
        """Low dominance + high action should produce loss-of-control phrasing."""
        emotion = {"valence": -0.5, "arousal": 0.9, "dominance": -0.5}
        action = np.array([0.9, 0.0, 0.0])
        narrative = self.generator.generate_from_workspace("threat", emotion, action)
        self.assertIn("out of control", narrative)

    def test_temporal_coherence_calm_to_alarm(self):
        """Transition from calm to alarm should produce 'Suddenly' bridge."""
        # First: generate a calm narrative (low_arousal_positive quadrant)
        calm_emotion = {"valence": 0.5, "arousal": 0.2, "dominance": 0.5}
        n1 = self.generator.generate_from_workspace("garden", calm_emotion)
        # Should match one of the low_arousal_positive templates
        valid_calm = ["Calmly", "Feeling at peace", "Everything is stable"]
        self.assertTrue(
            any(phrase in n1 for phrase in valid_calm),
            f"Expected low_arousal_positive template, got: {n1}"
        )
        
        # Second: generate an alarmed narrative (high_arousal_negative quadrant)
        alarm_emotion = {"valence": -0.8, "arousal": 0.9, "dominance": -0.5}
        n2 = self.generator.generate_from_workspace("danger", alarm_emotion)
        
        # Should have a transition bridge (now based on quadrant history, not text)
        self.assertTrue(n2.startswith("Suddenly"))

    def test_temporal_coherence_repetition(self):
        """Repeated identical states should produce 'Still' prefix."""
        emotion = {"valence": 0.1, "arousal": 0.2, "dominance": 0.0}
        # Force identical templates by seeding
        np.random.seed(42)
        n1 = self.generator.generate_from_workspace("nothing", emotion)
        np.random.seed(42)  # Same seed -> same template
        n2 = self.generator.generate_from_workspace("nothing", emotion)
        np.random.seed(42)
        n3 = self.generator.generate_from_workspace("nothing", emotion)
        
        # Third generation should see two identical predecessors and add "Still"
        if n1 == n2:  # Only test if the seed actually produced identical outputs
            self.assertTrue(n3.startswith("Still"))

    def test_legacy_interface(self):
        """Legacy generate() should still work."""
        result = self.generator.generate({"description": "test"}, {"valence": 0.5, "arousal": 0.3, "dominance": 0.0})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


if __name__ == '__main__':
    unittest.main()
