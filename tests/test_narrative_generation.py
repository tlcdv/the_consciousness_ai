import unittest
import numpy as np
from models.narrative.narrative_generator import NarrativeGenerator, NarrativeBuffer

class TestNarrativeGenerator(unittest.TestCase):
    """Tests for the Phase 5 template-based Narrative Generator."""
    
    def setUp(self):
        self.generator = NarrativeGenerator({"buffer_size": 5})
        
    def test_pad_quadrant_mapping(self):
        """Test that PAD values correctly map to semantic quadrants."""
        # High Valence, High Arousal
        q1 = self.generator._get_pad_quadrant(0.8, 0.8)
        self.assertEqual(q1, "high_arousal_positive")
        
        # High Valence, Low Arousal
        q2 = self.generator._get_pad_quadrant(0.6, 0.2)
        self.assertEqual(q2, "low_arousal_positive")
        
        # Low Valence, High Arousal
        q3 = self.generator._get_pad_quadrant(-0.7, 0.9)
        self.assertEqual(q3, "high_arousal_negative")
        
        # Low Valence, Low Arousal
        q4 = self.generator._get_pad_quadrant(-0.5, 0.1)
        self.assertEqual(q4, "low_arousal_negative")
        
        # Neutral
        q5 = self.generator._get_pad_quadrant(0.1, 0.3)
        self.assertEqual(q5, "neutral")

    def test_subject_extraction(self):
        """Test robust subject extraction from various broadcast types."""
        # String broadcast
        subj_str = self.generator._extract_subject("bright light")
        self.assertEqual(subj_str, "bright light")
        
        # Dict broadcast
        subj_dict = self.generator._extract_subject({"description": "loud noise", "priority": 0.9})
        self.assertEqual(subj_dict, "loud noise")
        
        # Tensor/Array broadcast
        subj_tensor = self.generator._extract_subject(np.zeros((3, 3)))
        self.assertEqual(subj_tensor, "complex spatial pattern")
        
    def test_generate_from_workspace(self):
        """Test the end-to-end generation and buffer insertion."""
        broadcast = "a red apple"
        emotion = {"valence": 0.9, "arousal": 0.8, "dominance": 0.5}
        action = np.array([0.1, 0.8, -0.2]) # High magnitude
        
        narrative = self.generator.generate_from_workspace(broadcast, emotion, action)
        
        # Check basic string properties
        self.assertIsInstance(narrative, str)
        self.assertIn("apple", narrative)
        
        # Check action agency modifier was added
        self.assertIn("strong, confident action", narrative)
        
        # Check buffer
        recent = self.generator.buffer.get_recent()
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0], narrative)
        
    def test_buffer_coherence(self):
        """Test rolling buffer capacity and coherence metric."""
        buffer = NarrativeBuffer(capacity=3)
        
        # Empty coherence
        self.assertEqual(buffer.get_coherence(), 1.0)
        
        buffer.add("thought 1")
        buffer.add("thought 2")
        self.assertEqual(len(buffer.get_recent()), 2)
        
        buffer.add("thought 3")
        buffer.add("thought 4") # Should evict thought 1
        
        recent = buffer.get_recent()
        self.assertEqual(len(recent), 3)
        self.assertNotIn("thought 1", recent)
        self.assertIn("thought 4", recent)

if __name__ == '__main__':
    unittest.main()
