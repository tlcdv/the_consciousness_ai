import unittest
import torch

from models.self_model.embodiment_core import SomatotopicMap, ProprioceptiveProcessor

class TestEmbodimentCore(unittest.TestCase):
    
    def test_somatotopic_homunculus_shapes(self):
        """Test that the 1D part features are correctly projected into 2D Maps"""
        num_parts = 10
        part_dim = 8
        map_size = 8
        B = 2
        
        somato_map = SomatotopicMap(num_body_parts=num_parts, feature_per_part=part_dim, map_size=map_size)
        
        # Fake part feature tensor [B, Parts, Dims]
        part_features = torch.randn(B, num_parts, part_dim)
        
        body_schema = somato_map(part_features)
        
        # Expecting shape [B, part_dim * 2, map_size, map_size] due to conv block
        self.assertEqual(body_schema.shape, (B, part_dim * 2, map_size, map_size))
        
    def test_proprioceptive_processor_pain_priority(self):
        """Test that hard collisions generate priority bids (>0.8) for consciousness"""
        raw_state_dim = 40 # E.g. Unity raw float array for joints
        num_parts = 10
        part_dim = 8
        B = 1
        
        processor = ProprioceptiveProcessor(raw_state_dim, num_parts, part_dim)
        
        raw_state = torch.randn(B, raw_state_dim)
        
        # Normal condition (No collisions)
        normal_schema, normal_bid = processor(raw_state)
        self.assertIsInstance(normal_bid, float)
        self.assertTrue(0.0 <= normal_bid <= 1.0)
        
        # Pain condition (Collisions on multiple parts)
        collision_flags = torch.zeros(B, num_parts)
        collision_flags[0, 2] = 1.0 # E.g. Left arm hit
        collision_flags[0, 4] = 1.0 # E.g. Left leg hit
        
        pain_schema, pain_bid = processor(raw_state, collision_flags)
        
        self.assertIsInstance(pain_bid, float)
        self.assertTrue(0.85 <= pain_bid <= 1.0, f"Bid should jump to high priority on collision. Got: {pain_bid}")
        
if __name__ == '__main__':
    unittest.main()
