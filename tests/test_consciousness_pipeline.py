# tests/test_consciousness_pipeline.py

import unittest
import torch
import numpy as np
from dataclasses import dataclass
from models.fusion.emotional_memory_fusion import EmotionalMemoryFusion
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.predictive.attention_mechanism import ConsciousnessAttention
from models.evaluation.consciousness_monitor import ConsciousnessMonitor
from simulations.scenarios.consciousness_scenarios import ConsciousnessScenarioManager
from models.core.consciousness_core import ConsciousnessCore

"""
End-to-end tests for The Consciousness AI development pipeline.

This test suite validates the complete consciousness development cycle including:
1. Attention mechanism activation through stress responses
2. Emotional memory formation and retrieval
3. Multimodal fusion processing
4. Consciousness metrics evaluation

Dependencies:
- models/core/consciousness_core.py for main system
- models/evaluation/consciousness_monitor.py for metrics
- models/memory/emotional_memory_core.py for memory storage
"""

@dataclass
class TestConfig:
    """Test configuration for consciousness pipeline"""
    memory_config = {
        'capacity': 1000,
        'embedding_size': 768,
        'attention_threshold': 0.7
    }
    fusion_config = {
        'text_model': 'llama-3.3',
        'vision_model': 'palm-e',
        'audio_model': 'whisper-v3',
        'fusion_hidden_size': 768
    }
    consciousness_thresholds = {
        'emotional_awareness': 0.7,
        'memory_coherence': 0.6,
        'attention_level': 0.8,
        'narrative_consistency': 0.7
    }
    test_episodes = 5
    min_consciousness_threshold = 0.1

    def get(self, key, default=None):
        return getattr(self, key, default)

class TestConsciousnessPipeline(unittest.TestCase):
    """Test suite for validating consciousness development pipeline"""
    
    def setUp(self):
        """Initialize test components"""
        self.config = TestConfig()
        
        # Core components
        self.consciousness = ConsciousnessCore(self.config)
        self.monitor = ConsciousnessMonitor(self.config)
        self.memory = EmotionalMemoryCore(self.config)
        
        # Test data
        self.test_scenarios = []
        self.consciousness_scores = []

        # Create stub objects for visual processing pipeline.
        class DummyProcessor:
            def __call__(self, x, return_tensors="pt"):
                import torch
                # Return a tensor with the proper shape
                return torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        
        class DummyModel:
            def generate(self, **kwargs):
                return {"dummy_output": 1}
        
        self.core = ConsciousnessCore({})
        self.attention = ConsciousnessAttention({})
        self.fusion = EmotionalMemoryFusion({})

    def _generate_test_scenario(self):
        """Generate a test scenario with state and stress_level attributes."""
        class _Scenario:
            def __init__(self):
                self.state = torch.randn(32)
                self.stress_level = float(np.random.uniform(0.3, 0.9))
        return _Scenario()

    def test_end_to_end_consciousness_development(self):
        """Test complete consciousness development cycle"""
        for episode in range(self.config.test_episodes):
            scenario = self._generate_test_scenario()

            attention_output = self.consciousness.process_attention(
                scenario.state,
                scenario.stress_level
            )

            self.assertGreater(
                attention_output.consciousness_score,
                self.config.min_consciousness_threshold,
                "Consciousness score below minimum threshold"
            )

    def test_stress_induced_attention(self):
        """Test attention activation through stress"""
        state = torch.randn(32)
        emotional_context = torch.randn(128)

        attention_output, metrics = self.attention.forward(
            input_state=state,
            emotional_context=emotional_context
        )

        self.assertIn('attention_level', metrics)
        
    def test_emotional_memory_formation(self):
        """Test memory formation during high-attention states"""
        
        # Create sequence of emotional experiences
        experiences = []
        base_emotion = {
            'valence': 0.3,
            'arousal': 0.8,
            'dominance': 0.4
        }
        
        for i in range(5):
            experience = {
                'state': torch.randn(32),
                'emotion_values': {
                    'valence': min(1.0, base_emotion['valence'] + 0.1 * i),
                    'arousal': base_emotion['arousal'],
                    'dominance': min(1.0, base_emotion['dominance'] + 0.05 * i)
                },
                'attention_level': 0.8 + 0.02 * i
            }
            experiences.append(experience)
            
            # Store experience
            self.memory.store_experience(**experience)
            
        # Retrieve similar experiences
        similar = self.memory.retrieve_similar_memories(
            emotion_query=experiences[-1]['emotion_values'],
            k=3
        )
        
        self.assertEqual(len(similar), 3)
        self.assertGreater(similar[0]['attention_level'], 0.8)
        
    def test_consciousness_monitoring(self):
        """Test consciousness development monitoring"""
        
        initial_state = {
            'encoded_state': torch.randn(32),
            'emotion': {
                'valence': 0.5,
                'arousal': 0.5,
                'dominance': 0.5
            }
        }
        
        # Monitor initial state
        initial_eval = self.monitor.evaluate_development(
            current_state=initial_state,
            emotion_values=initial_state['emotion'],
            attention_metrics={'attention_level': 0.5},
            stress_level=0.5
        )
        
        # Process experiences
        for _ in range(5):
            state = {
                'encoded_state': torch.randn(32),
                'emotion': {
                    'valence': np.random.uniform(0.3, 0.8),
                    'arousal': np.random.uniform(0.6, 0.9),
                    'dominance': np.random.uniform(0.4, 0.7)
                }
            }
            
            eval_result = self.monitor.evaluate_development(
                current_state=state,
                emotion_values=state['emotion'],
                attention_metrics={'attention_level': 0.8},
                stress_level=0.7
            )
            
        # Verify development progress
        self.assertGreater(
            eval_result['consciousness_score'],
            initial_eval['consciousness_score']
        )

    def test_stream_processing(self):
        """Test real-time stream processing capabilities"""
        frame = torch.randn(3, 224, 224)  # Test frame
        result = self.consciousness.process_visual_stream(frame)
        
        self.assertIn('visual_context', result)
        self.assertIn('attention_metrics', result)
        self.assertTrue(0 <= result['attention_metrics']['attention_level'] <= 1)

    def test_visual_stream_processing(self):
        # Create a dummy tensor input simulating a frame.
        frame = torch.randn(3, 224, 224)
        state = self.core.process_visual_stream(frame)
        self.assertIn("visual_context", state)
        self.assertIn("attention_level", state)

if __name__ == '__main__':
    unittest.main()