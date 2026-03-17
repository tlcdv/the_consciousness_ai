# tests/test_emotional_memory_integration.py

import unittest
import torch
import numpy as np
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.fusion.emotional_memory_fusion import EmotionalMemoryFusion
from models.generative.generative_emotional_core import GenerativeEmotionalCore
from models.evaluation.emotional_evaluation import EmotionalEvaluator
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork as EmotionalGraphNN

"""
Integration tests for emotional memory formation and retrieval.

Tests the integration between:
1. Emotional state detection
2. Memory indexing with emotional context
3. Temporal coherence in memory formation
4. Memory retrieval with emotional context

Dependencies:
- models/memory/emotional_memory_core.py for memory operations
- models/emotion/tgnn/emotional_graph.py for emotion processing
- models/core/consciousness_core.py for core system integration
"""

class TestEmotionalMemoryIntegration(unittest.TestCase):
    """Test suite for validating emotional memory formation and integration"""
    
    def setUp(self):
        """Initialize test components"""
        self.config = {
            'memory_config': {
                'capacity': 10000,
                'emotion_embedding_size': 256,
                'fusion_hidden_size': 768
            },
            'generative_config': {
                'model_name': 'llama-3.3',
                'max_length': 1024,
                'temperature': 0.7,
                'emotional_weight': 0.8
            },
            'evaluation_config': {
                'consciousness_thresholds': {
                    'emotional_awareness': 0.7,
                    'memory_coherence': 0.6,
                    'attention_level': 0.8
                }
            }
        }
        
        # Initialize components
        self.memory_core = EmotionalMemoryCore(self.config)
        self.fusion = EmotionalMemoryFusion(self.config)
        self.generative_core = GenerativeEmotionalCore(self.config)
        self.evaluator = EmotionalEvaluator(self.config)
        self.memory = EmotionalMemoryCore(self.config)
        self.emotion = EmotionalGraphNN(self.config)
        
    def test_memory_formation_during_stress(self):
        """Test memory formation during high-stress situations"""
        # Create stressful experience
        experience = {
            'state': torch.randn(32),
            'emotion_values': {
                'valence': 0.3,  # Low valence indicating stress
                'arousal': 0.8,  # High arousal
                'dominance': 0.4  # Low dominance
            },
            'attention_level': 0.9,  # High attention due to stress
            'narrative': "Agent encountered dangerous situation requiring immediate response"
        }
        
        # Store experience
        stored = self.memory_core.store_experience(
            state=experience['state'],
            emotion_values=experience['emotion_values'],
            attention_level=experience['attention_level'],
            context={'narrative': experience['narrative']}
        )
        
        self.assertTrue(stored, "High-stress memory should be stored")
        
        # Retrieve similar experiences
        similar_exp = self.memory_core.retrieve_similar_memories(
            emotion_query={'valence': 0.3, 'arousal': 0.8},
            k=1
        )
        
        self.assertEqual(len(similar_exp), 1)
        self.assertAlmostEqual(
            similar_exp[0].emotion_values['valence'],
            experience['emotion_values']['valence'],
            places=2
        )
        
    def test_emotional_fusion_integration(self):
        """Test multimodal fusion with emotional context"""
        # Create multimodal input
        text_input = torch.randn(1, 32, 768)  # Text embedding
        vision_input = torch.randn(1, 32, 768)  # Vision embedding
        audio_input = torch.randn(1, 32, 768)  # Audio embedding
        
        emotional_context = {
            'valence': 0.7,
            'arousal': 0.6,
            'dominance': 0.8
        }
        
        # Process through fusion
        fusion_output, fusion_info = self.fusion.forward(
            text_input=text_input,
            vision_input=vision_input,
            audio_input=audio_input,
            emotional_context=emotional_context
        )
        
        # Verify fusion output
        self.assertEqual(fusion_output.shape[-1], self.config['memory_config']['fusion_hidden_size'])
        self.assertIn('emotional_context', fusion_info)
        self.assertGreater(fusion_info['fusion_quality'], 0.5)
        
    def test_generative_emotional_response(self):
        """Test generation of emotionally-aware responses"""
        prompt = "How should the agent respond to a human in distress?"
        emotional_context = {
            'valence': 0.3,  # Negative situation
            'arousal': 0.7,  # High emotional intensity
            'dominance': 0.4
        }
        
        # Generate response
        response, metadata = self.generative_core.generate_response(
            prompt=prompt,
            emotional_context=emotional_context
        )
        
        # Verify response properties
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertIn('emotional_context', metadata)
        
        # Evaluate emotional coherence
        evaluation = self.evaluator.evaluate_interaction(
            state=torch.randn(32),
            emotion_values=emotional_context,
            attention_level=0.8,
            narrative=response,
            stress_level=0.7
        )
        
        self.assertGreater(
            evaluation['emotional_awareness'],
            self.config['evaluation_config']['consciousness_thresholds']['emotional_awareness']
        )
        
    def test_memory_consciousness_development(self):
        """Test consciousness development through memory formation"""
        experiences = []
        consciousness_scores = []
        
        # Create sequence of experiences
        for i in range(10):
            experience = {
                'state': torch.randn(32),
                'emotion_values': {
                    'valence': 0.5 + 0.1 * i,  # Improving emotional state
                    'arousal': 0.6,
                    'dominance': 0.5 + 0.05 * i
                },
                'attention_level': 0.7 + 0.02 * i,  # Increasing attention
                'narrative': f"Experience {i} with growing awareness"
            }
            experiences.append(experience)
            
            # Store and evaluate
            self.memory_core.store_experience(
                state=experience['state'],
                emotion_values=experience['emotion_values'],
                attention_level=experience['attention_level'],
                context={'narrative': experience['narrative']}
            )
            
            evaluation = self.evaluator.evaluate_interaction(
                state=experience['state'],
                emotion_values=experience['emotion_values'],
                attention_level=experience['attention_level'],
                narrative=experience['narrative'],
                stress_level=0.5
            )
            
            consciousness_scores.append(evaluation['consciousness_score'])
            
        # Verify consciousness development
        self.assertGreater(
            consciousness_scores[-1],
            consciousness_scores[0],
            "Consciousness score should improve over time"
        )
        
    def test_memory_formation(self):
        """Test emotional memory formation process"""
        test_input = {
            'visual': torch.randn(1, 3, 224, 224),
            'text': 'Test emotional experience',
            'attention': 0.8
        }
        
        # Process emotional context
        emotional_context = self.emotion.process(test_input)
        
        # Store in memory
        memory_id = self.memory.store(
            input_data=test_input,
            emotional_context=emotional_context,
            attention_level=test_input['attention']
        )
        
        # Verify storage
        retrieved = self.memory.retrieve(memory_id)
        self.assertIsNotNone(retrieved)

if __name__ == '__main__':
    unittest.main()