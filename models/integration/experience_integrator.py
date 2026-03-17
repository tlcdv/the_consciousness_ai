"""
Experience Integration Module

This module implements:
1. Integration of multimodal experiences
2. Memory consolidation from experiences
3. Emotional context binding
4. Temporal sequence tracking

Dependencies:
- models/emotion/tgnn/emotional_graph.py for emotion processing
- models/memory/emotional_memory_core.py for memory storage  
- models/evaluation/consciousness_monitor.py for metrics
"""
from __future__ import annotations

# models/integration/experience_integrator.py

import torch
import numpy as np
from dataclasses import dataclass
from models.fusion.emotional_memory_fusion import EmotionalMemoryFusion
from models.generative.generative_emotional_core import GenerativeEmotionalCore
from models.evaluation.emotional_evaluation import EmotionalEvaluator
from models.predictive.attention_mechanism import ConsciousnessAttention
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork as EmotionalGraphNN
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.evaluation.consciousness_monitor import ConsciousnessMonitor
from models.self_model.self_representation_core import SelfRepresentationCore
from models.integration.emotional_development_core import EmotionalDevelopmentCore
import logging

@dataclass
class ExperienceMetrics:
    """Tracks metrics for experience integration"""
    emotional_coherence: float = 0.0
    memory_consolidation: float = 0.0
    attention_focus: float = 0.0
    narrative_consistency: float = 0.0
    consciousness_level: float = 0.0

class ExperienceIntegrator:
    """
    Integrates experiences across modalities to develop consciousness through:
    1. Emotional memory formation during high-attention states
    2. Stress-induced learning through survival scenarios
    3. Narrative construction from emotional memories
    4. Meta-learning for rapid emotional adaptation
    """
    
    def __init__(self, config: dict):
        """Initialize experience integration"""
        self.config = config
        self.emotion_network = EmotionalGraphNN(config)
        self.memory = EmotionalMemoryCore(config)
        self.monitor = ConsciousnessMonitor(config)
        
    def integrate_experience(
        self,
        sensory_input: dict[str, torch.Tensor],
        emotional_context: dict[str, float],
        attention_level: float
    ) -> tuple[dict, dict[str, float]]:
        """Integrate new experience with emotional context"""
        # Process emotional features
        emotional_features = self.emotion_network.process(
            sensory_input,
            emotional_context
        )
        
        # Store in memory if attention is high
        if attention_level > self.config.memory_threshold:
            memory_id = self.memory.store(
                input_data=sensory_input,
                emotional_context=emotional_features,
                attention_level=attention_level
            )
            
        # Update consciousness metrics
        metrics = self.monitor.evaluate_state(
            current_state=sensory_input,
            emotional_context=emotional_features,
            attention_level=attention_level
        )
        
        return {
            'memory_id': memory_id if 'memory_id' in locals() else None,
            'emotional_features': emotional_features,
            'metrics': metrics
        }

    def process_experience(
        self,
        state: dict[str, torch.Tensor],
        emotion_values: dict[str, float],
        stress_level: float,
        context: dict | None = None
    ) -> dict:
        """Process and integrate a new experience"""
        
        # Get attention focus based on stress and emotion
        attention_output, attention_metrics = self.attention.forward(
            input_state=state.get('encoded_state'),
            emotional_context=self.fusion.emotion_network.get_embedding(emotion_values),
            environment_context=context.get('environment_embedding') if context else None
        )
        
        # Fuse multimodal inputs with emotional context
        fusion_output, fusion_info = self.fusion.forward(
            text_input=state.get('text'),
            vision_input=state.get('vision'),
            audio_input=state.get('audio'),
            emotional_context=emotion_values,
            memory_context=self._get_relevant_memories(emotion_values)
        )
        
        # Generate narrative description
        narrative = self.generative.generate_response(
            prompt="Describe the current experience and emotional state",
            emotional_context=emotion_values,
            situation_context={
                'attention': attention_metrics,
                'stress_level': stress_level,
                'fusion_info': fusion_info
            }
        )
        
        # Store integrated experience
        experience = {
            'state': state,
            'emotion': emotion_values,
            'attention': attention_metrics,
            'fusion': fusion_info,
            'narrative': narrative,
            'stress_level': stress_level,
            'context': context
        }
        self.store_experience(experience)
        
        # Update consciousness metrics
        self.update_metrics(
            attention_metrics=attention_metrics,
            fusion_info=fusion_info,
            stress_level=stress_level
        )
        
        return {
            'attention_output': attention_output,
            'fusion_output': fusion_output,
            'narrative': narrative,
            'metrics': self.get_metrics()
        }
        
    def store_experience(self, experience: dict):
        """Store experience in memory"""
        self.experience_history.append(experience)
        self.fusion.memory_core.store_experience(experience)
        
    def update_metrics(
        self,
        attention_metrics: dict[str, float],
        fusion_info: dict,
        stress_level: float
    ):
        """Update consciousness development metrics"""
        # Update emotional coherence
        self.metrics.emotional_coherence = self._calculate_emotional_coherence(
            fusion_info.get('emotional_context', {})
        )
        
        # Update memory consolidation
        self.metrics.memory_consolidation = self._calculate_memory_consolidation()
        
        # Update attention focus
        self.metrics.attention_focus = attention_metrics.get('attention_level', 0.0)
        
        # Update narrative consistency
        self.metrics.narrative_consistency = self._calculate_narrative_consistency()
        
        # Update overall consciousness level
        self.metrics.consciousness_level = self._calculate_consciousness_level(
            stress_level=stress_level
        )
        
    def _get_relevant_memories(
        self,
        emotion_values: dict[str, float],
        k: int = 5
    ) -> list[dict]:
        """Retrieve relevant memories based on emotional similarity"""
        return self.fusion.memory_core.retrieve_similar_memories(
            emotion_query=emotion_values,
            k=k
        )
        
    def _calculate_emotional_coherence(self, emotional_context: dict) -> float:
        """Calculate emotional coherence score"""
        if len(self.experience_history) < 2:
            return 0.0
            
        recent_emotions = [
            exp['emotion'] for exp in self.experience_history[-100:]
        ]
        
        # Calculate stability of emotional transitions
        coherence = np.mean([
            1 - abs(e1['valence'] - e2['valence'])
            for e1, e2 in zip(recent_emotions[:-1], recent_emotions[1:])
        ])
        
        return coherence
        
    def get_metrics(self) -> dict:
        """Get current consciousness metrics"""
        return {
            'emotional_coherence': self.metrics.emotional_coherence,
            'memory_consolidation': self.metrics.memory_consolidation,
            'attention_focus': self.metrics.attention_focus,
            'narrative_consistency': self.metrics.narrative_consistency,
            'consciousness_level': self.metrics.consciousness_level
        }

class SocialLearningPipeline:
    def __init__(self, config: dict):
        self.self_model = SelfRepresentationCore(config)
        self.emotional_core = EmotionalDevelopmentCore(config)
        
    def process_interaction(
        self,
        interaction_data: dict,
        emotion_values: dict[str, float],
        attention_level: float
    ):
        # Extract social feedback
        social_feedback = self._extract_social_signals(interaction_data)
        
        # Update self model
        self.self_model.update_self_model(
            internal_state={
                'emotion': emotion_values,
                'behavior': interaction_data['behavior']
            },
            social_feedback=social_feedback,
            attention_level=attention_level
        )
        
        # Integrate into emotional development
        self.emotional_core.process_experience(
            emotion_values=emotion_values,
            social_context=social_feedback,
            attention=attention_level
        )