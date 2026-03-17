from __future__ import annotations

# models/memory/emotional_sync.py

import torch
import numpy as np
from dataclasses import dataclass
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.fusion.emotional_memory_fusion import EmotionalMemoryFusion
from models.predictive.attention_mechanism import ConsciousnessAttention
from models.evaluation.emotional_evaluation import EmotionalEvaluator

@dataclass
class SyncConfig:
    """Configuration for emotional memory synchronization"""
    sync_frequency: int = 10
    batch_size: int = 32
    memory_threshold: float = 0.7
    attention_threshold: float = 0.8
    consolidation_rate: float = 0.1

class EmotionalMemorySync:
    """
    Synchronizes emotional memories across components and manages consciousness development
    
    Key Features:
    1. Cross-component memory synchronization
    2. Attention-guided memory consolidation
    3. Emotional coherence verification
    4. Consciousness development tracking
    """
    
    def __init__(self, config: SyncConfig):
        self.config = config
        
        # Core components
        self.memory_core = EmotionalMemoryCore(config)
        self.fusion = EmotionalMemoryFusion(config)
        self.attention = ConsciousnessAttention(config)
        self.evaluator = EmotionalEvaluator(config)
        
        # Sync tracking
        self.sync_counter = 0
        self.consolidated_memories = []
        
    def sync_memories(
        self,
        current_state: dict[str, torch.Tensor],
        emotion_values: dict[str, float],
        attention_metrics: dict[str, float]
    ) -> dict:
        """Synchronize emotional memories across components"""
        
        # Check if sync is needed
        self.sync_counter += 1
        if self.sync_counter % self.config.sync_frequency != 0:
            return {}
            
        # Get attention-weighted memories
        attention_memories = self._get_attention_memories(
            attention_metrics['attention_level']
        )
        
        # Get emotionally coherent memories
        emotional_memories = self._get_emotional_memories(
            emotion_values
        )
        
        # Consolidate memories
        consolidated = self._consolidate_memories(
            attention_memories=attention_memories,
            emotional_memories=emotional_memories,
            current_state=current_state
        )
        
        # Update consciousness metrics
        consciousness_metrics = self.evaluator.evaluate_interaction(
            state=current_state,
            emotion_values=emotion_values,
            attention_level=attention_metrics['attention_level'],
            narrative=consolidated.get('narrative', ''),
            stress_level=attention_metrics.get('stress_level', 0.0)
        )
        
        # Store consolidated memories
        self._store_consolidated_memories(consolidated)
        
        return {
            'consolidated_memories': consolidated,
            'consciousness_metrics': consciousness_metrics,
            'sync_status': 'success'
        }
        
    def _get_attention_memories(
        self,
        attention_level: float
    ) -> list[dict]:
        """Retrieve memories based on attention significance"""
        if attention_level < self.config.attention_threshold:
            return []
            
        return self.memory_core.get_memories_by_attention(
            min_attention=attention_level,
            limit=self.config.batch_size
        )
        
    def _get_emotional_memories(
        self,
        emotion_values: dict[str, float]
    ) -> list[dict]:
        """Retrieve emotionally coherent memories"""
        return self.memory_core.retrieve_similar_memories(
            emotion_query=emotion_values,
            k=self.config.batch_size
        )
        
    def _consolidate_memories(
        self,
        attention_memories: list[dict],
        emotional_memories: list[dict],
        current_state: dict[str, torch.Tensor]
    ) -> dict:
        """Consolidate memories through fusion and evaluation"""
        
        # Combine memory sets
        combined_memories = attention_memories + emotional_memories
        
        if not combined_memories:
            return {}
            
        # Get fusion output
        fusion_output, fusion_info = self.fusion.forward(
            state=current_state,
            memories=combined_memories
        )
        
        # Generate consolidated narrative
        narrative = self.fusion.generate_narrative(
            fusion_output=fusion_output,
            memories=combined_memories
        )
        
        return {
            'fusion_output': fusion_output,
            'fusion_info': fusion_info,
            'narrative': narrative,
            'source_memories': combined_memories
        }
        
    def _store_consolidated_memories(self, consolidated: dict):
        """Store consolidated memories"""
        if not consolidated:
            return
            
        self.consolidated_memories.append({
            'timestamp': np.datetime64('now'),
            'fusion_info': consolidated['fusion_info'],
            'narrative': consolidated['narrative']
        })
        
        # Prune old consolidated memories
        if len(self.consolidated_memories) > 1000:
            self.consolidated_memories = self.consolidated_memories[-1000:]
            
    def get_sync_status(self) -> dict:
        """Get current synchronization status"""
        return {
            'total_syncs': self.sync_counter,
            'consolidated_memories': len(self.consolidated_memories),
            'last_sync_time': self.consolidated_memories[-1]['timestamp'] if self.consolidated_memories else None,
            'memory_coherence': self._calculate_memory_coherence()
        }
        
    def _calculate_memory_coherence(self) -> float:
        """Calculate coherence of consolidated memories"""
        if len(self.consolidated_memories) < 2:
            return 0.0
            
        # Calculate narrative consistency
        narratives = [mem['narrative'] for mem in self.consolidated_memories[-100:]]
        consistency_scores = []
        
        for i in range(len(narratives) - 1):
            score = self.evaluator.calculate_narrative_similarity(
                narratives[i],
                narratives[i + 1]
            )
            consistency_scores.append(score)
            
        return float(np.mean(consistency_scores))