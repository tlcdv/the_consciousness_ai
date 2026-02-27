"""
Memory Consolidation System

This module implements:
1. Memory optimization and cleanup
2. Consolidation of related memories
3. Temporal sequence management
4. Integration with emotional context

Dependencies:
- models/memory/emotional_memory_core.py for base storage
- models/memory/temporal_coherence.py for sequence tracking
- models/emotion/tgnn/emotional_graph.py for emotional context
"""

import torch
import torch.nn as nn 
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ConsolidationMetrics:
    """Tracks memory consolidation metrics"""
    consolidated_count: int = 0
    optimization_ratio: float = 0.0
    coherence_score: float = 0.0
    emotional_alignment: float = 0.0

class MemoryConsolidation:
    def __init__(self, config: Dict):
        """Initialize memory consolidation"""
        self.config = config
        self.metrics = ConsolidationMetrics()
        
    def consolidate_memories(
        self,
        memories: List[Dict],
        emotional_context: Optional[Dict] = None
    ) -> Tuple[List[Dict], ConsolidationMetrics]:
        """Consolidate related memories"""
        # Group related memories
        memory_groups = self._group_related_memories(memories)
        
        # Consolidate each group
        consolidated = []
        for group in memory_groups:
            merged = self._merge_memory_group(
                group,
                emotional_context
            )
            consolidated.append(merged)
            
        # Update metrics
        self.metrics.consolidated_count = len(consolidated)
        self.metrics.optimization_ratio = len(consolidated) / len(memories)
        
        return consolidated, self.metrics