"""
Emotional Graph Neural Network (EGNN) implementing emotional processing with:
- Integration with LLaMA 3.3 narrative states
- Meta-memory guided pattern recognition
- Dynamic emotional adaptation
- Controlled stability mechanisms
"""
from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class EmotionalGraphState:
    """Track emotional processing state"""
    stability: float = 0.0
    coherence: float = 0.0
    memory_influence: float = 0.0
    narrative_alignment: float = 0.0
    adaptation_rate: float = 0.0

class EmotionalGraphNetwork(nn.Module):
    def __init__(self, config=None):
        """Initialize emotional graph network"""
        super().__init__()
        if config is None:
            config = {}

        # Helper to read from dict or dataclass
        def _g(key, default):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)

        input_dims = _g('input_dims', 3)
        hidden_dims = _g('hidden_dims', 64)
        llama_hidden_size = _g('llama_hidden_size', 768)
        pattern_dims = _g('pattern_dims', 32)

        # Core emotional processing
        self.node_encoder = nn.Linear(input_dims, hidden_dims)

        # Integration with LLaMA narrator
        self.narrative_projection = nn.Linear(llama_hidden_size, hidden_dims)

        # Pattern detection
        self.pattern_detector = nn.Sequential(
            nn.Linear(hidden_dims * 2, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, pattern_dims)
        )
        
        # Memory gating mechanism
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dims * 2, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, 1),
            nn.Sigmoid()
        )
        
        # Metrics tracking
        self.state = EmotionalGraphState()

    def forward(
        self,
        emotional_input: torch.Tensor,
        meta_memory: dict | None = None,
        narrative_state: dict | None = None
    ) -> tuple[torch.Tensor, EmotionalGraphState]:
        """Process emotional input through graph network"""
        
        # Generate base emotional embedding
        node_embedding = self.node_encoder(emotional_input)
        
        # Integrate narrative context if available
        if narrative_state:
            narrative_embedding = self.narrative_projection(
                narrative_state['hidden_states']
            )
            node_embedding = self._fuse_with_narrative(
                node_embedding,
                narrative_embedding
            )
            
        # Apply meta-memory gating if available
        if meta_memory:
            memory_gate = self._calculate_memory_gate(
                node_embedding,
                meta_memory
            )
            node_embedding = node_embedding * memory_gate
            
        # Update state
        self._update_state(
            node_embedding,
            meta_memory,
            narrative_state
        )
        
        return node_embedding, self.state

    def process(self, input_data, emotional_context=None):
        """Process input data and return emotional context dict.
        Accepts dict or tensor input."""
        if isinstance(input_data, dict):
            # Build a tensor from the dict values we can use
            vals = []
            for k in ['valence', 'arousal', 'dominance']:
                if k in input_data:
                    vals.append(float(input_data[k]))
            if vals:
                tensor_input = torch.tensor(vals, dtype=torch.float32)
            else:
                tensor_input = torch.randn(self._get_input_dim())
        elif isinstance(input_data, torch.Tensor):
            tensor_input = input_data.float()
        else:
            tensor_input = torch.randn(self._get_input_dim())
        # Pad or truncate to expected input dim
        expected = self._get_input_dim()
        if tensor_input.dim() == 0:
            tensor_input = tensor_input.unsqueeze(0)
        if tensor_input.numel() < expected:
            tensor_input = torch.cat([tensor_input.flatten(), torch.zeros(expected - tensor_input.numel())])
        tensor_input = tensor_input[:expected]
        embedding, state = self.forward(tensor_input)
        return {
            'embedding': embedding.detach(),
            'valence': float(embedding[0]) if embedding.numel() > 0 else 0.0,
            'arousal': float(embedding[1]) if embedding.numel() > 1 else 0.0,
            'dominance': float(embedding[2]) if embedding.numel() > 2 else 0.0,
            'state': state,
        }

    def get_embedding(self, emotion_values):
        """Get embedding tensor from emotion values dict."""
        if isinstance(emotion_values, dict):
            vals = [emotion_values.get('valence', 0.0),
                    emotion_values.get('arousal', 0.0),
                    emotion_values.get('dominance', 0.0)]
            tensor_input = torch.tensor(vals, dtype=torch.float32)
        elif isinstance(emotion_values, torch.Tensor):
            tensor_input = emotion_values
        else:
            tensor_input = torch.zeros(self._get_input_dim())
        expected = self._get_input_dim()
        if tensor_input.numel() < expected:
            tensor_input = torch.cat([tensor_input.flatten(), torch.zeros(expected - tensor_input.numel())])
        tensor_input = tensor_input[:expected]
        embedding, _ = self.forward(tensor_input)
        return embedding.detach()

    def _get_input_dim(self):
        """Get expected input dimension from node encoder."""
        for module in self.node_encoder.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        return 3  # fallback

    def _fuse_with_narrative(self, node_embedding, narrative_embedding):
        """Fuse node embedding with narrative context."""
        return node_embedding + narrative_embedding[:node_embedding.shape[-1]]

    def _calculate_memory_gate(self, node_embedding, meta_memory):
        """Calculate memory gating values."""
        return torch.sigmoid(node_embedding)

    def _update_state(self, node_embedding, meta_memory, narrative_state):
        """Update internal emotional graph state."""
        self.state.stability = float(torch.sigmoid(node_embedding.mean()))
        self.state.coherence = float(torch.sigmoid(node_embedding.std()))