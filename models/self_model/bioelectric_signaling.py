from __future__ import annotations

import torch
import torch.nn as nn

class BioelectricSignalingNetwork(nn.Module):
    """
    Implements Levin's concept of bioelectric signaling for regulating
    information flow between cognitive components.
    
    This module creates a dynamic signaling network that:
    1. Establishes voltage-like gradients between memory and attention systems
    2. Facilitates pattern recognition through field dynamics
    3. Self-organizes into functional cognitive units
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.field_dim = config.get('field_dimension', 128)
        self.num_channels = config.get('bioelectric_channels', 8)
        
        # Bioelectric field projectors
        self.field_projector = nn.Linear(config['hidden_size'], self.field_dim * self.num_channels)
        
        # Signaling network
        self.signaling_layers = nn.ModuleList([
            nn.Linear(self.field_dim, self.field_dim) 
            for _ in range(config.get('signaling_layers', 3))
        ])
        
        # Gap junction simulation (information transfer between components).
        # batch_first=True so query/key/value are [batch, seq, embed]; the
        # per-component field is [B, num_channels, field_dim], which maps to
        # batch=B, seq=num_channels, embed=field_dim.
        self.gap_junction = nn.MultiheadAttention(
            embed_dim=self.field_dim,
            num_heads=config.get('gap_junction_heads', 4),
            dropout=config.get('gap_junction_dropout', 0.1),
            batch_first=True,
        )

    def forward(self, component_states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Process states through bioelectric signaling network.

        Each component state is projected to a field [B, num_channels, field_dim],
        passed through the signaling stack, then mixed with the other components'
        fields via a gap-junction cross-attention. Query is the component's own
        field; key/value are the other components' fields concatenated along the
        channel/sequence axis ([B, num_channels * (num_components - 1), field_dim]).
        With a single component there is nothing to mix, so the field passes
        through unchanged.
        """
        # Project component states to bioelectric fields
        fields = {}
        for component, state in component_states.items():
            fields[component] = self.field_projector(state).view(-1, self.num_channels, self.field_dim)

        # Simulate bioelectric diffusion and gap junction signaling
        updated_fields = {}
        for component, field in fields.items():
            # Apply signaling transforms
            for layer in self.signaling_layers:
                field = torch.relu(layer(field))

            # Simulate gap junctions with other components. Concatenate the
            # other fields along the sequence axis (dim=1) so the attention
            # gets a 3-D [batch, seq, embed] key/value, not a 4-D stack.
            others = [f for c, f in fields.items() if c != component]
            if others:
                other_fields = torch.cat(others, dim=1)
                field_attended, _ = self.gap_junction(field, other_fields, other_fields)
                updated_fields[component] = field_attended
            else:
                updated_fields[component] = field

        return updated_fields