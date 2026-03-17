from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any
from models.self_model.bioelectric_signaling import BioelectricSignalingNetwork

class HolonUnit(nn.Module):
    """
    Implements Levin's concept of holons - entities that are both autonomous
    and part of a larger collective intelligence.
    
    Each holon:
    1. Maintains its own state representation
    2. Processes information autonomously
    3. Communicates with other holons through bioelectric signaling
    4. Participates in collective decision-making
    """
    def __init__(self, config: dict, id: int):
        super().__init__()
        self.id = id
        self.hidden_size = config['hidden_size']
        self.holon_type = config.get('holon_type', 'cognitive')
        
        # Holonic state representation
        self.state_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Feature networks based on holon type
        self.feature_networks = nn.ModuleDict({
            'goal_directed': nn.Linear(self.hidden_size, config.get('goal_dim', 32)),
            'memory': nn.Linear(self.hidden_size, config.get('memory_dim', 64)),
            'perception': nn.Linear(self.hidden_size, config.get('perception_dim', 64))
        })
        
        # Communication channel
        self.communication_channel = nn.Linear(self.hidden_size, self.hidden_size)
    
    def process(self, input_state: torch.Tensor) -> dict[str, torch.Tensor]:
        # Generate holonic state
        holon_state = self.state_network(input_state)
        
        # Process through feature networks
        features = {
            name: network(holon_state) 
            for name, network in self.feature_networks.items()
        }
        
        # Generate communication output
        comm_output = self.communication_channel(holon_state)
        
        return {
            'state': holon_state,
            'features': features,
            'communication': comm_output
        }

class HolonicSystem(nn.Module):
    """
    A system of interacting holons that collectively form higher-level cognition.
    
    Implements Levin's principles of:
    1. Collective intelligence through multi-scale organization
    2. Self-organization through bioelectric signaling
    3. Autonomous yet interconnected cognitive units
    """
    def __init__(self, config: dict):
        super().__init__()
        self.num_holons = config.get('num_holons', 8)
        
        # Create a collection of holons
        self.holons = nn.ModuleList([
            HolonUnit(config, i) for i in range(self.num_holons)
        ])
        
        # Bioelectric signaling between holons
        self.signaling = BioelectricSignalingNetwork(config)
        
        # Holonic integration network
        self.integration_network = nn.MultiheadAttention(
            embed_dim=config['hidden_size'],
            num_heads=config.get('integration_heads', 4)
        )
    
    def forward(self, input_states: torch.Tensor) -> dict[str, Any]:
        # Process inputs through individual holons
        holon_outputs = [holon.process(input_states) for holon in self.holons]
        
        # Extract holon states and communications
        holon_states = torch.stack([output['state'] for output in holon_outputs])
        communications = torch.stack([output['communication'] for output in holon_outputs])
        
        # Enable bioelectric signaling between holons
        component_states = {f"holon_{i}": holon_outputs[i]['state'] for i in range(self.num_holons)}
        updated_fields = self.signaling(component_states)
        
        # Integrate information through holonic attention
        integrated_state, attention_weights = self.integration_network(
            holon_states, holon_states, communications
        )
        
        # Collect all features across holons
        all_features = {}
        for i, output in enumerate(holon_outputs):
            for feature_name, feature_value in output['features'].items():
                if feature_name not in all_features:
                    all_features[feature_name] = []
                all_features[feature_name].append(feature_value)
        
        # Average features across holons
        integrated_features = {
            name: torch.mean(torch.stack(values), dim=0)
            for name, values in all_features.items()
        }
        
        return {
            'integrated_state': integrated_state,
            'holon_states': holon_states,
            'attention_weights': attention_weights,
            'integrated_features': integrated_features,
            'bioelectric_fields': updated_fields
        }