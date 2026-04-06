"""
Random Network Distillation (RND) curiosity module.

Generates intrinsic reward from prediction error between a fixed random
target network and a trainable predictor network. Operates on workspace
broadcast representations, so the consciousness pipeline's representational
quality directly determines exploration quality.

References:
- Burda et al. (2018) "Exploration by Random Network Distillation"
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RNDCuriosity(nn.Module):
    """Curiosity-driven intrinsic reward via Random Network Distillation.

    The target network is frozen (random initialization). The predictor
    network is trained to match the target's output. Novel states produce
    high prediction error (= high curiosity). As the predictor learns,
    familiar states produce low error.

    Because this module operates on workspace broadcast (not raw pixels),
    better workspace representations lead to better novelty detection,
    creating a functional link between consciousness quality and exploration.
    """

    def __init__(self, input_dim: int = 256, feature_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        # Fixed random target: produces random but consistent features.
        # Never trained. Parameters frozen.
        self.target_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )
        for p in self.target_network.parameters():
            p.requires_grad = False

        # Trainable predictor: tries to match target output.
        # Prediction error on novel inputs = curiosity signal.
        self.predictor_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, broadcast: torch.Tensor) -> tuple[float, torch.Tensor]:
        """Compute curiosity score and predictor loss.

        Args:
            broadcast: workspace broadcast tensor [B, input_dim] or [input_dim]

        Returns:
            curiosity_score: scalar float, higher = more novel
            predictor_loss: differentiable MSE loss for training the predictor
        """
        if broadcast.dim() == 1:
            broadcast = broadcast.unsqueeze(0)

        # Adapt input dimension if broadcast doesn't match expected size
        if broadcast.shape[-1] != self.input_dim:
            broadcast = broadcast[:, :self.input_dim] if broadcast.shape[-1] > self.input_dim else \
                nn.functional.pad(broadcast, (0, self.input_dim - broadcast.shape[-1]))

        with torch.no_grad():
            target_features = self.target_network(broadcast)

        predicted_features = self.predictor_network(broadcast)

        predictor_loss = nn.functional.mse_loss(predicted_features, target_features)
        curiosity_score = predictor_loss.item()

        return curiosity_score, predictor_loss
