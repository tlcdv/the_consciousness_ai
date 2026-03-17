"""
Meta-Learning Module

Implements meta-learning for self-model adaptation through:
1. Learning rate adaptation
2. Loss function modulation
3. Architecture search

Based on the holonic principles described in the research paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

class ConsciousnessMetaLearner(nn.Module):
    """
    Meta-learning system for consciousness development through:
    1. Experience-based learning rate adaptation
    2. Loss function modulation based on emotional state
    3. Architecture search for optimal self-representation
    """

    def __init__(self, config: dict):
        super(ConsciousnessMetaLearner, self).__init__()
        self.config = config
        
        # Base learning rate
        self.base_lr = config.get("base_learning_rate", 0.001)
        self.min_lr = config.get("min_learning_rate", 0.0001)
        self.max_lr = config.get("max_learning_rate", 0.01)
        
        # Metalearning parameters
        self.adaptation_rate = config.get("adaptation_rate", 0.1)
        self.emotion_factor = config.get("emotion_factor", 0.2)
        
        # Success/failure tracking
        self.success_history = []
        self.history_window = config.get("history_window", 50)
        
        # Learned emotion weights
        self.emotion_weights = {
            "joy": 0.1,
            "sadness": -0.05,
            "fear": 0.2,  # Fear increases learning rate - important for survival
            "surprise": 0.15,
            "anger": 0.05,
            "disgust": -0.02,
            "trust": 0.0,
            "anticipation": 0.1
        }
        
        # Learning rate adaptation network
        self.lr_adapter = nn.Sequential(
            nn.Linear(config['state_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], 1),
            nn.Sigmoid()
        )
        
        # Loss modulation network
        self.loss_modulator = nn.Sequential(
            nn.Linear(config['emotion_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], 1),
            nn.Sigmoid()
        )

    def adapt_learning(
        self,
        current_state: torch.Tensor,
        emotional_context: dict[str, float]
    ) -> tuple[float, float]:
        """
        Adapt learning parameters based on current state and emotions
        
        Returns:
            tuple containing:
            - Adapted learning rate
            - Loss modulation factor
        """
        # Get base learning rate
        base_lr = self.config['base_learning_rate']
        
        # Compute learning rate adaptation
        lr_factor = self.lr_adapter(current_state)
        adapted_lr = base_lr * lr_factor
        
        # Compute loss modulation
        emotion_tensor = torch.tensor([v for v in emotional_context.values()])
        loss_factor = self.loss_modulator(emotion_tensor)
        
        return adapted_lr, loss_factor

    def update_architecture(
        self,
        performance_metrics: dict[str, float]
    ) -> dict[str, torch.Tensor]:
        """Update architecture based on performance metrics"""
        # TODO: Implement architecture search
        pass

    def compute_learning_rate(self, 
                             emotional_state: dict[str, float],
                             success_rate: float | None = None,
                             consciousness_level: float = 0.5) -> float:
        """Calculate adaptive learning rate based on emotions and performance
        
        Args:
            emotional_state: Current emotional state as dict of values
            success_rate: Optional success rate from recent interactions
            consciousness_level: Current consciousness level (0.0-1.0)
            
        Returns:
            Adapted learning rate
        """
        # Start with base rate
        lr = self.base_lr
        
        # Adjust based on success rate
        if success_rate is not None:
            # Higher success = smaller learning rate (less need to adapt)
            # Lower success = higher learning rate (need to adapt more)
            success_factor = 1.0 - (success_rate * 0.8)
            lr = lr * (0.5 + success_factor)
        
        # Adjust based on emotional state
        emotion_modifier = 1.0
        if emotional_state:
            # Calculate weighted sum of emotions
            emotion_sum = sum(
                emotional_state.get(emotion, 0) * weight 
                for emotion, weight in self.emotion_weights.items()
            )
            # Convert to multiplicative factor
            emotion_modifier = 1.0 + (emotion_sum * self.emotion_factor)
        
        # Adjust based on consciousness level
        # Higher consciousness = more focused learning
        consciousness_modifier = 0.5 + (consciousness_level * 0.5)
        
        # Apply modifiers
        lr = lr * emotion_modifier * consciousness_modifier
        
        # Ensure within bounds
        lr = max(self.min_lr, min(self.max_lr, lr))
        
        return lr
        
    def update_success_history(self, success: bool):
        """Update history of successful interactions
        
        Args:
            success: Whether the recent interaction was successful
        """
        self.success_history.append(1.0 if success else 0.0)
        
        # Keep history within window
        if len(self.success_history) > self.history_window:
            self.success_history.pop(0)
    
    def get_success_rate(self) -> float:
        """Calculate success rate from history
        
        Returns:
            Success rate between 0.0-1.0
        """
        if not self.success_history:
            return 0.5
            
        return sum(self.success_history) / len(self.success_history)
        
    def update_emotion_weights(self, reward: float, emotions: dict[str, float]):
        """Update emotion weights based on rewards
        
        Args:
            reward: Reward value from recent experience
            emotions: Emotions present during experience
        """
        if not emotions:
            return
            
        # Update weights based on correlation with rewards
        for emotion, value in emotions.items():
            if emotion in self.emotion_weights:
                # Positive reward strengthens weight in its direction
                # Negative reward weakens or reverses weight
                update = self.adaptation_rate * reward * value
                self.emotion_weights[emotion] += update
                
    def get_emotion_weights(self) -> dict[str, float]:
        """Get current emotion weights
        
        Returns:
            dict of emotion weights
        """
        return dict(self.emotion_weights)