from __future__ import annotations

import torch
import numpy as np
from models.memory.memory_core import MemoryCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.predictive.dreamerv3_wrapper import DreamerV3


class MetaLearner:
    """
    Meta-learning system for adapting to new emotional experiences and scenarios.
    Implements MAML-style meta-learning optimized for emotional reinforcement learning.
    """

    def __init__(self, config: dict):
        """
        Initialize the MetaLearner.

        Args:
            config: Dictionary of meta-learning config, expecting keys:
                - 'dreamerV3' -> sub-config for DreamerV3
                - 'meta_config' -> sub-dict with 'inner_learning_rate', 'meta_batch_size', 'adaptation_steps'
                - 'memory_config' -> sub-dict with 'context_length' or other memory fields
                - 'emotional_scale' -> global scalar for emotional reward scaling
        """
        self.config = config

        # Initialize memory; fallback to default capacity if not provided.
        self.memory = MemoryCore(capacity=config.get('memory_capacity', 1000))
        self.emotion_network = EmotionalGraphNetwork()

        # Use dictionary-based dreamer config, fallback if missing.
        dreamer_cfg = config.get('dreamerV3', {})
        self.dreamer = DreamerV3(dreamer_cfg)

        meta_cfg = config.get('meta_config', {})
        self.inner_lr = meta_cfg.get('inner_learning_rate', 0.01)
        self.meta_batch_size = meta_cfg.get('meta_batch_size', 16)
        self.adaptation_steps = meta_cfg.get('adaptation_steps', 5)

        # Fallback or retrieve emotional scale from config.
        self.emotional_scale = float(config.get('emotional_scale', 1.0))

        # Initialize meta-parameters used during adaptation.
        self.meta_parameters = {}
        self.initialize_meta_parameters()

    def initialize_meta_parameters(self) -> None:
        """
        Initialize meta-parameters for fast adaptation.
        """
        # Retrieve context_length if available, fallback to 32.
        memory_cfg = self.config.get('memory_config', {})
        context_length = memory_cfg.get('context_length', 32)

        self.meta_parameters = {
            'emotional_scale': torch.nn.Parameter(
                torch.ones(1) * self.emotional_scale
            ),
            'context_weights': torch.nn.Parameter(
                torch.randn(context_length)
            )
        }

    def inner_loop_update(self, task_data: dict) -> tuple[float, dict[str, torch.Tensor]]:
        """
        Perform the inner loop update for a single task to adapt parameters.

        Args:
            task_data: Dictionary containing task-specific data (states, actions, etc.).

        Returns:
            A tuple of (average loss over adaptation steps, adapted_params dict).
        """
        # Make a copy of parameters so we can adapt them locally.
        adapted_params = {k: v.clone() for k, v in self.meta_parameters.items()}
        task_loss = 0.0

        for _ in range(self.adaptation_steps):
            # Sample a batch of experiences from the task data.
            batch = self.memory.sample_batch(
                task_data,
                batch_size=self.meta_batch_size
            )

            loss, metrics = self.compute_adaptation_loss(batch, adapted_params)

            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
            adapted_params = {
                k: v - self.inner_lr * g
                for (k, v), g in zip(adapted_params.items(), grads)
            }

            task_loss += loss.item()

        avg_loss = task_loss / max(self.adaptation_steps, 1)
        return avg_loss, adapted_params

    def compute_adaptation_loss(
        self,
        batch: dict,
        params: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute the adaptation loss for a given batch, using the adapted parameters.

        Args:
            batch: A dictionary of data for the current batch (e.g. 'states', 'actions', etc.).
            params: Dictionary of meta-parameters being adapted.

        Returns:
            A tuple of (loss tensor, dictionary of metric values).
        """
        # Example: compute emotional embeddings from 'emotion_values' in batch.
        emotional_context = self.emotion_network.get_embeddings(batch['emotion_values'])

        # Multiply by the context weights (placeholder).
        weighted_context = emotional_context * params['context_weights']

        # Rescale rewards by emotional_scale.
        scaled_rewards = batch['rewards'] * params['emotional_scale']

        # Compute DreamerV3 loss. This is a placeholder function you’d define in DreamerV3.
        world_model_loss = self.dreamer.compute_loss(
            states=batch['states'],
            actions=batch['actions'],
            rewards=scaled_rewards,
            next_states=batch['next_states'],
            additional_context=weighted_context
        )

        metrics = {
            'world_model_loss': float(world_model_loss.item()),
            'emotional_scale': float(params['emotional_scale'].item())
        }

        return world_model_loss, metrics

    def adapt_to_task(self, task_data: dict) -> dict[str, object]:
        """
        Adapt the model to a new task or scenario.

        Args:
            task_data: Dictionary containing details about the new task (e.g. 'task_id', plus data).

        Returns:
            A dictionary of adaptation results, containing:
            - 'task_loss': average loss from the adaptation steps
            - 'adapted_params': the new locally adapted parameters
        """
        task_loss, adapted_params = self.inner_loop_update(task_data)

        # Optionally store the adaptation result in memory or a global register
        adaptation_record = {
            'task_id': task_data.get('task_id', 'unknown_task'),
            'adapted_params': adapted_params,
            'performance': -task_loss  # Higher is better if loss is negative
        }
        # If your MemoryCore supports storing adaptation results:
        self.memory.store_adaptation(adaptation_record)

        return {
            'task_loss': task_loss,
            'adapted_params': adapted_params
        }
