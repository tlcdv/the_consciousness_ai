from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Any

from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore

class PrefrontalCortex(nn.Module):
    """
    Biological Counterpart: Prefrontal Cortex (PFC)
    
    Acts as the working memory and executive controller. It receives the highly 
    dynamic/sporadic broadcast from the Global Workspace and stabilizes it into 
    a persistent "policy context" (task goal/state representation).
    """
    def __init__(self, workspace_dim: int, context_dim: int = 256):
        super().__init__()
        self.context_dim = context_dim
        
        # Recurrent layer to maintain context over time
        self.working_memory = nn.GRUCell(workspace_dim, context_dim)
        
        # Projects to the Striatum (Basal Ganglia input)
        self.striatum_projection = nn.Linear(context_dim, context_dim)
        
    def forward(self, workspace_broadcast: torch.Tensor, hidden_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            workspace_broadcast: [B, workspace_dim] Output of Global Workspace
            hidden_context: [B, context_dim] Previous working memory state
            
        Returns:
            pfc_state: [B, context_dim] Stable representation for the Basal Ganglia
            new_hidden: [B, context_dim] Updated working memory
        """
        new_hidden = self.working_memory(workspace_broadcast, hidden_context)
        pfc_state = F.gelu(self.striatum_projection(new_hidden))
        return pfc_state, new_hidden

class BasalGanglia(nn.Module):
    """
    Biological Counterpart: Basal Ganglia (Striatum, GPi/SNr, GPe, STN, Thalamus)
    
    The brain's reinforcement learning engine. Evaluates the PFC's proposed state, 
    calculates expected value (Critic), and uses Go/No-Go pathways for action selection.
    
    The Reward Prediction Error (RPE) acts as simulated Dopamine to modulate the pathways.
    
    Includes:
    - Direct Pathway (Go): D1 receptors, excited by dopamine. Facilitates action.
    - Indirect Pathway (No-Go): D2 receptors, inhibited by dopamine. Suppresses action.
    - Hyperdirect Pathway (STN): Global inhibition. Emergency brake for uncertain states.
    - Thalamic Relay: Final output gating before motor execution.
    """
    def __init__(self, context_dim: int, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        
        # Striatum (Value estimator / Critic)
        self.critic = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Direct Pathway ("Go" - facilitates action)
        # D1 receptors: Excited by Dopamine
        self.direct_pathway = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # Proposes action magnitude
        )
        
        # Indirect Pathway ("No-Go" - inhibits action)
        # D2 receptors: Inhibited by Dopamine
        self.indirect_pathway = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
            nn.Sigmoid() # Gating/Inhibition strength [0, 1]
        )
        
        # Hyperdirect Pathway (STN - Subthalamic Nucleus)
        # Global emergency brake. Fires broadly to suppress ALL actions 
        # when the state is novel or highly uncertain. This gives the system 
        # time to evaluate before committing. Biologically, this is the 
        # "stop and think" signal that overrides both Go and No-Go.
        self.stn_pathway = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Global inhibition strength [0, 1]
        )
        
        # Thalamic Relay (final output gate)
        # In the brain, the thalamus relays BG output to motor cortex.
        # This adds a learned transformation so the raw Go/No-Go competition 
        # maps properly to the motor action space.
        self.thalamic_relay = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, pfc_state: torch.Tensor, dopamine_rpe: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates actions by comparing Go vs No-Go signals, with STN global inhibition.
        
        Args:
            pfc_state: [B, context_dim] from Prefrontal Cortex
            dopamine_rpe: Float proxy for current systemic dopamine level
            
        Returns:
            action_mean: [B, action_dim] The selected action vector
            value: [B, 1] The expected value of the state V(s)
        """
        # 1. Critic evaluates state
        value = self.critic(pfc_state)
        
        # 2. All three pathways evaluate simultaneously
        go_signal = self.direct_pathway(pfc_state)        # [-1, 1]
        no_go_signal = self.indirect_pathway(pfc_state)    # [0, 1]
        stn_brake = self.stn_pathway(pfc_state)            # [0, 1] global inhibition
        
        # 3. Dopaminergic Modulation
        # High dopamine strengthens 'Go' (D1) and weakens 'No-Go' (D2)
        # Low dopamine weakens 'Go' and strengthens 'No-Go'
        # STN is NOT modulated by dopamine (it operates independently)
        da_boost = torch.clamp(
            torch.tensor(dopamine_rpe, dtype=pfc_state.dtype, device=pfc_state.device), 
            -1.0, 1.0
        )
        
        modulated_go = go_signal * (1.0 + 0.5 * da_boost)
        modulated_nogo = no_go_signal * (1.0 - 0.5 * da_boost)
        
        # 4. Action gating: Go must overcome both No-Go AND STN inhibition
        # STN provides a global brake (same scalar applied to all action dims)
        global_release = 1.0 - stn_brake  # [B, 1] broadcast across action dims
        raw_action = modulated_go * (1.0 - modulated_nogo) * global_release
        
        # 5. Thalamic relay transforms to motor space
        action_mean = self.thalamic_relay(raw_action)
        
        # Ensure action bounds
        action_mean = torch.clamp(action_mean, -1.0, 1.0)
        
        return action_mean, value

class ActionSelectionCore:
    """
    Replaces ReinforcementCore. Integrates PFC, Basal Ganglia, and Amygdala (Emotion).
    """
    def __init__(self, config: dict[str, Any], emotion_shaper: EmotionalRewardShaper, memory: MemoryCore):
        self.config = config
        self.emotion_shaper = emotion_shaper
        self.memory = memory
        
        # Hyperparameters
        self.workspace_dim = config.get("workspace_dim", 256)
        self.context_dim = config.get("context_dim", 256)
        self.action_dim = config.get("action_dim", 4)
        self.gamma = config.get("gamma", 0.99)
        self.lr = config.get("learning_rate", 3e-4)
        
        self.device = config.get("device", "cpu")
        
        # Models
        self.pfc = PrefrontalCortex(self.workspace_dim, self.context_dim).to(self.device)
        self.bg = BasalGanglia(self.context_dim, self.action_dim).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.pfc.parameters()) + list(self.bg.parameters()), 
            lr=self.lr
        )
        
        # State
        self.pfc_hidden = torch.zeros(1, self.context_dim, device=self.device)
        self.last_value = 0.0
        self.rollout_buffer = []

    def reset_state(self, batch_size: int = 1):
        """Reset the PFC working memory between episodes"""
        self.pfc_hidden = torch.zeros(batch_size, self.context_dim, device=self.device)
        self.last_value = 0.0

    def select_action(self, workspace_broadcast: torch.Tensor, emotion_arousal: float = 0.5, rpe_cache: float = 0.0) -> tuple[np.ndarray, float]:
        """
        Step the PFC and BG to determine the next action.
        Uses emotional arousal to scale exploration (panic/urgency vs calm precision).
        """
        self.pfc.eval()
        self.bg.eval()
        
        with torch.no_grad():
            # 1. Update PFC Working Memory with new conscious broadcast
            # Ensure proper batch dimensions
            if workspace_broadcast.dim() == 1:
                workspace_broadcast = workspace_broadcast.unsqueeze(0)
            
            pfc_state, self.pfc_hidden = self.pfc(workspace_broadcast, self.pfc_hidden)
            
            # 2. Basal Ganglia logic
            action_mean, value = self.bg(pfc_state, dopamine_rpe=rpe_cache)
            
            # 3. Emotional Modulation of Exploration (Temperature)
            # High arousal -> high variance (panic/frantic search)
            # Low arousal -> low variance (calculated, habitual execution)
            base_noise_std = 0.1
            arousal_scaler = max(0.1, emotion_arousal * 2.0) # Arousal is typically [-1, 1] mapped to [0, 1] elsewhere. Assuming [0, 1] here.
            noise_std = base_noise_std * (1.0 + arousal_scaler)
            
            noise = torch.randn_like(action_mean) * noise_std
            action = action_mean + noise
            action = torch.clamp(action, -1.0, 1.0)
            
        self.last_value = value.item()
        return action.squeeze(0).cpu().numpy(), self.last_value

    def step(self, 
             workspace_broadcast: torch.Tensor, 
             action: np.ndarray, 
             raw_reward: float, 
             next_broadcast: torch.Tensor, 
             done: bool, 
             emotion_state: dict[str, float],
             attention_level: float,
             narrative: str = "") -> dict[str, float]:
        """
        Process the environment step, compute Dopaminergic RPE, and store for learning.
        """
        
        # 1. Emotional Reward Shaping
        shaped_reward = self.emotion_shaper.compute_emotional_reward(
            emotion_values=emotion_state,
            base_reward=raw_reward,
            context={"adaptation_detected": False}
        )
        
        # 2. Calculate local RPE (Reward Prediction Error) for immediate Dopamine proxy
        # RPE = r + gamma * V(s') - V(s)
        # Note: In a real training step we recompute V(s') with gradients, but we need
        # a fast proxy here to feed back into the BG forward pass.
        with torch.no_grad():
            if workspace_broadcast.dim() == 1:
                workspace_broadcast = workspace_broadcast.unsqueeze(0)
                next_broadcast = next_broadcast.unsqueeze(0)
            
            # Peek at next state value using current working memory context securely
            temp_hidden = self.pfc_hidden.clone()
            next_pfc, _ = self.pfc(next_broadcast, temp_hidden)
            _, next_value_tensor = self.bg(next_pfc)
            next_value = next_value_tensor.item()
            
        rpe = shaped_reward + (self.gamma * next_value * (1 - int(done))) - self.last_value
        
        # 3. Store in Memory
        action_tensor = torch.tensor(action, device=self.device)
        self.memory.store_experience(
            state=workspace_broadcast.squeeze(0), # Store the conscious state, without batch dim
            action=action_tensor,
            reward=shaped_reward,
            emotion_values=emotion_state,
            attention_level=attention_level,
            narrative=narrative
        )
        
        # 4. Add to rollout buffer for formal backprop
        self.rollout_buffer.append({
            "state": workspace_broadcast,
            "hidden": self.pfc_hidden.clone(), # Need the context used at that step
            "action": action_tensor,
            "reward": shaped_reward,
            "next_state": next_broadcast,
            "done": done,
            "rpe": rpe # The dopamine spike
        })
        
        return {
            "raw_reward": raw_reward,
            "shaped_reward": shaped_reward,
            "dopamine_rpe": rpe
        }
        
    def update_policy(self) -> dict[str, float]:
        """
        Train the pathways. Uses standard policy gradients / Actor-Critic mathematics
        to update the BG and PFC structures.
        """
        if len(self.rollout_buffer) < 10:
            return {}
            
        self.pfc.train()
        self.bg.train()
        
        states = torch.cat([x["state"] for x in self.rollout_buffer], dim=0)
        hiddens = torch.cat([x["hidden"] for x in self.rollout_buffer], dim=0)
        rewards = torch.tensor([x["reward"] for x in self.rollout_buffer], device=self.device).unsqueeze(1)
        next_states = torch.cat([x["next_state"] for x in self.rollout_buffer], dim=0)
        dones = torch.tensor([x["done"] for x in self.rollout_buffer], device=self.device).unsqueeze(1)
        cached_rpes = torch.tensor([x["rpe"] for x in self.rollout_buffer], device=self.device).float()
        
        # Calculate Returns
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d: R = 0
            R = r.item() + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device).unsqueeze(1)
        
        # Forward Pass
        # To accurately backprop through time, we must pass the sequences.
        # For this simplified continuous batch, we just evaluate the 1-step transitions.
        pfc_states, _ = self.pfc(states, hiddens)
        _, values = self.bg(pfc_states)
        
        # Real Advantage 
        advantage = returns - values.detach()
        value_loss = nn.MSELoss()(values, returns)
        
        # Custom Actor Loss matching to Go/No-Go
        # Since this is a continuous action space without probability distributions implemented, 
        # we construct a loss that encourages Go when advantage > 0 and No-Go when advantage < 0.
        
        # Get the pathway activations again manually for the loss logic
        go_signal = self.bg.direct_pathway(pfc_states)
        no_go_signal = self.bg.indirect_pathway(pfc_states)
        
        # If advantage > 0 (good action): We want Go to be large, No-Go to be small
        # If advantage < 0 (bad action): We want No-Go to be large, Go to be small
        # This is a highly stylized loss representing dopaminergic learning in the BG
        
        # Sign of advantage directs learning
        adv_sign = torch.sign(advantage)
        
        # Maximize go_signal magnitude in the direction of the taken action if Good
        # Minimize it if Bad
        actions = torch.stack([x["action"] for x in self.rollout_buffer])
        
        # Go Loss: Move `go_signal` towards `action * advantage_sign`
        # Using MSE as a proxy for pushing the network
        target_go = actions * (advantage > 0).float() - actions * (advantage < 0).float()
        go_loss = nn.MSELoss()(go_signal, target_go.detach())
        
        # No-Go Loss: Increase inhibition if bad, decrease if good
        target_nogo = (advantage < 0).float() # 1 if bad, 0 if good
        # Match target shape
        target_nogo = target_nogo.expand_as(no_go_signal)
        nogo_loss = nn.BCELoss()(no_go_signal, target_nogo.detach())
        
        # STN Loss: Global brake should activate when advantage magnitude is high
        # (uncertain about whether action is good or bad = should pause)
        # and deactivate when advantage is near zero (well-predicted states)
        stn_output = self.bg.stn_pathway(pfc_states)
        # Target: high brake for high |advantage| (uncertainty), low brake for low |advantage|
        advantage_magnitude = torch.abs(advantage).detach()
        # Normalize to [0, 1] range using tanh
        stn_target = torch.tanh(advantage_magnitude)
        stn_loss = nn.MSELoss()(stn_output, stn_target)
        
        actor_loss = go_loss + nogo_loss + 0.3 * stn_loss
        
        total_loss = actor_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.rollout_buffer = []
        
        return {
            "policy_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item()
        }
