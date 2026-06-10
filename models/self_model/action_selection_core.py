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
    def __init__(self, workspace_dim: int, context_dim: int = 256,
                 spatial_conv: bool = False,
                 spatial_shape: tuple[int, int, int] | None = None):
        super().__init__()
        self.context_dim = context_dim

        # Optional convolutional front-end. When the policy reads a topographic
        # map (the --policy-input spatial-conv tap feeds the flattened obs_map),
        # a flat GRU input cannot exploit the spatial structure: every flat
        # readout of every representation plateaus far below a CNN over pixels
        # (obs_map_routing_2026_06_10.md). This conv stack restores spatial
        # processing in the perception->action path, trained by the control
        # gradient (the PFC params are in the policy optimizer). The input is the
        # detached obs_map, so no gradient flows into the tectum.
        self.spatial_conv = spatial_conv
        self.spatial_shape = spatial_shape
        if spatial_conv:
            assert spatial_shape is not None, "spatial_conv needs spatial_shape (C,H,W)"
            c, _, _ = spatial_shape
            self.conv = nn.Sequential(
                nn.Conv2d(c, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(32, 16, 3, stride=2, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d(4), nn.Flatten(),
                nn.Linear(16 * 4 * 4, context_dim), nn.GELU(),
            )
            gru_input = context_dim
        else:
            self.conv = None
            gru_input = workspace_dim

        # Recurrent layer to maintain context over time
        self.working_memory = nn.GRUCell(gru_input, context_dim)

        # Projects to the Striatum (Basal Ganglia input)
        self.striatum_projection = nn.Linear(context_dim, context_dim)

    def forward(self, workspace_broadcast: torch.Tensor, hidden_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            workspace_broadcast: [B, workspace_dim] Output of Global Workspace
                (or, when spatial_conv, the flattened topographic obs_map)
            hidden_context: [B, context_dim] Previous working memory state

        Returns:
            pfc_state: [B, context_dim] Stable representation for the Basal Ganglia
            new_hidden: [B, context_dim] Updated working memory
        """
        if self.conv is not None:
            b = workspace_broadcast.shape[0]
            spatial = workspace_broadcast.view(b, *self.spatial_shape)
            workspace_broadcast = self.conv(spatial)
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

        # Phase 5 deliverable 3 (P3): optionally make the self-vector causally
        # central by concatenating it onto the workspace broadcast that drives the
        # PFC, so the policy consumes and learns from the self-model (the PFC GRU's
        # input weights for the self-vector columns are trained by update_policy).
        # Default off keeps the PFC input dim = workspace_dim (baseline
        # bit-identical). When on, action-time, rollout/memory storage, and
        # update-time all use the same augmented state, so training stays
        # consistent.
        self.use_self_vector = config.get("use_self_vector", False)
        self.self_vector_dim = config.get("self_vector_dim", 64)
        # Policy input width. Defaults to the workspace broadcast dim, but can be
        # overridden when the policy reads a different tap (e.g. --policy-input
        # spatial feeds the larger flattened obs_map). Previously the PFC was
        # hardcoded to workspace_dim, so --policy-input spatial crashed the
        # Go/No-Go policy (it only worked with the --policy dqn diagnostic).
        self.policy_input_dim = config.get("policy_input_dim", self.workspace_dim)
        pfc_input_dim = self.policy_input_dim + (self.self_vector_dim if self.use_self_vector else 0)

        # Optional conv front-end on the PFC when the policy reads a topographic
        # map (--policy-input spatial-conv). spatial_shape is (C, H, W) of the
        # obs_map; pfc_input_dim must equal C*H*W (no self-vector in this mode).
        self.policy_spatial_conv = config.get("policy_spatial_conv", False)
        self.policy_spatial_shape = config.get("policy_spatial_shape", None)

        # Models
        self.pfc = PrefrontalCortex(
            pfc_input_dim, self.context_dim,
            spatial_conv=self.policy_spatial_conv,
            spatial_shape=self.policy_spatial_shape,
        ).to(self.device)
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

    def _augment(self, broadcast: torch.Tensor, self_vector: torch.Tensor | None) -> torch.Tensor:
        """Concatenate the self-vector onto the broadcast (P3) so the PFC consumes
        the self-model. No-op when use_self_vector is off. When on but no
        self_vector is supplied, the self-vector slot is zero-filled so the PFC
        input dim is always workspace_dim + self_vector_dim."""
        if broadcast.dim() == 1:
            broadcast = broadcast.unsqueeze(0)
        if not self.use_self_vector:
            return broadcast
        b = broadcast.shape[0]
        if self_vector is None:
            sv = torch.zeros(b, self.self_vector_dim, device=broadcast.device)
        else:
            sv = self_vector.detach().to(broadcast.device)
            if sv.dim() == 1:
                sv = sv.unsqueeze(0)
            if sv.shape[0] != b:
                sv = sv.expand(b, -1)
        return torch.cat([broadcast, sv], dim=-1)

    def select_action(self, workspace_broadcast: torch.Tensor, emotion_arousal: float = 0.5, rpe_cache: float = 0.0, self_vector: torch.Tensor | None = None) -> tuple[np.ndarray, float]:
        """
        Step the PFC and BG to determine the next action.
        Uses emotional arousal to scale exploration (panic/urgency vs calm precision).
        """
        self.pfc.eval()
        self.bg.eval()

        with torch.no_grad():
            # 1. Update PFC Working Memory with new conscious broadcast.
            # P3: optionally augment with the self-vector (no-op when disabled).
            workspace_broadcast = self._augment(workspace_broadcast, self_vector)

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
             narrative: str = "",
             self_vector: torch.Tensor | None = None,
             next_self_vector: torch.Tensor | None = None) -> dict[str, float]:
        """
        Process the environment step, compute Dopaminergic RPE, and store for learning.
        """
        # P3: build self-vector-augmented views for the PFC and the rollout buffer
        # (so the policy is driven by, and learns from, the self-model), but keep
        # the RAW broadcast for memory storage so the memory's fixed-dim
        # coherence/replay logic is unaffected. No-op when use_self_vector is off.
        if workspace_broadcast.dim() == 1:
            workspace_broadcast = workspace_broadcast.unsqueeze(0)
        if next_broadcast.dim() == 1:
            next_broadcast = next_broadcast.unsqueeze(0)
        aug_broadcast = self._augment(workspace_broadcast, self_vector)
        aug_next = self._augment(next_broadcast, next_self_vector)

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
            next_pfc, _ = self.pfc(aug_next, temp_hidden)
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
        
        # 4. Add to rollout buffer for formal backprop. Use the AUGMENTED views so
        # update_policy re-forwards the same PFC input the action was driven with.
        self.rollout_buffer.append({
            "state": aug_broadcast,
            "hidden": self.pfc_hidden.clone(), # Need the context used at that step
            "action": action_tensor,
            "reward": shaped_reward,
            "next_state": aug_next,
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

    def replay_update(self, experiences: list[dict]) -> dict[str, float]:
        """Policy update from replayed memory experiences.

        Unlike update_policy(), replay entries lack hidden states, next_states,
        and done flags. We reconstruct PFC states from zero hidden and compute
        simple discounted returns without bootstrapping. Loss is scaled by 0.5
        to prevent replay from overwhelming online learning.
        """
        valid = [e for e in experiences
                 if "state" in e and "action" in e and "reward" in e]
        if len(valid) < 4:
            return {}

        self.pfc.train()
        self.bg.train()

        # Reconstruct states: ensure [B, workspace_dim]
        states = []
        for e in valid:
            s = e["state"]
            if isinstance(s, np.ndarray):
                s = torch.tensor(s, dtype=torch.float, device=self.device)
            elif isinstance(s, torch.Tensor):
                s = s.to(device=self.device, dtype=torch.float)
            if s.dim() == 1:
                s = s.unsqueeze(0)
            states.append(s)
        states = torch.cat(states, dim=0)

        # P3: memory stores RAW broadcasts; the augmented PFC expects the
        # self-vector slot too. Zero-fill it for replay (replayed memory carries
        # no self-vector). Guard on the raw width so double-padding cannot happen.
        if self.use_self_vector and states.shape[-1] == self.workspace_dim:
            pad = torch.zeros(states.shape[0], self.self_vector_dim, device=self.device)
            states = torch.cat([states, pad], dim=-1)

        # Zero hidden for replay (no recurrent context)
        hiddens = torch.zeros(len(valid), self.context_dim, device=self.device)

        rewards = torch.tensor(
            [float(e["reward"]) for e in valid], device=self.device
        ).unsqueeze(1)

        actions = []
        for e in valid:
            a = e["action"]
            if isinstance(a, np.ndarray):
                a = torch.tensor(a, dtype=torch.float, device=self.device)
            elif isinstance(a, torch.Tensor):
                a = a.to(device=self.device, dtype=torch.float)
            actions.append(a)
        actions = torch.stack(actions)

        # Discounted returns (no bootstrapping, single sequence assumption)
        returns_list: list[float] = []
        R = 0.0
        for r in reversed(rewards):
            R = r.item() + self.gamma * R
            returns_list.insert(0, R)
        returns = torch.tensor(returns_list, device=self.device).unsqueeze(1)

        # Forward through PFC + BG
        pfc_states, _ = self.pfc(states, hiddens)
        _, values = self.bg(pfc_states)

        advantage = returns - values.detach()
        value_loss = nn.MSELoss()(values, returns)

        # Go/No-Go losses (same pattern as update_policy)
        go_signal = self.bg.direct_pathway(pfc_states)
        no_go_signal = self.bg.indirect_pathway(pfc_states)

        target_go = actions * (advantage > 0).float() - actions * (advantage < 0).float()
        go_loss = nn.MSELoss()(go_signal, target_go.detach())

        target_nogo = (advantage < 0).float().expand_as(no_go_signal)
        nogo_loss = nn.BCELoss()(no_go_signal, target_nogo.detach())

        stn_output = self.bg.stn_pathway(pfc_states)
        stn_target = torch.tanh(torch.abs(advantage).detach())
        stn_loss = nn.MSELoss()(stn_output, stn_target)

        actor_loss = go_loss + nogo_loss + 0.3 * stn_loss
        # Scale by 0.5 so replay doesn't dominate online learning
        total_loss = 0.5 * (actor_loss + 0.5 * value_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "replay_policy_loss": actor_loss.item(),
            "replay_value_loss": value_loss.item(),
            "replay_total_loss": total_loss.item(),
        }
