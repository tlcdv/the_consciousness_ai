import torch
import numpy as np
try:
    import pyphi
except ImportError:
    pyphi = None
from typing import Optional, List, Dict, Any, Tuple
import logging
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)

class IITMetrics:
    def __init__(self, logger=None):
        """
        Calculates Integrated Information (Phi) using an Empirical TPM.
        Instead of using random logic, we build the Transition Probability Matrix
        from the agent's actual 'Working Memory' history.
        """
        self.logger = logger
        # Cache for PyPhi networks
        self.network_cache = {} 
        
        # Empirical History Buffer
        # Stores the binary state of the workspace for the last N steps.
        # Used to infer the causal structure (TPM).
        self.history_len = 100
        self.state_history = deque(maxlen=self.history_len)
        
        # PyPhi config
        if pyphi is not None:
            pyphi.config.PROGRESS_BARS = False
            pyphi.config.PARALLEL_CUTS = False

    def update_history(self, current_state: Tuple[int]) -> None:
        """Add the current binarized state to history."""
        self.state_history.append(current_state)

    def build_empirical_tpm(self, num_nodes: int) -> np.ndarray:
        """
        Constructs a TPM based on observed transitions in state_history.
        
        Args:
            num_nodes: Number of nodes in the subsystem (e.g., 4).
            
        Returns:
            TPM: (2^N, N) state-by-node transition matrix.
        """
        num_states = 2**num_nodes
        # Initialize counts with Laplace smoothing (alpha=1) to avoid zero probabilities
        # Shape: (From_State_Index, To_Node_Index) -> Count of times Node J turned ON given State I
        tpm_counts = np.ones((num_states, num_nodes)) 
        state_counts = np.ones(num_states) * 2 # Denominator (smoothing)
        
        if len(self.state_history) < 2:
            return np.random.rand(num_states, num_nodes) # Fallback if no history
            
        # Iterate through history pairs (t, t+1)
        history_list = list(self.state_history)
        for i in range(len(history_list) - 1):
            state_t = history_list[i]
            state_next = history_list[i+1]
            
            # Convert binary tuple to integer index (e.g., (1,0,1) -> 5)
            # Assumes state is tuple of 0s and 1s
            try:
                state_idx = 0
                for bit in state_t:
                    state_idx = (state_idx << 1) | bit
                
                # Check bounds (if history has different dimensions than current request)
                if state_idx >= num_states: continue

                state_counts[state_idx] += 1
                
                # Update node activation counts for next state
                for node_idx, bit in enumerate(state_next):
                    if node_idx < num_nodes and bit == 1:
                        tpm_counts[state_idx, node_idx] += 1
                        
            except Exception:
                continue
                
        # Normalize to get Probabilities
        # P(Node J = 1 | State I) = Count(I -> J=1) / Count(I)
        # Broadcasting division
        tpm = tpm_counts / state_counts[:, None]
        return tpm

    def calculate_phi(self, tpm: np.ndarray, current_state: tuple) -> float:
        """
        Calculate Phi using PyPhi on the provided TPM.
        """
        if pyphi is None:
            return 0.0
        try:
            num_nodes = len(current_state)
            if num_nodes > 5:
                return 0.0

            network = pyphi.Network(tpm)
            subsystem = pyphi.Subsystem(network, current_state)
            
            # Calculate Phi (Big Phi)
            sia = subsystem.sia() 
            if sia:
                return sia.phi
            return 0.0
            
        except Exception as e:
            # logger.error(f"Phi Calc Error: {e}") # Suppress for prototype speed
            return 0.0

    def compute_phi_proxy(self, global_workspace_state: torch.Tensor) -> float:
        """
        Main entry point.
        1. Extract Subsystem (Top K nodes).
        2. Binarize State.
        3. Update History.
        4. Build Empirical TPM.
        5. Calculate Phi.
        """
        subsystem_data = self.extract_subsystem_state(global_workspace_state)
        if not subsystem_data:
            return 0.0
            
        current_state = subsystem_data['state']
        num_nodes = len(current_state)
        
        # 1. Update History
        self.update_history(current_state)
        
        # 2. Build Empirical TPM (The "Causal Structure" of the agent's recent thoughts)
        tpm = self.build_empirical_tpm(num_nodes)
        
        # 3. Calculate Phi
        return self.calculate_phi(tpm, current_state)

    def extract_subsystem_state(self, attention_weights: torch.Tensor, threshold: float = 0.1) -> Dict[str, Any]:
        """
        Extracts the current binary state of the workspace.
        Does NOT generate a random TPM anymore.
        """
        if attention_weights.ndim < 1:
            return None
            
        # 1. Normalize and Select Top K
        # Assuming input is already a 1D tensor of "slot activations"
        k = 4
        if attention_weights.numel() < k:
            k = attention_weights.numel()
            
        top_values, top_indices = torch.topk(attention_weights, k)
        
        # 2. Binarize State
        current_state = tuple((top_values > threshold).int().tolist())
        
        return {
            "state": current_state,
            "node_indices": top_indices.tolist()
        }
