import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class QualiaState:
    """
    Represents the geometric properties of a subjective experience.
    Based on IIT's concept of the 'Shape' of information structure (MICS)
    and mapped to a 3D phenomenological space.
    """
    intensity: float  # Magnitude/Salience (0.0 to 1.0)
    valence: float    # Emotional polarity (-1.0 to 1.0)
    complexity: float # Information density/Entropy (0.0 to 1.0)
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.intensity, self.valence, self.complexity], dtype=np.float32)

class QualiaMapper:
    """
    Maps high-dimensional Global Workspace states into a low-dimensional
    Phenomenological Space (Qualia).
    
    This bridges the gap between the mathematical vector space of the AI
    and the 'Subjective Feel' that we want to visualize/monitor.
    """
    
    def __init__(self, subspace_dim: int = 3):
        self.subspace_dim = subspace_dim
        # Reference vectors for alignment (can be learned over time)
        # For now, we use fixed heuristic bases.

    def map_state(self, workspace_tensor: torch.Tensor, goal_vector: torch.Tensor) -> QualiaState:
        """
        Projects the workspace state into Qualia Space.
        
        Args:
            workspace_tensor: The ignited state vector from the Global Workspace.
            goal_vector: The agent's current homeostatic goal vector (from Emotion Core).
            
        Returns:
            QualiaState object representing the 'feel' of the thought.
        """
        # Ensure tensor inputs on the same device
        if not isinstance(workspace_tensor, torch.Tensor):
            workspace_tensor = torch.tensor(workspace_tensor, dtype=torch.float32)
        device = workspace_tensor.device
        if not isinstance(goal_vector, torch.Tensor):
            goal_vector = torch.tensor(goal_vector, dtype=torch.float32, device=device)
        else:
            goal_vector = goal_vector.to(device)
            
        # 1. Intensity (L2 Norm)
        # How "loud" or salient is this thought?
        # A faint thought has low magnitude; an insight has high magnitude.
        intensity = torch.norm(workspace_tensor).item()
        # Normalize roughly to 0-1 range (assuming unit variance inputs)
        intensity = np.tanh(intensity) 

        # 2. Valence (Alignment)
        # Is this thought aligned with my goals?
        # Cosine similarity between Thought and Goal.
        # Aligned = Good (Positive), Misaligned = Bad (Negative).
        if workspace_tensor.shape != goal_vector.shape:
             # Handle shape mismatch via mean pooling or projection if needed
             # For prototype, we assume flattened compatibility or resize
             flat_ws = workspace_tensor.view(-1)
             flat_goal = goal_vector.view(-1)
             # Pad or truncate to match length for dot product
             min_len = min(len(flat_ws), len(flat_goal))
             valence = F.cosine_similarity(
                 flat_ws[:min_len].unsqueeze(0), 
                 flat_goal[:min_len].unsqueeze(0)
             ).item()
        else:
            valence = F.cosine_similarity(workspace_tensor.flatten().unsqueeze(0), goal_vector.flatten().unsqueeze(0)).item()

        # 3. Complexity (Entropy)
        # How rich is the information?
        # A pure tone is low complexity; a symphony is high complexity.
        # We calculate the Shannon entropy of the softmaxed vector.
        probs = F.softmax(workspace_tensor.flatten(), dim=0)
        # Add epsilon for numerical stability
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        # Normalize entropy (Max entropy for N dimensions is log(N))
        max_entropy = np.log(probs.numel())
        complexity = entropy / (max_entropy + 1e-9)

        return QualiaState(
            intensity=float(intensity),
            valence=float(valence),
            complexity=float(complexity)
        )
