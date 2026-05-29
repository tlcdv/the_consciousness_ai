from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class LevinConsciousnessMetrics:
    """Metrics based on Michael Levin's principles of consciousness"""
    bioelectric_complexity: float = 0.0  # Measure of bioelectric field complexity
    morphological_adaptation: float = 0.0  # Ability to adapt internal representations
    collective_intelligence: float = 0.0  # Degree of holonic integration
    goal_directed_behavior: float = 0.0  # Evidence of purposeful behavior
    basal_cognition: float = 0.0  # Non-neural cognitive processes

    def get_overall_score(self) -> float:
        """Calculate overall Levin consciousness score"""
        metrics = [
            self.bioelectric_complexity,
            self.morphological_adaptation,
            self.collective_intelligence,
            self.goal_directed_behavior,
            self.basal_cognition
        ]
        return sum(metrics) / len(metrics)

class LevinConsciousnessEvaluator:
    """
    Evaluates consciousness based on Levin's theories of:
    1. Bioelectric signaling and field dynamics
    2. Collective intelligence across scales
    3. Goal-directed behavior and basal cognition
    4. Morphological computation
    """
    
    def __init__(self, config: dict):
        self.config = config
        
    def evaluate_bioelectric_complexity(self, bioelectric_state: dict[str, torch.Tensor]) -> float:
        """
        Evaluate complexity of bioelectric fields
        Similar to IIT's phi measure but focused on field dynamics
        """
        if not bioelectric_state:
            return 0.0
            
        # Calculate field differentials as a measure of complexity
        field_values = [field for field in bioelectric_state.values() if field is not None]
        if len(field_values) < 2:
            return 0.0
            
        # Calculate field gradients between components
        gradients = []
        for i in range(len(field_values)):
            for j in range(i + 1, len(field_values)):
                if field_values[i].shape == field_values[j].shape:
                    gradient = torch.norm(field_values[i] - field_values[j]).item()
                    gradients.append(gradient)
        
        if not gradients:
            return 0.0
            
        # Calculate mean gradient as complexity measure, normalized to [0, 1]
        mean_gradient = sum(gradients) / len(gradients)
        return float(min(1.0, mean_gradient / (mean_gradient + 1.0)))
        
    def evaluate_morphological_adaptation(
        self, 
        past_states: list[dict], 
        current_state: dict
    ) -> float:
        """
        Evaluate adaptation of internal representations over time
        Based on Levin's concept of morphological computation
        """
        if not past_states or not current_state:
            return 0.0
            
        # Check for state representation changes
        changes = []
        for past_state in past_states[-5:]:  # Look at last 5 states
            if 'integrated_state' in past_state and 'integrated_state' in current_state:
                past_integrated = past_state['integrated_state']
                current_integrated = current_state['integrated_state']
                
                if isinstance(past_integrated, torch.Tensor) and isinstance(current_integrated, torch.Tensor):
                    if past_integrated.shape == current_integrated.shape:
                        # Calculate cosine similarity as measure of change
                        similarity = torch.nn.functional.cosine_similarity(
                            past_integrated.reshape(1, -1), 
                            current_integrated.reshape(1, -1),
                            dim=1
                        ).item()
                        changes.append(1.0 - similarity)  # Convert to distance
        
        if not changes:
            return 0.0
            
        # Average change as adaptation score, clamped to [0, 1]
        return min(1.0, sum(changes) / len(changes))
        
    def evaluate_collective_intelligence(self, holonic_output: dict) -> float:
        """
        Evaluate the degree of integration between holonic components.

        Integration is the mean pairwise cosine similarity of the holon state
        vectors, mapped to [0, 1]: when the holons converge to similar states
        they are acting as a coherent collective (high integration); when they
        diverge they act independently (low integration). This is a
        structural-coherence proxy for Levin's notion of collective
        intelligence, the spatio-temporal scale a system measures and controls
        (Levin 2019, "The Computational Boundary of a 'Self'").

        Note (2026-05-29): the previous implementation used 1 - normalized
        entropy of the holonic attention matrix. With an untrained HolonicSystem
        the integration attention is near-uniform regardless of input, so that
        metric was inert (constant ~2e-6 across 64 diverse inputs and under
        training; see docs/results/levin_derisk_2026_05_29.md). The
        holon_states-based measure below responds to input because each holon
        applies its own learned map to the shared input, so the pairwise
        alignment of holon states varies with the input.
        """
        holon_states = holonic_output.get('holon_states')
        if not isinstance(holon_states, torch.Tensor):
            return 0.0

        # Collapse to [num_holons, features]
        states = holon_states.reshape(holon_states.shape[0], -1)
        n = states.shape[0]
        if n < 2:
            return 0.0

        normed = torch.nn.functional.normalize(states, dim=1, eps=1e-8)
        sim = normed @ normed.t()  # [n, n], diagonal entries are 1
        off_diag_sum = (sim.sum() - torch.diagonal(sim).sum())
        mean_cos = (off_diag_sum / (n * (n - 1))).item()

        # Map cosine [-1, 1] to integration [0, 1]
        return float(max(0.0, min(1.0, (mean_cos + 1.0) / 2.0)))
        
    def evaluate_goal_directed_behavior(
        self,
        actions: list[dict],
        goals: list[dict],
        outcomes: list[dict]
    ) -> float:
        """
        Evaluate evidence of goal-directed behavior
        Based on Levin's concept of goal-directedness
        """
        if not actions or not goals or not outcomes or len(actions) != len(goals) != len(outcomes):
            return 0.0
            
        # Calculate alignment between goals and outcomes
        alignments = []
        for goal, outcome in zip(goals, outcomes):
            if 'embedding' in goal and 'embedding' in outcome:
                goal_embed = goal['embedding']
                outcome_embed = outcome['embedding']
                
                if isinstance(goal_embed, torch.Tensor) and isinstance(outcome_embed, torch.Tensor):
                    if goal_embed.shape == outcome_embed.shape:
                        # Calculate cosine similarity as measure of alignment
                        similarity = torch.nn.functional.cosine_similarity(
                            goal_embed.reshape(1, -1),
                            outcome_embed.reshape(1, -1),
                            dim=1
                        ).item()
                        alignments.append(similarity)
        
        if not alignments:
            return 0.0
            
        # Average alignment as goal-directedness score
        return sum(alignments) / len(alignments)
        
    def evaluate_basal_cognition(self, component_states: dict[str, torch.Tensor]) -> float:
        """
        Evaluate non-neural cognitive processes
        Based on Levin's concept of basal cognition
        """
        if not component_states:
            return 0.0
            
        # Look for patterns in component activities
        component_values = [state.mean().item() for state in component_states.values() 
                           if isinstance(state, torch.Tensor)]
        
        if not component_values:
            return 0.0
            
        # Calculate coefficient of variation as measure of basal activity
        mean = np.mean(component_values)
        std = np.std(component_values)
        
        if mean == 0:
            return 0.0
            
        cv = std / mean
        
        # Normalize to 0-1 range (assuming cv range of 0-2)
        normalized_cv = min(cv / 2.0, 1.0)
        
        return normalized_cv
        
    def evaluate_levin_consciousness(
        self,
        bioelectric_state: dict[str, torch.Tensor],
        holonic_output: dict,
        past_states: list[dict],
        current_state: dict,
        actions: list[dict] = None,
        goals: list[dict] = None,
        outcomes: list[dict] = None,
        component_states: dict[str, torch.Tensor] = None
    ) -> dict[str, float]:
        """
        Evaluate consciousness metrics based on Levin's principles
        """
        # Set default values for optional parameters
        actions = actions or []
        goals = goals or []
        outcomes = outcomes or []
        component_states = component_states or {}
        
        # Calculate individual metrics, clamped to [0, 1]
        def _clamp01(v):
            return max(0.0, min(1.0, v))

        bioelectric_complexity = _clamp01(self.evaluate_bioelectric_complexity(bioelectric_state))
        morphological_adaptation = _clamp01(self.evaluate_morphological_adaptation(past_states, current_state))
        collective_intelligence = _clamp01(self.evaluate_collective_intelligence(holonic_output))
        goal_directed_behavior = _clamp01(self.evaluate_goal_directed_behavior(actions, goals, outcomes))
        basal_cognition = _clamp01(self.evaluate_basal_cognition(component_states))

        # Create metrics object
        metrics = LevinConsciousnessMetrics(
            bioelectric_complexity=bioelectric_complexity,
            morphological_adaptation=morphological_adaptation,
            collective_intelligence=collective_intelligence,
            goal_directed_behavior=goal_directed_behavior,
            basal_cognition=basal_cognition
        )
        
        # Return as dictionary with overall score
        return {
            'bioelectric_complexity': bioelectric_complexity,
            'morphological_adaptation': morphological_adaptation,
            'collective_intelligence': collective_intelligence,
            'goal_directed_behavior': goal_directed_behavior,
            'basal_cognition': basal_cognition,
            'overall_levin_score': metrics.get_overall_score()
        }

    def evaluate(self, state: dict | None = None) -> dict[str, float]:
        """Convenience adapter: evaluate Levin metrics from a single state dict.

        Pulls the structured inputs (bioelectric_state, holonic_output,
        past_states, current_state, actions, goals, outcomes, component_states)
        from the dict when present, falling back to empty or neutral defaults so
        the call never raises. ConsciousnessMonitor.periodic_update uses this
        path with only a flat core-state dict, where most structured inputs are
        absent (so the returned metrics are mostly zero there). The training
        loop calls evaluate_levin_consciousness directly with the full
        structured inputs.
        """
        state = state or {}
        return self.evaluate_levin_consciousness(
            bioelectric_state=state.get('bioelectric_state', {}),
            holonic_output=state.get('holonic_output', {}),
            past_states=state.get('past_states', []),
            current_state=state.get('current_state', state),
            actions=state.get('actions', []),
            goals=state.get('goals', []),
            outcomes=state.get('outcomes', []),
            component_states=state.get('component_states', {}),
        )