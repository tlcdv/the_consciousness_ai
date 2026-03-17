"""
Reentrant Processing Module (Phase 6)

Implements the adaptive predictive coding loop that wraps the Global Workspace
competition. Instead of a single pass (specialists → GNW → broadcast), this
module runs 5-10 recurrent cycles where:

    1. Specialists submit initial bids
    2. Workspace selects winners and broadcasts
    3. Specialists receive the broadcast (top-down prediction)
    4. Specialists update their bids (bottom-up prediction error)
    5. Workspace re-competes with updated bids
    6. Repeat until prediction error stabilizes (or max cycles reached)

Biological basis: Cortical recurrence settles within ~200ms (~5-10 relay 
cycles at ~10ms per relay). Easy/familiar stimuli settle in 2-3 cycles.
Novel or ambiguous stimuli require all 10.

Reference: Rao & Ballard (1999), Predictive Coding in the Visual Cortex.
"""
from __future__ import annotations

import torch
import numpy as np
from typing import Any
from dataclasses import dataclass, field


@dataclass
class SettleResult:
    """Output of the reentrant settle loop."""
    broadcast_content: Any                 # Final settled broadcast payload
    final_bids: dict[str, float]           # Final bid values after convergence
    phi: float                             # IIT Phi at settle
    is_conscious: bool                     # Whether ignition occurred
    cycles_used: int                       # How many cycles it took to settle
    prediction_errors: list[float]         # PE at each cycle (should decrease)
    converged: bool                        # Whether PE dropped below threshold


class ReentrantProcessor:
    """
    Wraps the GNW competition in an adaptive convergence loop.
    
    This is the computational analog of cortical recurrence: top-down 
    predictions meet bottom-up errors across multiple cycles until a 
    stable "settled" percept emerges. The settled state IS the conscious 
    content.
    """
    
    def __init__(self, config: dict = None):
        config = config or {}
        self.min_cycles = config.get("min_cycles", 2)
        self.max_cycles = config.get("max_cycles", 10)
        self.convergence_threshold = config.get("convergence_threshold", 0.01)
        
        # Tracking
        self.last_settle_result: SettleResult | None = None
        
    def settle(
        self,
        workspace,
        specialists: dict[str, Any],
        initial_bids: dict[str, float],
        payloads: dict[str, Any],
        goal_vector: torch.Tensor,
    ) -> SettleResult:
        """
        Run the reentrant convergence loop.
        
        Args:
            workspace: GlobalWorkspace instance
            specialists: dict mapping module names to objects with receive_broadcast()
            initial_bids: Starting bid values from the first forward pass
            payloads: Semantic content payloads for each module
            goal_vector: Homeostasis target tensor
            
        Returns:
            SettleResult with final broadcast, convergence metrics, etc.
        """
        current_bids = dict(initial_bids)
        prev_broadcast = None
        prediction_errors = []
        converged = False
        cycles_used = 0
        
        for cycle in range(self.max_cycles):
            cycles_used = cycle + 1
            
            # --- Run workspace competition with current bids ---
            broadcast_content, raw_bids = workspace.run_competition(
                inputs={},
                goal_vector=goal_vector,
                bids=current_bids,
                payloads=payloads
            )
            
            # --- Compute prediction error ---
            # PE = how much the broadcast changed from the previous cycle
            # If the broadcast is a tensor, use L2 distance
            # If nothing was broadcast (subconscious), PE is 0
            pe = self._compute_prediction_error(prev_broadcast, broadcast_content)
            prediction_errors.append(pe)
            
            # --- Early termination check ---
            # Only after minimum cycles
            if cycle >= self.min_cycles - 1 and pe < self.convergence_threshold:
                converged = True
                break
                
            # --- Top-down feedback to specialists ---
            # Send the current broadcast back down to each module.
            # Modules that support receive_broadcast() will update their bids.
            for name, specialist in specialists.items():
                if hasattr(specialist, 'receive_broadcast') and name in current_bids:
                    updated_bid = specialist.receive_broadcast(
                        broadcast_content, 
                        current_bids[name]
                    )
                    if updated_bid is not None:
                        current_bids[name] = float(updated_bid)
            
            prev_broadcast = broadcast_content
        
        # Build result
        result = SettleResult(
            broadcast_content=broadcast_content,
            final_bids=current_bids,
            phi=workspace.state.phi_value,
            is_conscious=workspace.state.is_conscious,
            cycles_used=cycles_used,
            prediction_errors=prediction_errors,
            converged=converged
        )
        
        self.last_settle_result = result
        return result
    
    def _compute_prediction_error(self, prev_broadcast: Any, curr_broadcast: Any) -> float:
        """
        Compute prediction error between consecutive cycles.
        
        For tensors: L2 distance normalized by dimensionality.
        For dicts/strings: binary change detection.
        For first cycle: returns 1.0 (maximum uncertainty).
        """
        if prev_broadcast is None:
            return 1.0  # First cycle: no prediction exists
            
        # Tensor broadcasts (from SensoryTectum)
        if isinstance(curr_broadcast, torch.Tensor) and isinstance(prev_broadcast, torch.Tensor):
            diff = torch.norm(curr_broadcast - prev_broadcast)
            # Normalize by broadcast magnitude to get relative error
            magnitude = torch.norm(curr_broadcast) + 1e-8
            return (diff / magnitude).item()
        
        # Dict broadcasts
        if isinstance(curr_broadcast, dict) and isinstance(prev_broadcast, dict):
            # Compare key sets and values
            if curr_broadcast == prev_broadcast:
                return 0.0
            # Count changed keys
            all_keys = set(list(curr_broadcast.keys()) + list(prev_broadcast.keys()))
            changed = sum(1 for k in all_keys if curr_broadcast.get(k) != prev_broadcast.get(k))
            return changed / (len(all_keys) + 1e-8)
        
        # Fallback: if types differ or are strings, binary comparison
        if curr_broadcast == prev_broadcast:
            return 0.0
        return 0.5  # Changed but can't quantify
