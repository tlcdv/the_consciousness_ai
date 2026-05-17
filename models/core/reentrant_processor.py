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
        self.convergence_threshold = config.get("convergence_threshold", 0.05)
        
        # Tracking
        self.last_settle_result: SettleResult | None = None
        
    def settle(
        self,
        workspace,
        specialists: dict[str, Any],
        initial_bids: dict[str, float],
        payloads: dict[str, Any],
        goal_vector: torch.Tensor,
        pad_state: dict[str, float] | None = None,
        interoceptive_state: dict[str, float] | None = None,
    ) -> SettleResult:
        """
        Run the reentrant convergence loop.

        Args:
            workspace: GlobalWorkspace instance
            specialists: dict mapping module names to objects with receive_broadcast()
            initial_bids: Starting bid values from the first forward pass
            payloads: Semantic content payloads for each module
            goal_vector: Homeostasis target tensor
            pad_state: Optional PAD emotion to drive the workspace's affective
                modulator on every cycle.
            interoceptive_state: Optional homeostatic drives, paired with pad_state.

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
                payloads=payloads,
                pad_state=pad_state,
                interoceptive_state=interoceptive_state,
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
            
        def _extract_tensor(b):
            """Extract a tensor from a broadcast that is either a tensor or a
            dict (with "_fused" or "tensor" key). Returns None if neither."""
            if isinstance(b, torch.Tensor):
                return b
            if isinstance(b, dict):
                t = b.get("_fused")
                if isinstance(t, torch.Tensor):
                    return t
                t = b.get("tensor")
                if isinstance(t, torch.Tensor):
                    return t
            return None

        # Tensor PE path: works for raw tensors and for dict broadcasts that
        # contain a "_fused" (Phase A) or "tensor" (legacy) key. This handles
        # the attention_weighted fusion mode where broadcast_content is
        # {"_fused": tensor, "_weights": dict} instead of a bare tensor.
        curr_t = _extract_tensor(curr_broadcast)
        prev_t = _extract_tensor(prev_broadcast)
        if curr_t is not None and prev_t is not None:
            # Flatten and align shapes; broadcasts may differ in dim across cycles
            cf, pf = curr_t.view(-1), prev_t.view(-1)
            if cf.shape != pf.shape:
                n = min(cf.shape[0], pf.shape[0])
                cf, pf = cf[:n], pf[:n]
            diff = torch.norm(cf - pf)
            magnitude = torch.norm(cf) + 1e-8
            return (diff / magnitude).item()

        # Dict-only path (no extractable tensor): compare key sets safely
        if isinstance(curr_broadcast, dict) and isinstance(prev_broadcast, dict):
            all_keys = set(list(curr_broadcast.keys()) + list(prev_broadcast.keys()))
            if not all_keys:
                return 0.0
            # Skip tensor values to avoid the ambiguous-truth-value bug; compare
            # only scalar/string keys.
            def _scalar_eq(a, b):
                if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                    return False  # tensors handled above; treat as changed here
                return a == b
            changed = sum(
                1 for k in all_keys
                if not _scalar_eq(curr_broadcast.get(k), prev_broadcast.get(k))
            )
            return changed / (len(all_keys) + 1e-8)
        
        # Fallback: if types differ or are strings, binary comparison
        if curr_broadcast == prev_broadcast:
            return 0.0
        return 0.5  # Changed but can't quantify
