from __future__ import annotations

import time
from collections import deque, defaultdict
import numpy as np
import torch

class GNWMetrics:
    def __init__(self, num_modules: int, ignition_threshold_delta: float = 0.5, 
                 min_modules_for_ignition: int = 1, logger=None, 
                 max_history_len: int = 1000, broadcast_content_window: int = 100):
        self.logger = logger
        self.num_modules = num_modules # Expected number of distinct information sources/types
        
        # For GNW Ignition Index based on broadcast strength
        self.broadcast_strength_history = deque(maxlen=max_history_len)
        self.last_broadcast_strength = 0.0
        self.ignition_threshold_delta = ignition_threshold_delta # Min change to count as new ignition peak
        self.min_modules_for_ignition = min_modules_for_ignition # If using module count from competition

        # For Global Availability Latency
        self.sensory_event_timestamps = {} # event_id: start_time
        self.broadcast_event_timestamps = {} # event_id: broadcast_time

        # For Content Diversity / Richness
        self.broadcast_content_signatures = deque(maxlen=broadcast_content_window) # Store hashes or simplified reps

        # For Event Reuse / Reportability
        self.event_module_access_log = defaultdict(lambda: defaultdict(list)) # event_id -> module_name -> [access_timestamps]

        print(f"GNWMetrics initialized. Num modules: {num_modules}, Ignition Delta: {ignition_threshold_delta}")

    def update_workspace_status(self, workspace_state: dict, step: int):
        """
        Receives the current state from GlobalWorkspace after a competition cycle.
        Args:
            workspace_state (dict): Expected keys:
                'active_content': The content that was broadcast (if any).
                'broadcast_strength': The strength of the winning bid.
                'competition_results': dict of all bids {module_name: bid_strength}.
                'winners': list of winning module names.
            step (int): Current simulation step.
        """
        current_strength = workspace_state.get('broadcast_strength', 0.0)
        self.broadcast_strength_history.append(current_strength)

        # --- 1. GNW Ignition Index ---
        # Simple version: count significant broadcast events
        ignition_detected_this_step = 0
        if current_strength > 0 and (current_strength - self.last_broadcast_strength) > self.ignition_threshold_delta:
            # Could also check if workspace_state.get('winners') meets min_modules_for_ignition
            ignition_detected_this_step = 1
            if self.logger:
                self.logger.log_scalar_data("gnw_ignition_event", step, 1, 
                                            {"strength": current_strength, 
                                             "winners": workspace_state.get('winners', [])})
        
        self.last_broadcast_strength = current_strength
        
        # More complex: Analyze pattern of broadcast_strength_history for "ignition events"
        # (e.g., sustained high activity, rapid increases)
        # For now, we log the raw strength and the simple event detection.
        if self.logger:
            self.logger.log_scalar_data("gnw_broadcast_strength", step, current_strength)

        # --- 2. Global Availability Latency (Update) ---
        active_content = workspace_state.get('active_content', {})
        if active_content: # Something was broadcast
            # Assuming broadcast content has an 'event_id' if it's tied to a sensory event
            event_id = active_content.get('metadata', {}).get('source_event_id')
            if event_id and event_id in self.sensory_event_timestamps:
                latency = time.time() - self.sensory_event_timestamps[event_id]
                if self.logger:
                    self.logger.log_scalar_data("gnw_availability_latency", step, latency, {"event_id": event_id})
                # Clean up to avoid re-calculating for the same event if broadcast multiple times
                # Or allow multiple broadcasts and log each. For now, simple cleanup.
                del self.sensory_event_timestamps[event_id] 

        # --- 3. Content Diversity / Richness ---
        if active_content:
            # Create a simple signature of the content for diversity tracking
            # This needs a more robust implementation based on content structure
            try:
                content_str = str(sorted(active_content.items()))
                signature = hash(content_str)
                self.broadcast_content_signatures.append(signature)
            except TypeError: # if content is not easily sortable/hashable
                self.broadcast_content_signatures.append(hash(str(active_content)))


    def log_sensory_event_start(self, event_id: str, timestamp: float | None = None):
        """Call this when a distinct sensory event begins, to track latency."""
        self.sensory_event_timestamps[event_id] = timestamp or time.time()

    def log_event_reuse(self, event_id: str, module_name: str, step: int, timestamp: float | None = None):
        """Logs when a module accesses/reuses information related to an event_id that might have been broadcast."""
        access_time = timestamp or time.time()
        self.event_module_access_log[event_id][module_name].append(access_time)
        if self.logger:
            self.logger.log_event(
                event_name="gnw_event_information_reuse",
                step=step,
                metadata={"event_id": event_id, "module_name": module_name, "access_time": access_time}
            )

    def calculate_periodic_gnw_metrics(self, step: int) -> dict:
        """Calculates summary GNW metrics over recent history."""
        metrics = {}
        
        # Ignition Index (e.g., average strength or frequency of ignitions)
        if self.broadcast_strength_history:
            metrics["avg_broadcast_strength"] = np.mean(self.broadcast_strength_history)
            # A more sophisticated ignition index could be calculated here based on patterns.
        
        # Content Diversity (e.g., number of unique content signatures)
        if self.broadcast_content_signatures:
            metrics["unique_broadcasts_window"] = len(set(self.broadcast_content_signatures))
            metrics["broadcast_window_size"] = len(self.broadcast_content_signatures)

        if self.logger and metrics:
            for key, value in metrics.items():
                self.logger.log_scalar_data(f"gnw_summary_{key}", step, value)
        
        return metrics

# Example Usage (Conceptual - would be driven by ConsciousnessMonitor)
if __name__ == '__main__':
    # Mock logger
    class MockLogger:
        def log_scalar_data(self, *args, **kwargs): print(f"LogScalar: {args}, {kwargs}")
        def log_event(self, *args, **kwargs): print(f"LogEvent: {args}, {kwargs}")

    gnw_metrics_calculator = GNWMetrics(num_modules=5, logger=MockLogger())
    
    # Simulate a sensory event
    event_1_id = "visual_stimulus_001"
    gnw_metrics_calculator.log_sensory_event_start(event_1_id)
    
    # Simulate a few workspace updates
    for i in range(5):
        mock_workspace_state = {
            'active_content': {"data": f"info_{i}", "metadata": {"source_event_id": event_1_id if i==2 else None}}, # Broadcast event_1 at step 2
            'broadcast_strength': np.random.rand() * 0.5 + (0.5 if i==2 else 0.1), # Higher strength at step 2
            'competition_results': {f"module_{j}": np.random.rand() for j in range(3)},
            'winners': [f"module_{np.random.randint(0,3)}"] if (np.random.rand() * 0.5 + 0.1) > 0.3 else []
        }
        gnw_metrics_calculator.update_workspace_status(mock_workspace_state, step=i)
        time.sleep(0.1) # Simulate time passing

    summary = gnw_metrics_calculator.calculate_periodic_gnw_metrics(step=5)
    print("GNW Summary Metrics:", summary)