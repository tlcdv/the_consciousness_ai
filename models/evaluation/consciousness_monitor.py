"""
Consciousness Development Monitoring System

This module implements:
1. Tracking of consciousness development metrics
2. Stage transition monitoring
3. Development milestone validation
4. Integration with emotional and memory systems
"""
from __future__ import annotations

from typing import Any
import torch
import numpy as np
from dataclasses import dataclass
import time
from collections import deque

# Attempt to import necessary logging and newly created metric modules
try:
    from scripts.logging.metrics_logger import MetricsLogger
    from models.evaluation.iit_phi import IITMetrics
    from models.evaluation.gnw_metrics import GNWMetrics
except ImportError as e:
    print(f"Error importing base evaluation submodules (MetricsLogger, IITMetrics, GNWMetrics): {e}")
    print("Ensure all evaluation modules and scripts.logging.metrics_logger are in PYTHONPATH.")
    # Define dummy classes if imports fail for base functionality
    if 'MetricsLogger' not in globals():
        class MetricsLogger:
            def __init__(self, *args, **kwargs): print("Dummy MetricsLogger used.")
            def log_scalar_data(self, *args, **kwargs): pass
            def log_tensor_data(self, *args, **kwargs): pass
            def close(self): pass
    if 'IITMetrics' not in globals():
        class IITMetrics:
            def __init__(self, *args, **kwargs): print("Dummy IITMetrics used.")
            def calculate_phi_star_mismatched_decoding(self, *args, **kwargs): return 0.0
            def calculate_ces_graph_metrics(self, *args, **kwargs): return {}
    if 'GNWMetrics' not in globals():
        class GNWMetrics:
            def __init__(self, *args, **kwargs): print("Dummy GNWMetrics used.")
            def update_activations(self, *args, **kwargs): return 0
            def log_sensory_event_start(self, *args, **kwargs): pass
            def log_event_reuse(self, *args, **kwargs): pass

# Import existing specific metric calculators from the project
try:
    from .consciousness_metrics import (
        IntegratedInformationCalculator,
        GlobalWorkspaceTracker,
        PerturbationTester,
        SelfAwarenessMonitor,
        ConsciousnessCapabilityTester # <<< ADD THIS IMPORT
    )
    # Assuming LevinConsciousnessEvaluator might be in a different path or part of consciousness_metrics
    # If it's a separate file in the same directory, it would be:
    # from .levin_consciousness_metrics import LevinConsciousnessEvaluator
    # If it's as originally shown:
    from models.evaluation.levin_consciousness_metrics import LevinConsciousnessEvaluator
except ImportError as e:
    print(f"Error importing project-specific metric calculators (IntegratedInformationCalculator, etc.): {e}")
    # Define dummy classes for these as well if they are critical for instantiation
    if 'IntegratedInformationCalculator' not in globals():
        class IntegratedInformationCalculator:
            def __init__(self, *args, **kwargs): print("Dummy IntegratedInformationCalculator used.")
            def calculate_phi_approximation(self, *args, **kwargs): return 0.0
    if 'GlobalWorkspaceTracker' not in globals():
        class GlobalWorkspaceTracker:
            def __init__(self, *args, **kwargs): print("Dummy GlobalWorkspaceTracker used.")
            def detect_ignition(self, *args, **kwargs): return False, {}
    if 'PerturbationTester' not in globals():
        class PerturbationTester:
            def __init__(self, *args, **kwargs): print("Dummy PerturbationTester used.")
            def calculate_pci_approximation(self, *args, **kwargs): return 0.0
    if 'SelfAwarenessMonitor' not in globals():
        class SelfAwarenessMonitor:
            def __init__(self, *args, **kwargs): print("Dummy SelfAwarenessMonitor used.")
            def evaluate_self_awareness(self, *args, **kwargs): return {}
    if 'LevinConsciousnessEvaluator' not in globals():
        class LevinConsciousnessEvaluator:
            def __init__(self, *args, **kwargs): print("Dummy LevinConsciousnessEvaluator used.")
            def evaluate(self, *args, **kwargs): return {}
    if 'ConsciousnessCapabilityTester' not in globals(): # <<< ADD DUMMY CLASS
        class ConsciousnessCapabilityTester:
            def __init__(self, *args, **kwargs): print("Dummy ConsciousnessCapabilityTester used.")
            def run_all_tests(self, *args, **kwargs): return {}
            def shutdown_logger(self, *args, **kwargs): pass


class ConsciousnessMonitor:
    """
    Continuously monitors the system's state and calculates various theoretical
    and practical metrics related to consciousness and self-awareness.
    Provides data for the evaluation dashboard and logs metrics using MetricsLogger.
    """
    def __init__(self, consciousness_core=None, memory_system=None, config=None, experiment_name: str = "consciousness_default_run"):
        # Support single-arg call: ConsciousnessMonitor(config_dict)
        if config is None and isinstance(consciousness_core, dict):
            config = consciousness_core
            consciousness_core = None
        if config is None:
            config = {}
        self.core = consciousness_core
        self.memory = memory_system
        self.config = config
        self.update_interval = config.get('monitor_update_interval', 1.0) if isinstance(config, dict) else 1.0

        # Central Logger
        self.logger = MetricsLogger(experiment_name=experiment_name)
        print(f"ConsciousnessMonitor initialized. Logging for experiment: {experiment_name}")

        # Initialize existing metric calculators (from user's original design)
        # Config for these should be under a main key e.g., config['calculators']
        phi_config = config.get('phi_calculator_config', {})
        gwt_config = config.get('gwt_tracker_config', {})
        pci_config = config.get('pci_tester_config', {})
        self_awareness_config = config.get('self_awareness_monitor_config', {})
        levin_config = config.get('levin_evaluator_config', {})

        self.phi_calculator = IntegratedInformationCalculator(phi_config)
        self.gwt_tracker = GlobalWorkspaceTracker(gwt_config)
        self.pci_tester = PerturbationTester(pci_config, self.core)
        self.self_awareness_monitor = SelfAwarenessMonitor(self_awareness_config, self.core, self.memory)
        self.levin_evaluator = LevinConsciousnessEvaluator(levin_config) # Assuming it takes a config

        # Initialize newly added metric evaluators (IITMetrics, GNWMetrics)
        # Config for these might be under config['evaluation'] or a specific key
        eval_config = config.get('evaluation', {})
        gnw_num_modules = eval_config.get("gnw_num_modules", 10)
        self.gnw_evaluator = GNWMetrics(
            num_modules=gnw_num_modules,
            ignition_threshold_delta=eval_config.get("gnw_ignition_threshold_delta", 0.5),
            min_modules_for_ignition=eval_config.get("gnw_min_modules_for_ignition", 2),
            logger=self.logger
        )
        self.iit_evaluator = IITMetrics(logger=self.logger) # IITMetrics might take its own config too

        # Initialize ConsciousnessCapabilityTester
        # It requires an 'acm_agent_interface'. We'll assume the 'consciousness_core'
        # can serve this role or provide access to such an interface.
        # If 'consciousness_core' is not the interface, this part might need adjustment
        # or the interface needs to be passed to ConsciousnessMonitor's init.
        capability_tester_config = config.get('capability_tester_config', {})
        try:
            self.capability_tester = ConsciousnessCapabilityTester(
                agent_interface=self.core, # Assuming self.core is or provides the interface
                config=capability_tester_config,
                logger=self.logger # Pass the main logger, or it can create its own
            )
        except Exception as e:
            print(f"Error initializing ConsciousnessCapabilityTester: {e}. It will be disabled.")
            self.capability_tester = None


        # State for periodic updates and logging
        self.metric_history = deque(maxlen=config.get('history_length', 1000))
        self.last_periodic_update_time = 0
        self.step_count = 0 # For event-driven updates via update_step
        self.last_summary_log_time = time.time()
        self.summary_log_interval = eval_config.get("summary_log_interval_seconds", 60)


    def periodic_update(self, current_timestamp: float):
        """
        Periodically calculates and stores consciousness metrics based on the original design.
        This method calls the original set of metric calculators.
        """
        if current_timestamp - self.last_periodic_update_time < self.update_interval:
            return

        # These get_current_state() and get_recent_activity_log() methods
        # need to be implemented in the 'self.core' object.
        current_core_state = self.core.get_current_state() if hasattr(self.core, 'get_current_state') else {}
        recent_core_activity = self.core.get_recent_activity_log() if hasattr(self.core, 'get_recent_activity_log') else []

        phi_approx = self.phi_calculator.calculate_phi_approximation(current_core_state, recent_core_activity)
        ignition_detected, ignition_details = self.gwt_tracker.detect_ignition(current_core_state, recent_core_activity)
        pci_approx = self.pci_tester.calculate_pci_approximation(current_core_state)
        self_awareness_scores = self.self_awareness_monitor.evaluate_self_awareness(current_core_state)
        levin_metrics = self.levin_evaluator.evaluate(current_core_state) # Assuming evaluate method

        metrics_snapshot = {
            "timestamp": current_timestamp,
            "phi_approx_legacy": phi_approx, # Renamed to avoid clash if iit_evaluator logs 'phi_star'
            "gwt_ignition": ignition_detected,
            "gwt_details": ignition_details,
            "pci_approx": pci_approx,
            "self_awareness_scores": self_awareness_scores,
            "levin_metrics": levin_metrics,
            "emotional_valence": current_core_state.get('emotional_state', {}).get('valence'),
        }
        self.metric_history.append(metrics_snapshot)
        self.last_periodic_update_time = current_timestamp

        # Log these snapshot metrics using MetricsLogger
        for key, value in metrics_snapshot.items():
            if isinstance(value, (int, float, bool)):
                self.logger.log_scalar_data(f"periodic_{key}", self.step_count, value, {"source": "periodic_update"})
            elif isinstance(value, dict): # Log dictionary items as separate scalars if simple
                 for sub_key, sub_value in value.items():
                     if isinstance(sub_value, (int, float, bool)):
                        self.logger.log_scalar_data(f"periodic_{key}_{sub_key}", self.step_count, sub_value, {"source": "periodic_update"})
            # Add more sophisticated logging for complex structures if needed

    def update_step_metrics(self, step_data: dict):
        """
        Called with data from a specific step of the system's operation.
        This method orchestrates metric calculation using IITMetrics and GNWMetrics.

        Args:
            step_data (dict): Contains current data for metric calculation, e.g.
                                  'gnw_activations', 'system_hidden_states_t', etc.
        """
        self.step_count += 1

        if 'gnw_activations' in step_data:
            self.gnw_evaluator.update_activations(
                current_activations=step_data['gnw_activations'],
                step=self.step_count
            )
        if 'sensory_event_id' in step_data:
            self.gnw_evaluator.log_sensory_event_start(step_data['sensory_event_id'])
        if 'event_reuse_info' in step_data:
            info = step_data['event_reuse_info']
            self.gnw_evaluator.log_event_reuse(
                event_id=info['event_id'],
                module_name=info['module_name'],
                step=self.step_count
            )

        if 'system_hidden_states_t' in step_data and \
           'system_hidden_states_t_minus_1' in step_data and \
           'iit_partition_P' in step_data:
            self.iit_evaluator.calculate_phi_star_mismatched_decoding(
                z_t=step_data['system_hidden_states_t'],
                z_t_minus_1=step_data['system_hidden_states_t_minus_1'],
                partition_P=step_data['iit_partition_P'],
                step=self.step_count
            )
        
        # Placeholder for calling other specific metric evaluators that operate on step data
        # e.g., self.some_other_capability_tester.evaluate_at_step(step_data, self.step_count)

        # Optionally, run capability tests periodically or based on certain conditions
        # For example, run every N steps:
        # capability_test_frequency = self.config.get("evaluation", {}).get("capability_test_frequency_steps", 100)
        # if self.capability_tester and self.step_count % capability_test_frequency == 0:
        #     self.run_capability_tests(self.step_count)


        current_time = time.time()
        if current_time - self.last_summary_log_time > self.summary_log_interval:
            self.log_summary()
            self.last_summary_log_time = current_time

    def log_summary(self):
        """Logs a periodic summary of the system's state or key metrics."""
        summary_data = { "total_steps_processed": self.step_count }
        self.logger.log_scalar_data("monitor_operational_summary", self.step_count, summary_data, {"source": "ConsciousnessMonitor"})
        print(f"Step {self.step_count}: ConsciousnessMonitor summary logged.")

    def run_capability_tests(self, step: int, agent_interface_override=None):
        """
        Runs the suite of consciousness capability tests.
        The results are logged internally by ConsciousnessCapabilityTester.
        """
        if self.capability_tester:
            print(f"Step {step}: ConsciousnessMonitor triggering capability tests.")
            try:
                # If the interface provided at init was not sufficient, or a specific one is needed now
                interface_to_use = agent_interface_override if agent_interface_override else self.capability_tester.agent_interface
                if not interface_to_use: # Fallback if self.core wasn't a good interface and no override
                    print(f"Step {step}: Cannot run capability tests, agent interface not available.")
                    return {}

                # If the tester's interface was set at its init (e.g. self.core)
                # and is still valid, this call is fine.
                # If self.core was not the right interface, the tester might have been init with None
                # or a dummy, and this call might not work as expected without a proper interface.
                # The tester's __init__ uses self.core as the interface.
                test_results = self.capability_tester.run_all_tests(step=step)
                # The capability_tester already logs its own results.
                # The monitor could log a summary or specific high-level scores if needed.
                self.logger.log_scalar_data("capability_tests_executed", step, 1, {"source": "ConsciousnessMonitor"})
                # Example: Log a summary score if available
                # overall_score = self._calculate_overall_capability_score(test_results)
                # if overall_score is not None:
                #    self.logger.log_scalar_data("capability_tests_overall_score", step, overall_score)
                return test_results
            except Exception as e:
                print(f"Step {step}: Error running capability tests: {e}")
                self.logger.log_scalar_data("capability_tests_error", step, 1, {"error_message": str(e)})
                return {"error": str(e)}
        else:
            print(f"Step {step}: Capability tester not available.")
            return {"status": "Capability tester not initialized."}

    def evaluate_current_state(self) -> dict[str, Any]:
        """Evaluate the current consciousness state without arguments.
        Pulls state from core if available, otherwise returns baseline metrics."""
        state = {}
        if self.core and hasattr(self.core, 'get_current_state'):
            state = self.core.get_current_state()
        emotional = state.get('emotional_state', {}) if isinstance(state, dict) else {}
        attention = state.get('attention_level', 0.0) if isinstance(state, dict) else 0.0
        if isinstance(emotional, dict):
            valence = emotional.get('valence', 0.0)
            arousal = emotional.get('arousal', 0.0)
        else:
            valence = 0.0
            arousal = 0.0
        consciousness_score = min(1.0, (float(attention) + valence + arousal) / 3.0)
        return {
            'consciousness_score': consciousness_score,
            'attention_level': float(attention),
            'emotional_valence': valence,
            'emotional_arousal': arousal,
            'development_stage': 'initial' if consciousness_score < 0.3 else 'developing',
        }

    def evaluate_development(self, current_state=None, emotion_values=None,
                             attention_metrics=None, stress_level=0.0,
                             self_model_state=None, memory_state=None) -> dict[str, Any]:
        """Evaluate consciousness development progress.
        Supports multiple call signatures used by different tests."""
        self.step_count += 1
        # Extract attention level
        if isinstance(attention_metrics, dict):
            attention = attention_metrics.get('attention_level', 0.5)
        else:
            attention = 0.5
        # Extract emotion values
        ev = emotion_values or {}
        if isinstance(current_state, dict):
            ev = ev or current_state.get('emotion', {})
        valence = ev.get('valence', 0.5) if isinstance(ev, dict) else 0.5
        arousal = ev.get('arousal', 0.5) if isinstance(ev, dict) else 0.5
        # Compute scores purely from input metrics, no step-count inflation
        consciousness_level = min(1.0, attention * 0.4 + valence * 0.3 + arousal * 0.3)
        emotional_awareness = min(1.0, valence * 0.5 + arousal * 0.3 + attention * 0.2)
        memory_coherence = min(1.0, 0.3 + attention * 0.3 + valence * 0.2)
        consciousness_score = (consciousness_level + emotional_awareness + memory_coherence) / 3.0
        return {
            'consciousness_score': consciousness_score,
            'consciousness_level': consciousness_level,
            'emotional_awareness': emotional_awareness,
            'memory_coherence': memory_coherence,
            'attention_level': attention,
            'stress_level': float(stress_level),
            'development_stage': 'initial' if consciousness_score < 0.3 else 'developing',
            'step': self.step_count,
        }

    def evaluate_state(self, consciousness_state=None, emotional_context=None,
                       attention_metrics=None, **kwargs) -> dict[str, Any]:
        """Evaluate a given consciousness state with emotional and attention context."""
        self.step_count += 1
        if isinstance(attention_metrics, dict):
            attention = attention_metrics.get('attention_level', 0.5)
        else:
            attention = 0.5
        ec = emotional_context or {}
        if isinstance(ec, dict):
            valence = ec.get('valence', 0.5)
            arousal = ec.get('arousal', 0.5)
        else:
            valence = 0.5
            arousal = 0.5
        score = min(1.0, attention * 0.4 + valence * 0.3 + arousal * 0.3)
        return {
            'consciousness_score': score,
            'consciousness_level': score,
            'attention_level': attention,
            'development_stage': 'initial' if score < 0.3 else 'developing',
            'step': self.step_count,
        }

    def evaluate_emotional_awareness(self, state: dict) -> dict:
        """Evaluate emotional awareness from a state dict."""
        emotion = state.get('emotion', {})
        attention = state.get('attention', 0.5)
        valence = emotion.get('valence', 0.5) if isinstance(emotion, dict) else 0.5
        arousal = emotion.get('arousal', 0.5) if isinstance(emotion, dict) else 0.5
        awareness = min(1.0, (valence + arousal + float(attention)) / 3.0)
        return {
            'emotional_awareness': awareness,
            'confidence': min(1.0, awareness * 0.8 + 0.1),
        }

    def get_latest_periodic_metrics(self):
        if self.metric_history:
            return self.metric_history[-1]
        return None

    def get_periodic_metric_history(self):
        return list(self.metric_history)

    def shutdown(self):
        print("ConsciousnessMonitor shutting down...")
        self.logger.close()
        if self.capability_tester and hasattr(self.capability_tester, 'shutdown_logger'):
            # If the capability tester created its own logger instance, shut it down.
            # If it shares self.logger, this is redundant but safe.
            # The current ConsciousnessCapabilityTester scaffold creates its own if one isn't passed.
            # Since we pass self.logger, its shutdown_logger might not be strictly necessary here
            # if it only closes the logger we passed (which self.logger.close() already does).
            # However, it's good practice if it might manage other resources.
            self.capability_tester.shutdown_logger()
        print("ConsciousnessMonitor shutdown complete.")

if __name__ == '__main__':
    import torch # For example usage

    # Mock dependencies for example usage
    class MockCoreAgentInterface: # Renamed to reflect its role for CapabilityTester
        def get_current_state(self): return {"emotional_state": {"valence": 0.5, "arousal": 0.3}, "active_modules": 5}
        def get_recent_activity_log(self): return [{"event": "perception", "details": "object detected"}]
        
        # Methods expected by ConsciousnessCapabilityTester's mock tests
        def perform_action(self, action_command):
            print(f"MockCoreAgentInterface: Performing action '{action_command}'")
            return {"status": "success", "details": f"Action {action_command} completed."}

        def query(self, query_text):
            print(f"MockCoreAgentInterface: Received query '{query_text}'")
            return {"answer": "Mock answer to " + query_text, "confidence_raw": torch.rand(1).item()}
        
        def query_internal_focus(self):
            return {"focus": "simulating future states", "details": "high activity in predictive module"}


    class MockMemory: pass

    mock_config = {
        'monitor_update_interval': 0.5, 
        'history_length': 100,
        'phi_calculator_config': {}, 
        'gwt_tracker_config': {},
        'pci_tester_config': {},
        'self_awareness_monitor_config': {},
        'levin_evaluator_config': {},
        'capability_tester_config': {}, # Config for the capability tester
        'evaluation': { 
            "gnw_num_modules": 5,
            "gnw_ignition_threshold_delta": 0.5,
            "gnw_min_modules_for_ignition": 2,
            "summary_log_interval_seconds": 5 
        }
    }

    mock_core_instance = MockCoreAgentInterface() # Use the interface mock
    mock_memory_instance = MockMemory()

    monitor = ConsciousnessMonitor(
        consciousness_core=mock_core_instance, # Passed as core, also used as agent interface
        memory_system=mock_memory_instance,
        config=mock_config,
        experiment_name="consciousness_monitor_with_capability_tests"
    )

    # Simulate a few steps for update_step_metrics
    for i in range(1, 6):
        mock_gnw_activations = torch.rand(mock_config["evaluation"]["gnw_num_modules"])
        mock_hidden_states_t = torch.rand(128)
        mock_hidden_states_t_minus_1 = torch.rand(128)
        mock_partition = [list(range(64)), list(range(64, 128))]
        step_data = {
            'gnw_activations': mock_gnw_activations,
            'system_hidden_states_t': mock_hidden_states_t,
            'system_hidden_states_t_minus_1': mock_hidden_states_t_minus_1,
            'iit_partition_P': mock_partition,
        }
        if i == 2: step_data['sensory_event_id'] = f"event_{i}"
        if i == 3: step_data['event_reuse_info'] = {'event_id': f"event_{i-1}", 'module_name': "TestModule"}
        
        monitor.update_step_metrics(step_data)
        
        monitor.periodic_update(time.time())

        # Example: Run capability tests every 2 steps in this simulation
        if i % 2 == 0:
            print(f"\n--- Triggering Capability Tests at step {monitor.step_count} ---")
            monitor.run_capability_tests(step=monitor.step_count)
        
        print(f"--- End of simulated step {i} (Monitor step: {monitor.step_count}) ---")
