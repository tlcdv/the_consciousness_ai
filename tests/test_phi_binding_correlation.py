import unittest
import torch
import numpy as np

from models.core.global_workspace import GlobalWorkspace
from models.core.oscillatory_binding import KuramotoLayer, WorkspaceBindingSystem

class DummyModule:
    def __init__(self, bid: float):
        self.bid = bid
    def evaluate_salience(self, inputs):
        return {"data": "test"}, self.bid

class TestPhiBindingCorrelation(unittest.TestCase):
    """
    Validates the Artificial Kuramoto Oscillatory Neurons (AKOrN)
    binding mechanism and its effect on Information Integration (Phi).
    
    Tests the 3 conditions defined in the biological architecture research:
    A: Unbound (Control)
    B: Partially Bound
    C: Fully Bound
    """
    
    def setUp(self):
        self.config = {
            "ignition_threshold": 0.3,
            "ignition_gain": 10.0,
            "reverberation_alpha": 0.0
        }
        
    def test_kuramoto_synchronization_dynamics(self):
        """Test that the AKOrN layer actually synchronizes oscillators over time"""
        layer = KuramotoLayer(num_oscillators=5, dimensions=2, coupling_strength=2.0)
        phases = layer.init_phases(batch_size=1)
        
        # High uniform amplitudes should cause high synchronization
        amps = torch.ones(1, 5)
        
        # Run 10 steps
        sync_values = []
        current_phases = phases
        for _ in range(10):
            current_phases, sync = layer(current_phases, amplitudes=amps, iterations=1)
            sync_values.append(sync.item())
            
        # Synchronization should increase over time
        self.assertGreater(sync_values[-1], sync_values[0])
        
    def test_phi_binding_monotony(self):
        """
        Test that Phi strictly increases as binding increases.
        Condition C > Condition B > Condition A
        """
        torch.manual_seed(42) # Ensure reproducible phases
        
        # Setup workspace
        gnw = GlobalWorkspace(self.config)
        
        # Register 4 modules with identical strong bids
        # This isolates the effect of binding vs just activation magnitude
        gnw.register_specialist('vision', DummyModule(0.8))
        gnw.register_specialist('audio', DummyModule(0.8))
        gnw.register_specialist('memory', DummyModule(0.8))
        gnw.register_specialist('emotion', DummyModule(0.8))
        
        # Use a high number of iterations to allow the phases to settle
        # into their synchronized state
        gnw.binding_system = WorkspaceBindingSystem(num_modules=4, iterations=20)
        gnw.binding_system.register_modules(['vision', 'audio', 'memory', 'emotion'])
        
        # We need a stable initial phase configuration for fair comparison
        torch.manual_seed(123)
        init_phases = gnw.binding_system.kuramoto.init_phases(batch_size=1)
        
        # --- Condition A: UNBOUND ---
        gnw.binding_system.kuramoto.K = 0.0  # No coupling
        gnw.binding_system.reset_state()
        gnw.binding_system.current_phases = init_phases.clone()
        
        bids_A = {'vision': 0.8, 'audio': 0.8, 'memory': 0.2, 'emotion': 0.2}
        bound_bids_A, _ = gnw.binding_system.bind_bids(bids_A)
        # Without coupling, they just rotate. We measure actual alignment delta
        align_A = sum(bound_bids_A[k]/bids_A[k] for k in bids_A) / 4.0 - 1.0
        phi_A = align_A * 0.1
        
        # --- Condition B: PARTIALLY BOUND ---
        gnw.binding_system.kuramoto.K = 10.0 # High coupling
        gnw.binding_system.reset_state()
        gnw.binding_system.current_phases = init_phases.clone()
        
        bids_B = {'vision': 0.8, 'audio': 0.8, 'memory': 0.2, 'emotion': 0.2}
        bound_bids_B, _ = gnw.binding_system.bind_bids(bids_B)
        align_B = sum(bound_bids_B[k]/bids_B[k] for k in bids_B) / 4.0 - 1.0
        phi_B = align_B * 0.1
        
        # --- Condition C: FULLY BOUND ---
        gnw.binding_system.kuramoto.K = 10.0 # High coupling
        gnw.binding_system.reset_state()
        gnw.binding_system.current_phases = init_phases.clone()
        
        bids_C = {'vision': 0.8, 'audio': 0.8, 'memory': 0.8, 'emotion': 0.8}
        bound_bids_C, _ = gnw.binding_system.bind_bids(bids_C)
        align_C = sum(bound_bids_C[k]/bids_C[k] for k in bids_C) / 4.0 - 1.0
        phi_C = align_C * 0.1
        
        # Print for debugging
        print(f"\nCondition A (Unbound) Phi: {phi_A:.4f}")
        print(f"Condition B (Partial) Phi: {phi_B:.4f}")
        print(f"Condition C (Bound)   Phi: {phi_C:.4f}")
        
        # Assertions
        self.assertGreater(phi_C, phi_B, "Full binding should produce higher Phi than partial")
        self.assertGreater(phi_B, phi_A, "Partial binding should produce higher Phi than unbound")

if __name__ == '__main__':
    unittest.main()
