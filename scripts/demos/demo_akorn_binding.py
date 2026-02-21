import torch
import sys
import os

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.core.oscillatory_binding import WorkspaceBindingSystem

def run_demo():
    print("==================================================")
    print("🧠 AKOrN Oscillatory Binding Demo (ICLR 2025 Model)")
    print("==================================================\n")
    
    # Initialize the binding system with 4 specialist modules
    binding_system = WorkspaceBindingSystem(num_modules=4, iterations=15)
    binding_system.register_modules(['visual_color', 'visual_shape', 'auditory', 'memory'])
    
    print("The system has 4 cognitive modules competing for the Global Workspace.")
    print("Binding relies on phase synchronization on an N-dimensional sphere.\n")
    
    # Setup initial predictable phases for fair comparison
    torch.manual_seed(42)
    init_phases = binding_system.kuramoto.init_phases(batch_size=1)
    
    # ---------------------------------------------------------
    # Scenario 1: Unrelated weak stimuli (Noise / Background)
    # ---------------------------------------------------------
    print("--- SCENARIO 1: Unrelated Weak Stimuli (Background Noise) ---")
    print("Inputs: Random low activation across all modules. No obvious object.")
    
    binding_system.kuramoto.K = 10.0 # High coupling capability
    binding_system.reset_state()
    binding_system.current_phases = init_phases.clone()
    
    # Weak, disparate bids
    bids_noise = {
        'visual_color': 0.2,
        'visual_shape': 0.1,
        'auditory': 0.15,
        'memory': 0.25
    }
    
    print(f"Initial Bids: {bids_noise}")
    bound_bids_noise, sync_R_noise = binding_system.bind_bids(bids_noise)
    
    # Calculate how much things aligned
    align_noise = sum(bound_bids_noise[k]/bids_noise[k] for k in bids_noise) / 4.0 - 1.0
    
    print(f"Final Bids:   {{k: round(v, 3) for k, v in bound_bids_noise.items()}}")
    print(f"-> Phase Alignment Boost: {align_noise * 100:.1f}%")
    print(f"-> Result: No strong binding occurred because the input amplitudes were too weak to overcome natural phase frequencies.\n")
    
    
    # ---------------------------------------------------------
    # Scenario 2: Multisensory Object (Cross-modal Binding)
    # ---------------------------------------------------------
    print("--- SCENARIO 2: Multisensory Object (e.g., A Barking Dog) ---")
    print("Inputs: Strong visual shape and loud auditory signal. Core binding scenario.")
    
    binding_system.reset_state()
    binding_system.current_phases = init_phases.clone()
    
    bids_dog = {
        'visual_color': 0.1,
        'visual_shape': 0.9,   # Seeing the dog shape
        'auditory': 0.85,      # Hearing the bark
        'memory': 0.6          # Retrieving "dog" concept
    }
    
    print(f"Initial Bids: {bids_dog}")
    bound_bids_dog, sync_R_dog = binding_system.bind_bids(bids_dog)
    
    align_dog = sum(bound_bids_dog[k]/bids_dog[k] for k in bids_dog) / 4.0 - 1.0
    
    print(f"Final Bids:   {{k: round(v, 3) for k, v in bound_bids_dog.items()}}")
    print(f"-> Phase Alignment Boost: {align_dog * 100:.1f}%")
    print(f"-> Shape bid boosted from 0.900 to {bound_bids_dog['visual_shape']:.3f}")
    print(f"-> Audio bid boosted from 0.850 to {bound_bids_dog['auditory']:.3f}")
    print(f"-> Result: The active modules pulled each other into phase. Their combined synchrony boosted their likelihood of entering the conscious workspace.\n")


    # ---------------------------------------------------------
    # Scenario 3: Subliminal vs Conscious (Ignition Threshold)
    # ---------------------------------------------------------
    print("--- SCENARIO 3: Binding tips the scale for Ignition ---")
    print("Assume the workspace ignition threshold is 0.75.")
    
    binding_system.reset_state()
    binding_system.current_phases = init_phases.clone()
    
    bids_sub = {
        'visual_color': 0.70,  # Below threshold
        'visual_shape': 0.72,  # Below threshold
        'auditory': 0.1,
        'memory': 0.2
    }
    
    print("Without binding, these isolated visual features (0.70, 0.72) would remain SUBCONSCIOUS (< 0.75).")
    
    bound_bids_sub, _ = binding_system.bind_bids(bids_sub)
    new_color = bound_bids_sub['visual_color']
    new_shape = bound_bids_sub['visual_shape']
    
    print(f"After Kuramoto Binding:")
    print(f"Color: 0.70 -> {new_color:.3f}")
    print(f"Shape: 0.72 -> {new_shape:.3f}")
    
    if new_shape >= 0.75:
        print("-> Result: Because color and shape synchronized, their binding boosted them OVER the threshold. The features are bound into a unified CONSCIOUS percept.")
    else:
        print("-> Result: Still subconscious.")

if __name__ == "__main__":
    run_demo()
