# filepath: examples/emotional_agent_example.py

from models.self_model.action_selection_core import ActionSelectionCore
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.emotional_memory_core import EmotionalMemoryCore
from simulations.scenarios.emotional_scenarios import EmotionalScenario

def run_emotional_rl():
    # Configure emotional reward shaping
    reward_config = {
        "valence_weight": 0.1,
        "dominance_weight": 0.05,
        "arousal_penalty": 0.1,
        "arousal_threshold": 0.8
    }
    emotion_shaper = EmotionalRewardShaper(reward_config)

    # Configure RL
    rl_config = {"gamma": 0.95, "workspace_dim": 256, "action_dim": 2, "context_dim": 128}
    rl_core = ActionSelectionCore(rl_config, emotion_shaper)

    mem_config = {"capacity": 5000}
    memory_core = EmotionalMemoryCore(mem_config)

    scenario_config = {"threshold_stage_1": 100}
    scenario = EmotionalScenario(scenario_config)

    # Example training loop
    for episode in range(1000):
        state = scenario.get_initial_state()
        emotion_values = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        done = False
        
        while not done:
            # Stub for an action selection
            action = 0  
            base_reward = 1.0

            # Calculate shaped reward
            shaped_reward = rl_core.compute_reward(state, action, emotion_values, base_reward)

            # Next state logic
            next_state = {}  # Stub
            done = True  # Stub

            # Store in memory
            transition = {
                "state": state,
                "action": action,
                "emotion_values": emotion_values,
                "reward": shaped_reward,
                "next_state": next_state,
                "done": done
            }
            memory_core.store_transition(transition)

            # Policy update
            rl_core.update_policy(transition)

            state = next_state

        # Possibly update scenario stage
        scenario.update_scenario(rl_core)