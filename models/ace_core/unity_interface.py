from __future__ import annotations

import uuid
import numpy as np
from typing import Any
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage, OutgoingMessage
from mlagents_envs.base_env import ActionTuple

# Define UUIDs for Side Channels (Must match Unity implementation)
CONSCIOUSNESS_CHANNEL_ID = uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7")
EMOTION_CHANNEL_ID = uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f8")

class ConsciousnessChannel(SideChannel):
    """
    Side Channel for sending internal consciousness metrics to Unity
    for visualization (Phi, GWN state, etc).
    """
    def __init__(self):
        super().__init__(CONSCIOUSNESS_CHANNEL_ID)

    def on_message_received(self, msg: IncomingMessage) -> None:
        """Handle messages coming FROM Unity (if any)."""
        pass

    def send_metrics(self, phi: float, gwn_active: bool, focus_content: str):
        """
        Send consciousness metrics to Unity.
        Format: [phi (float), gwn_active (float), focus_content (string)]
        """
        msg = OutgoingMessage()
        msg.write_float32(phi)
        msg.write_float32(1.0 if gwn_active else 0.0)
        msg.write_string(focus_content)
        self.queue_message_to_send(msg)

class EmotionChannel(SideChannel):
    """
    Side Channel for sending detailed emotional state to Unity.
    """
    def __init__(self):
        super().__init__(EMOTION_CHANNEL_ID)

    def on_message_received(self, msg: IncomingMessage) -> None:
        pass

    def send_emotion(self, valence: float, arousal: float, dominance: float, label: str):
        msg = OutgoingMessage()
        msg.write_float32(valence)
        msg.write_float32(arousal)
        msg.write_float32(dominance)
        msg.write_string(label)
        self.queue_message_to_send(msg)

class UnityACEInterface:
    def __init__(self, build_path: str | None = None, worker_id: int = 0):
        """
        Interface between the Python ACE Agent and the Unity Environment.
        
        Args:
            build_path: Path to the Unity executable (None for editor connection).
            worker_id: ID for parallel training.
        """
        self.consciousness_channel = ConsciousnessChannel()
        self.emotion_channel = EmotionChannel()
        
        self.env = UnityEnvironment(
            file_name=build_path, 
            worker_id=worker_id,
            side_channels=[self.consciousness_channel, self.emotion_channel]
        )
        self.env.reset()
        
        # Get behavior names
        self.behavior_names = list(self.env.behavior_specs.keys())
        if not self.behavior_names:
            raise ValueError("No behaviors found in Unity environment")
        
        self.behavior_name = self.behavior_names[0]
        print(f"Connected to Unity. Controlling Agent: {self.behavior_name}")

    def step(self, action_vector: np.ndarray):
        """
        Step the environment with an action.
        """
        action_tuple = ActionTuple()
        # Assuming continuous actions for now, adjust based on Unity setup
        action_tuple.add_continuous(action_vector.reshape(1, -1))
        
        self.env.set_actions(self.behavior_name, action_tuple)
        self.env.step()
        
        # Get new observation
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        
        done = len(terminal_steps) > 0
        reward = 0.0
        
        if done:
            obs = terminal_steps.obs[0] # Main visual observation
            reward = terminal_steps.reward[0]
            self.env.reset()
        else:
            obs = decision_steps.obs[0]
            reward = decision_steps.reward[0]
            
        return obs, reward, done

    def broadcast_consciousness_state(self, phi: float, content: str, emotion_state: dict[str, Any]):
        """
        Send internal state to Unity for debugging/visualization without affecting physics.
        """
        self.consciousness_channel.send_metrics(phi, True, content)
        
        if emotion_state:
            self.emotion_channel.send_emotion(
                emotion_state.get('valence', 0.0),
                emotion_state.get('arousal', 0.0),
                emotion_state.get('dominance', 0.0),
                emotion_state.get('label', 'neutral')
            )

    def close(self):
        self.env.close()
