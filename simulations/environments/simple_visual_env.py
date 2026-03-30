from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
from typing import Any

from simulations.environments.audio_mixin import DarkRoomAudioMixin


class SimpleVisualEnv(DarkRoomAudioMixin, gym.Env):
    """
    A lightweight, visual environment for testing Artificial Consciousness.
    Rendered via PyGame to provide raw pixel input to Vision Models (Qwen2-VL).
    
    Scenario: 'The Dark Room' (Emotional Bootstrapping)
    - Agent starts in the dark (High Anxiety).
    - Light source exists at a fixed or random location.
    - Agent must find the light to reduce Anxiety (Prediction Error).
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        
        # Action Space: [Move X, Move Y] (Continuous -1.0 to 1.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation Space: RGB Image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )
        
        # State
        self.agent_pos = np.array([width // 2, height // 2], dtype=np.float32)
        self.light_pos = np.array([width // 4, height // 4], dtype=np.float32)
        self.agent_radius = 20
        self.light_radius = 40
        self.battery = 1.0
        
        # PyGame Setup
        self.window = None
        self.clock = None
        pygame.init()
        self._canvas = pygame.Surface((self.width, self.height))
        
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        # Randomize positions
        self.agent_pos = np.random.rand(2) * [self.width, self.height]
        self.light_pos = np.random.rand(2) * [self.width, self.height]
        self.battery = 1.0
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
            
        return observation, info
        
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Update Physics
        move = np.clip(action, -1.0, 1.0) * 10.0 # Speed
        self.agent_pos += move
        self.agent_pos = np.clip(self.agent_pos, 0, [self.width, self.height])
        
        # Decay Battery
        self.battery -= 0.001
        
        # Calculate Distance to Light
        dist = np.linalg.norm(self.agent_pos - self.light_pos)
        in_light = dist < (self.light_radius + self.agent_radius)
        
        # Reward Function (Standard RL - NOT Emotional RL)
        # We return a basic signal; the Agent's Brain will interpret this emotionally.
        # Here: 1.0 if in light, -0.01 step penalty
        reward = 1.0 if in_light else -0.01
        
        terminated = self.battery <= 0
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()

        # Generate audio waveform from environment state (DarkRoomAudioMixin)
        info["audio_waveform"] = self._generate_audio(info)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info
        
    def _get_obs(self) -> np.ndarray:
        # Render the current frame to an RGB array (reuse cached surface)
        self._canvas.fill((0, 0, 0))

        # Draw Light
        pygame.draw.circle(
            self._canvas, (255, 255, 200), self.light_pos.astype(int), self.light_radius
        )

        # Draw Agent
        pygame.draw.circle(
            self._canvas, (0, 100, 255), self.agent_pos.astype(int), self.agent_radius
        )

        # Convert to numpy
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self._canvas)), axes=(1, 0, 2)
        )

    def _get_info(self) -> dict:
        dist = float(np.linalg.norm(self.agent_pos - self.light_pos))
        in_light = dist < (self.light_radius + self.agent_radius)
        return {
            "distance_to_light": dist,
            "in_light": in_light,
            "battery": self.battery,
        }

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            
        if self.window is None:
            return

        # Reuse drawing logic
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((10, 10, 10)) # Ambient Darkness
        
        # Draw Light (Gradient)
        pygame.draw.circle(canvas, (255, 255, 220), self.light_pos.astype(int), self.light_radius)
        
        # Draw Agent
        pygame.draw.circle(canvas, (50, 150, 255), self.agent_pos.astype(int), self.agent_radius)

        # Blit to window
        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
