from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
from typing import Any

from simulations.environments.audio_mixin import DarkRoomAudioMixin

LIGHT_COLOUR = (255, 255, 200)
AGENT_COLOUR = (0, 100, 255)
WALL_COLOUR = (60, 60, 60)  # outside the room, visible in the agent-centered view
# Half-width of the agent-centered window, in room pixels. Set 2026-09-15 before any run:
# a 96-pixel window in a 224-pixel room, so a far light is out of view.
DEFAULT_VIEW_RADIUS = 48


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

    def __init__(self, render_mode: str | None = None, width: int = 512, height: int = 512,
                 audio: str = "legacy", audio_channels: int | None = None,
                 report_collision: bool = False, view: str = "full",
                 view_radius: int = DEFAULT_VIEW_RADIUS):
        """`audio`: "legacy" (mono events, the behaviour of every run before
        2026-09-15) or "binaural" (the light emits a tone heard across the room,
        with direction; `audio_channels` 2 = left/right, 4 = adds upper/lower).
        `report_collision`: add info["collision"] when a move is stopped by a wall.
        `view`: "full" (the whole room from above) or "agent_centered" (a window of
        half-width `view_radius` around the agent, scaled to the frame size; tectal
        maps are egocentric). The defaults reproduce the earlier environment exactly."""
        if view not in ("full", "agent_centered"):
            raise ValueError("view must be 'full' or 'agent_centered', got %r" % view)
        self.view = view
        self.view_radius = int(view_radius)
        if audio not in ("legacy", "binaural"):
            raise ValueError("audio must be 'legacy' or 'binaural', got %r" % audio)
        if audio == "binaural" and audio_channels not in (2, 4):
            raise ValueError("binaural audio needs audio_channels 2 or 4, got %r" % audio_channels)
        self.audio_mode = audio
        self.audio_channels = audio_channels
        self.report_collision = report_collision
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
        if self.view == "agent_centered":
            pad = 2 * self.view_radius
            self._padded_canvas = pygame.Surface((self.width + pad, self.height + pad))
        
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
        unclipped = self.agent_pos.copy()
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

        if self.view == "agent_centered":
            # Ground truth for analysis only; no agent code reads keys starting '_truth_'.
            reach = self.view_radius + self.light_radius
            gap = np.abs(self.light_pos - self.agent_pos)
            info["_truth_light_in_view"] = bool(gap[0] <= reach and gap[1] <= reach)

        if self.report_collision:
            # Touch, not damage: no homeostatic variable changes (ethics rule E6).
            info["collision"] = bool(np.any(unclipped != self.agent_pos))

        # Generate audio waveform from environment state (DarkRoomAudioMixin)
        if self.audio_mode == "binaural":
            # Ground truth for analysis only; no agent code reads keys starting '_truth_'.
            offset = self.light_pos - self.agent_pos
            info["_truth_light_offset"] = [float(offset[0]), float(offset[1])]
            info["audio_waveform"] = self._generate_binaural_audio(info, self.audio_channels)
        else:
            info["audio_waveform"] = self._generate_audio(info)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info
        
    def _get_obs(self) -> np.ndarray:
        if self.view == "agent_centered":
            return self._agent_centered_obs()
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

    def _agent_centered_obs(self) -> np.ndarray:
        """A window around the agent, scaled to the frame. Nearest-pixel scaling keeps
        exact colours. Outside the room is WALL_COLOUR."""
        r = self.view_radius
        canvas = self._padded_canvas
        canvas.fill(WALL_COLOUR)
        pygame.draw.rect(canvas, (0, 0, 0), (r, r, self.width, self.height))
        pygame.draw.circle(canvas, LIGHT_COLOUR, (self.light_pos + r).astype(int), self.light_radius)
        pygame.draw.circle(canvas, AGENT_COLOUR, (self.agent_pos + r).astype(int), self.agent_radius)
        left, top = self.agent_pos.astype(int)
        window = canvas.subsurface((left, top, 2 * r, 2 * r))
        scaled = pygame.transform.scale(window, (self.width, self.height))
        return np.transpose(np.array(pygame.surfarray.pixels3d(scaled)), axes=(1, 0, 2))

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
