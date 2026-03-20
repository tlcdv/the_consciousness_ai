"""
Navigation Environment: multi-room grid with fog of war.

A 2x2 room grid connected by doorways. The agent collects colored goals
that respawn in random rooms. Fog of war limits visibility to the current
room, testing memory. A battery system adds time pressure.

Observation: RGB image [height, width, 3] uint8.
Action: [move_x, move_y] continuous in [-1, 1].
Info dict: current_room, goal_room, rooms_visited, goals_collected, battery.
"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np


# Room layout constants
_ROOM_GRID = 2  # 2x2 rooms
_DOORWAY_WIDTH = 40
_WALL_THICKNESS = 4


class NavigationEnv(gym.Env):
    """
    Multi-room navigation with fog of war, colored goals, and battery.

    Rooms are arranged on a 2x2 grid. Each room is separated by walls
    with doorways. The agent can only see the room it is currently in
    (fog of war). Goals are colored objects (green, blue, red) that give
    +1.0, +0.5, +0.2 reward respectively. Collected goals respawn in a
    random room. Battery drains over time; reaching 0 ends the episode.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    GOAL_COLORS = {
        "green": (0, 200, 0),
        "blue": (0, 100, 255),
        "red": (255, 50, 50),
    }
    GOAL_REWARDS = {"green": 1.0, "blue": 0.5, "red": 0.2}

    def __init__(
        self,
        render_mode: str | None = None,
        width: int = 224,
        height: int = 224,
        battery_drain: float = 0.002,
        max_goals: int = 3,
    ):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.battery_drain = battery_drain
        self.max_goals = max_goals

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 3), dtype=np.uint8
        )

        # Derived room dimensions
        self.room_w = width // _ROOM_GRID
        self.room_h = height // _ROOM_GRID

        # State (initialized in reset)
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.battery = 1.0
        self.goals: list[dict] = []
        self.goals_collected = 0
        self.rooms_visited: set[tuple[int, int]] = set()
        self._step_count = 0

    def _get_room(self, pos: np.ndarray) -> tuple[int, int]:
        """Return (row, col) room index for a position."""
        col = int(np.clip(pos[0] // self.room_w, 0, _ROOM_GRID - 1))
        row = int(np.clip(pos[1] // self.room_h, 0, _ROOM_GRID - 1))
        return (row, col)

    def _room_bounds(self, room: tuple[int, int]) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) for a room."""
        row, col = room
        x_min = col * self.room_w
        y_min = row * self.room_h
        return (x_min, y_min, x_min + self.room_w, y_min + self.room_h)

    def _is_in_doorway(self, pos: np.ndarray) -> bool:
        """Check if position is within a doorway between rooms."""
        x, y = pos
        half_door = _DOORWAY_WIDTH / 2
        mid_x = self.width / 2
        mid_y = self.height / 2

        # Vertical doorways (between left-right rooms)
        near_vertical_wall = abs(x - mid_x) < _WALL_THICKNESS * 2
        in_vertical_doorway = near_vertical_wall and abs(y % self.room_h - self.room_h / 2) < half_door

        # Horizontal doorways (between top-bottom rooms)
        near_horizontal_wall = abs(y - mid_y) < _WALL_THICKNESS * 2
        in_horizontal_doorway = near_horizontal_wall and abs(x % self.room_w - self.room_w / 2) < half_door

        return in_vertical_doorway or in_horizontal_doorway

    def _clamp_with_walls(self, old_pos: np.ndarray, new_pos: np.ndarray) -> np.ndarray:
        """Clamp position to stay within room boundaries unless in a doorway."""
        # Clamp to world bounds
        new_pos = np.clip(new_pos, [2, 2], [self.width - 2, self.height - 2])

        old_room = self._get_room(old_pos)
        new_room = self._get_room(new_pos)

        if old_room == new_room:
            return new_pos

        # Trying to cross rooms: only allow if passing through doorway
        if self._is_in_doorway(new_pos):
            return new_pos

        # Block: stay in old room
        return old_pos.copy()

    def _spawn_goal(self) -> dict:
        """Spawn a goal at a random position in a random room."""
        room = (self.np_random.integers(0, _ROOM_GRID), self.np_random.integers(0, _ROOM_GRID))
        bounds = self._room_bounds(room)
        margin = 15
        x = self.np_random.uniform(bounds[0] + margin, bounds[2] - margin)
        y = self.np_random.uniform(bounds[1] + margin, bounds[3] - margin)
        color_name = self.np_random.choice(list(self.GOAL_COLORS.keys()))
        return {
            "pos": np.array([x, y], dtype=np.float32),
            "color": color_name,
            "room": room,
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Start in room (0, 0)
        self.agent_pos = np.array([
            self.room_w / 2, self.room_h / 2
        ], dtype=np.float32)
        self.battery = 1.0
        self.goals_collected = 0
        self.rooms_visited = {(0, 0)}
        self._step_count = 0

        # Spawn goals
        self.goals = [self._spawn_goal() for _ in range(self.max_goals)]

        return self._render_frame(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0)
        speed = 4.0
        old_pos = self.agent_pos.copy()
        new_pos = self.agent_pos + action * speed
        self.agent_pos = self._clamp_with_walls(old_pos, new_pos)

        current_room = self._get_room(self.agent_pos)
        self.rooms_visited.add(current_room)

        # Battery
        self.battery = max(0.0, self.battery - self.battery_drain)
        self._step_count += 1

        # Goal collection
        reward = -0.01  # Small step penalty
        collected_indices = []
        for i, goal in enumerate(self.goals):
            dist = np.linalg.norm(self.agent_pos - goal["pos"])
            if dist < 15.0:
                reward += self.GOAL_REWARDS[goal["color"]]
                self.goals_collected += 1
                collected_indices.append(i)

        # Respawn collected goals
        for i in sorted(collected_indices, reverse=True):
            self.goals[i] = self._spawn_goal()

        terminated = self.battery <= 0.0
        truncated = False

        return self._render_frame(), reward, terminated, truncated, self._get_info()

    def _render_frame(self) -> np.ndarray:
        """Render the current state as an RGB image with fog of war."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        current_room = self._get_room(self.agent_pos)
        bounds = self._room_bounds(current_room)

        # Room floor (visible room only)
        x_min, y_min, x_max, y_max = [int(v) for v in bounds]
        brightness = int(40 + 60 * self.battery)
        frame[y_min:y_max, x_min:x_max] = [brightness, brightness, brightness + 10]

        # Walls
        mid_x = self.width // 2
        mid_y = self.height // 2
        frame[mid_y - _WALL_THICKNESS // 2:mid_y + _WALL_THICKNESS // 2, :] = [120, 100, 80]
        frame[:, mid_x - _WALL_THICKNESS // 2:mid_x + _WALL_THICKNESS // 2] = [120, 100, 80]

        # Doorways (clear wall at doorway locations)
        half_door = _DOORWAY_WIDTH // 2
        # Horizontal wall doorways
        for col in range(_ROOM_GRID):
            cx = col * self.room_w + self.room_w // 2
            frame[mid_y - _WALL_THICKNESS:mid_y + _WALL_THICKNESS,
                  max(0, cx - half_door):min(self.width, cx + half_door)] = [brightness, brightness, brightness + 10]
        # Vertical wall doorways
        for row in range(_ROOM_GRID):
            cy = row * self.room_h + self.room_h // 2
            frame[max(0, cy - half_door):min(self.height, cy + half_door),
                  mid_x - _WALL_THICKNESS:mid_x + _WALL_THICKNESS] = [brightness, brightness, brightness + 10]

        # Goals (only visible in current room)
        for goal in self.goals:
            if goal["room"] == current_room:
                gx, gy = int(goal["pos"][0]), int(goal["pos"][1])
                r = 8
                y_lo, y_hi = max(0, gy - r), min(self.height, gy + r)
                x_lo, x_hi = max(0, gx - r), min(self.width, gx + r)
                frame[y_lo:y_hi, x_lo:x_hi] = self.GOAL_COLORS[goal["color"]]

        # Agent
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        ar = 6
        y_lo, y_hi = max(0, ay - ar), min(self.height, ay + ar)
        x_lo, x_hi = max(0, ax - ar), min(self.width, ax + ar)
        frame[y_lo:y_hi, x_lo:x_hi] = [255, 255, 255]

        return frame

    def _get_info(self) -> dict:
        return {
            "current_room": self._get_room(self.agent_pos),
            "goal_rooms": [g["room"] for g in self.goals],
            "rooms_visited": len(self.rooms_visited),
            "goals_collected": self.goals_collected,
            "battery": self.battery,
            "step": self._step_count,
        }
