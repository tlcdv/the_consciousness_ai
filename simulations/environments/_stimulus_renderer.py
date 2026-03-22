"""
Shared stimulus rendering utilities for cognitive task environments.

Provides polygon shape drawing, color palettes, and card rendering used
by DMTS and WCST environments. All rendering is pure numpy (no pygame).
"""
from __future__ import annotations

import numpy as np
import math


# ── Palette ──────────────────────────────────────────────────────────────

COLORS = {
    "red":     (220,  50,  50),
    "blue":    ( 50, 100, 220),
    "green":   ( 50, 200,  50),
    "yellow":  (220, 220,  50),
    "magenta": (200,  50, 200),
    "cyan":    ( 50, 200, 200),
}

COLOR_NAMES = list(COLORS.keys())

SHAPE_NAMES = ["triangle", "square", "pentagon", "hexagon", "circle", "star"]

BACKGROUND_GRAY = 128
CARD_BORDER_COLOR = (220, 220, 220)
FEEDBACK_GREEN = (50, 200, 50)
FEEDBACK_RED = (220, 50, 50)


# ── Polygon vertices ────────────────────────────────────────────────────

def _regular_polygon(n: int, radius: float, cx: float, cy: float) -> list[tuple[float, float]]:
    """Return vertices of a regular n-gon centered at (cx, cy)."""
    angle_offset = -math.pi / 2  # point up
    return [
        (cx + radius * math.cos(2 * math.pi * i / n + angle_offset),
         cy + radius * math.sin(2 * math.pi * i / n + angle_offset))
        for i in range(n)
    ]


def _star_vertices(radius: float, cx: float, cy: float,
                   points: int = 5, inner_ratio: float = 0.4) -> list[tuple[float, float]]:
    """Return vertices of a star polygon."""
    verts = []
    angle_offset = -math.pi / 2
    for i in range(points * 2):
        r = radius if i % 2 == 0 else radius * inner_ratio
        angle = math.pi * i / points + angle_offset
        verts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return verts


def get_shape_vertices(shape: str, radius: float, cx: float, cy: float) -> list[tuple[float, float]]:
    """Get vertices for a named shape."""
    if shape == "triangle":
        return _regular_polygon(3, radius, cx, cy)
    elif shape == "square":
        return _regular_polygon(4, radius, cx, cy)
    elif shape == "pentagon":
        return _regular_polygon(5, radius, cx, cy)
    elif shape == "hexagon":
        return _regular_polygon(6, radius, cx, cy)
    elif shape == "circle":
        return _regular_polygon(32, radius, cx, cy)
    elif shape == "star":
        return _star_vertices(radius, cx, cy)
    else:
        raise ValueError(f"Unknown shape: {shape}")


# ── Rasterization ───────────────────────────────────────────────────────

def _fill_polygon(canvas: np.ndarray, vertices: list[tuple[float, float]],
                  color: tuple[int, int, int]) -> None:
    """Fill a convex-ish polygon on an RGB canvas using scanline."""
    if len(vertices) < 3:
        return
    h, w = canvas.shape[:2]
    ys = [v[1] for v in vertices]
    y_min = max(0, int(min(ys)))
    y_max = min(h - 1, int(max(ys)))

    for y in range(y_min, y_max + 1):
        intersections = []
        n = len(vertices)
        for i in range(n):
            x0, y0 = vertices[i]
            x1, y1 = vertices[(i + 1) % n]
            if y0 == y1:
                continue
            if min(y0, y1) <= y < max(y0, y1):
                x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                intersections.append(x_int)
        intersections.sort()
        for j in range(0, len(intersections) - 1, 2):
            x_start = max(0, int(intersections[j]))
            x_end = min(w - 1, int(intersections[j + 1]))
            canvas[y, x_start:x_end + 1] = color


def draw_shape(canvas: np.ndarray, shape: str, color: tuple[int, int, int],
               cx: float, cy: float, radius: float) -> None:
    """Draw a filled shape on the canvas."""
    verts = get_shape_vertices(shape, radius, cx, cy)
    _fill_polygon(canvas, verts, color)


def draw_rect(canvas: np.ndarray, x: int, y: int, w: int, h: int,
              color: tuple[int, int, int], thickness: int = 1) -> None:
    """Draw a rectangle outline."""
    ch, cw = canvas.shape[:2]
    # Top
    canvas[max(0, y):min(ch, y + thickness), max(0, x):min(cw, x + w)] = color
    # Bottom
    canvas[max(0, y + h - thickness):min(ch, y + h), max(0, x):min(cw, x + w)] = color
    # Left
    canvas[max(0, y):min(ch, y + h), max(0, x):min(cw, x + thickness)] = color
    # Right
    canvas[max(0, y):min(ch, y + h), max(0, x + w - thickness):min(cw, x + w)] = color


def draw_filled_rect(canvas: np.ndarray, x: int, y: int, w: int, h: int,
                     color: tuple[int, int, int]) -> None:
    """Draw a filled rectangle."""
    ch, cw = canvas.shape[:2]
    canvas[max(0, y):min(ch, y + h), max(0, x):min(cw, x + w)] = color


def draw_cross(canvas: np.ndarray, cx: int, cy: int, size: int,
               color: tuple[int, int, int], thickness: int = 2) -> None:
    """Draw a fixation cross."""
    h, w = canvas.shape[:2]
    half = size // 2
    t = thickness // 2
    # Horizontal
    canvas[max(0, cy - t):min(h, cy + t + 1),
           max(0, cx - half):min(w, cx + half + 1)] = color
    # Vertical
    canvas[max(0, cy - half):min(h, cy + half + 1),
           max(0, cx - t):min(w, cx + t + 1)] = color


def draw_card(canvas: np.ndarray, x: int, y: int, card_w: int, card_h: int,
              shape: str, color_name: str, count: int = 1,
              radius: float | None = None) -> None:
    """Draw a card with border, containing count copies of a colored shape.

    Layout for count:
      1: center
      2: side by side
      3: triangle arrangement
      4: 2x2 grid
    """
    # White border
    draw_rect(canvas, x, y, card_w, card_h, CARD_BORDER_COLOR, thickness=2)
    # Dark interior
    draw_filled_rect(canvas, x + 2, y + 2, card_w - 4, card_h - 4, (40, 40, 40))

    rgb = COLORS[color_name]
    if radius is None:
        radius = min(card_w, card_h) * 0.25

    # Compute centers for each shape copy
    inner_cx = x + card_w / 2
    inner_cy = y + card_h / 2
    spacing = radius * 1.4

    if count == 1:
        centers = [(inner_cx, inner_cy)]
    elif count == 2:
        centers = [(inner_cx - spacing * 0.6, inner_cy),
                   (inner_cx + spacing * 0.6, inner_cy)]
    elif count == 3:
        centers = [(inner_cx, inner_cy - spacing * 0.5),
                   (inner_cx - spacing * 0.6, inner_cy + spacing * 0.4),
                   (inner_cx + spacing * 0.6, inner_cy + spacing * 0.4)]
    elif count >= 4:
        centers = [(inner_cx - spacing * 0.5, inner_cy - spacing * 0.5),
                   (inner_cx + spacing * 0.5, inner_cy - spacing * 0.5),
                   (inner_cx - spacing * 0.5, inner_cy + spacing * 0.5),
                   (inner_cx + spacing * 0.5, inner_cy + spacing * 0.5)]
    else:
        centers = [(inner_cx, inner_cy)]

    small_r = radius * 0.6 if count > 1 else radius
    for cx, cy in centers:
        draw_shape(canvas, shape, rgb, cx, cy, small_r)
