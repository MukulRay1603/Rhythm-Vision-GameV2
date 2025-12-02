from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class FloatingText:
    text: str
    position: Tuple[int, int]
    ttl: float
    created_at: float
    color: Tuple[int, int, int]
    scale: float = 0.9


@dataclass
class BeatRipple:
    center: Tuple[int, int]
    radius: float
    max_radius: float
    thickness: int
    color: Tuple[int, int, int]


ICON_CACHE: Dict[str, np.ndarray] = {}


def _put_text_with_shadow(frame, text, org, font_scale, color, thickness: int = 2) -> None:
    x, y = org
    cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_xp_bar(frame, xp: int, max_xp: int, level: int, combo: int, prompt: str) -> None:
    h, w, _ = frame.shape
    bar_height = 36
    x0, y0 = 20, 16
    x1, y1 = w - 20, y0 + bar_height

    cv2.rectangle(frame, (x0, y0), (x1, y1), (20, 20, 30), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 110), 2)

    ratio = min(1.0, xp / max_xp) if max_xp > 0 else 0.0
    fill_w = int((x1 - x0 - 4) * ratio)
    grad_start = (255, 255, 0)
    grad_end = (0, 255, 255)
    if fill_w > 0:
        for i in range(fill_w):
            t = i / max(1, fill_w - 1)
            b = int(grad_start[0] * (1 - t) + grad_end[0] * t)
            g = int(grad_start[1] * (1 - t) + grad_end[1] * t)
            r = int(grad_start[2] * (1 - t) + grad_end[2] * t)
            cv2.line(frame, (x0 + 2 + i, y0 + 2), (x0 + 2 + i, y1 - 2), (b, g, r), 1)

    hud_text = f"LV {level} | XP {xp}/{max_xp} | COMBO {combo}"
    _put_text_with_shadow(frame, hud_text, (x0 + 12, y0 + 25), 0.6, (255, 255, 255))

    if prompt:
        text = f"{prompt}"
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
        px = x1 - size[0] - 12
        _put_text_with_shadow(frame, text, (px, y0 + 25), 0.7, (255, 215, 0))


def draw_floating_texts(frame, texts: List[FloatingText], now: float) -> List[FloatingText]:
    remaining: List[FloatingText] = []
    for ft in texts:
        age = now - ft.created_at
        if age > ft.ttl:
            continue
        x, y = ft.position
        alpha = max(0.0, 1.0 - age / ft.ttl)
        color = tuple(int(c * alpha) for c in ft.color)
        _put_text_with_shadow(frame, ft.text, (x, y), ft.scale, color)
        remaining.append(ft)
    return remaining


def draw_ripples(frame, ripples: List[BeatRipple]) -> List[BeatRipple]:
    updated: List[BeatRipple] = []
    for r in ripples:
        if r.radius < r.max_radius:
            cv2.circle(frame, r.center, int(r.radius), r.color, r.thickness)
            r.radius += 4.0
            updated.append(r)
    return updated


def draw_face_aura(frame, box, combo: int, ult: bool = False, show_zone: bool = False) -> None:
    x, y, w, h = map(int, box)
    if ult:
        color = (0, 255, 255)
        thickness = 5
    elif combo >= 10:
        color = (255, 0, 255)
        thickness = 4
    elif combo >= 5:
        color = (0, 180, 255)
        thickness = 3
    else:
        color = (0, 255, 170)
        thickness = 2

    cv2.rectangle(frame, (x - 3, y - 3), (x + w + 3, y + h + 3), (color[0] // 3, color[1] // 3, color[2] // 3), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    if show_zone:
        zone_top = max(0, y - int(h * 0.4))
        zone_bottom = min(frame.shape[0] - 1, y + int(h * 0.2))
        cv2.rectangle(frame, (x - 10, zone_top), (x + w + 10, zone_bottom), (60, 60, 120), 1)


def draw_prompt_lane(frame, prompt: str, phase: float = 0.0, beat_pop: float = 0.0) -> None:
    h, w, _ = frame.shape
    lane_h = int(h * 0.18)
    y0 = h - lane_h
    cv2.rectangle(frame, (0, y0), (w, h), (15, 15, 25), -1)

    cv2.rectangle(frame, (6, y0 + 6), (w - 6, h - 6), (90, 90, 140), 2)

    center = (w // 2, y0 + lane_h // 2 + 6)
    base_size = int(min(w, h) * 0.09)
    size = int(base_size * (1.0 + 0.08 * beat_pop))

    glow = int(120 + 60 * abs(np.sin(phase * 3.1415 * 2)))

    if prompt in ("MOVE LEFT", "MOVE RIGHT"):
        base_color = (glow, 255, 255) if prompt == "MOVE RIGHT" else (255, glow, 255)
        direction = "RIGHT" if prompt == "MOVE RIGHT" else "LEFT"
        _draw_arrow(frame, center, size // 2, direction, base_color)
    elif prompt in ("LEAN IN", "LEAN BACK"):
        base_color = (255, glow, 128)
        direction = "DOWN" if prompt == "LEAN IN" else "UP"
        _draw_arrow(frame, center, size // 2, direction, base_color)
    elif prompt in ("TURN LEFT", "TURN RIGHT"):
        base_color = (255, 128, 255)
        direction = "TURN_LEFT" if prompt == "TURN LEFT" else "TURN_RIGHT"
        _draw_turn_arrow(frame, center, size // 2, direction, base_color)
    elif prompt in ("TILT UP", "TILT DOWN"):
        base_color = (128, 255, 255)
        direction = "UP" if prompt == "TILT UP" else "DOWN"
        _draw_arrow(frame, center, size // 2, direction, base_color)
    else:
        base_color = (255, 200, 0)
        _draw_gesture_icon(frame, center, size // 2, prompt, base_color)


def _draw_arrow(frame, center, size: int, direction: str, color) -> None:
    cx, cy = center
    if direction == "LEFT":
        pts = np.array([[cx + size, cy - size], [cx - size, cy], [cx + size, cy + size]], dtype=np.int32)
    elif direction == "RIGHT":
        pts = np.array([[cx - size, cy - size], [cx + size, cy], [cx - size, cy + size]], dtype=np.int32)
    elif direction == "UP":
        pts = np.array([[cx - size, cy + size], [cx, cy - size], [cx + size, cy + size]], dtype=np.int32)
    else:
        pts = np.array([[cx - size, cy - size], [cx, cy + size], [cx + size, cy - size]], dtype=np.int32)
    cv2.fillConvexPoly(frame, pts, color)
    cv2.polylines(frame, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_turn_arrow(frame, center, size: int, direction: str, color) -> None:
    cx, cy = center
    radius = size
    axes = (radius, radius)
    startAngle = 45 if direction == "TURN_LEFT" else 135
    endAngle = 315 if direction == "TURN_LEFT" else 225
    cv2.ellipse(frame, (cx, cy), axes, 0, startAngle, endAngle, color, 4)
    if direction == "TURN_LEFT":
        tip = (cx - radius, cy)
        left = (tip[0] + 18, tip[1] - 10)
        right = (tip[0] + 18, tip[1] + 10)
    else:
        tip = (cx + radius, cy)
        left = (tip[0] - 18, tip[1] - 10)
        right = (tip[0] - 18, tip[1] + 10)
    cv2.fillConvexPoly(frame, np.array([tip, left, right], dtype=np.int32), color)


def _draw_gesture_icon(frame, center, size: int, prompt: str, color) -> None:
    cx, cy = center
    if prompt == "CLAP":
        w = size // 2
        h = size
        cv2.rectangle(frame, (cx - w - 12, cy - h // 2), (cx - 12, cy + h // 2), color, -1)
        cv2.rectangle(frame, (cx + 12, cy - h // 2), (cx + w + 12, cy + h // 2), color, -1)
    elif prompt == "WAVE":
        pts = np.array(
            [
                [cx - size, cy + size // 2],
                [cx - size // 2, cy - size // 2],
                [cx, cy + size // 2],
                [cx + size // 2, cy - size // 2],
                [cx + size, cy + size // 2],
            ],
            dtype=np.int32,
        )
        cv2.polylines(frame, [pts], False, color, 3)
    else:
        h = size
        w = size // 2
        cv2.rectangle(frame, (cx - w - 24, cy - h), (cx - 24, cy), color, -1)
        cv2.rectangle(frame, (cx + 24, cy - h), (cx + w + 24, cy), color, -1)
