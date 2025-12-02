from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class MotionState:
    prev_center: np.ndarray | None = None
    prev_size: float | None = None


def compute_motion(center: np.ndarray, size_scalar: float, state: MotionState) -> Tuple[float, float]:
    if state.prev_center is None:
        dx = 0.0
    else:
        dx = float(center[0] - state.prev_center[0])

    if state.prev_size is None:
        dz = 0.0
    else:
        dz = float(size_scalar - state.prev_size)

    state.prev_center = center.copy()
    state.prev_size = float(size_scalar)
    return dx, dz


def combo_multiplier(combo: int) -> int:
    if combo >= 20:
        return 5
    if combo >= 15:
        return 4
    if combo >= 10:
        return 3
    if combo >= 5:
        return 2
    return 1
