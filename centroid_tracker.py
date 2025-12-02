from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np


def smooth(prev: np.ndarray | None, new: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    if prev is None:
        return new.astype(float)
    return alpha * new + (1.0 - alpha) * prev


class CentroidTracker:
    def __init__(self, max_lost: int = 15, trail_size: int = 60) -> None:
        self.next_id: int = 0
        self.boxes: Dict[int, np.ndarray] = {}
        self.lost: Dict[int, int] = {}
        self.trails: Dict[int, deque[np.ndarray]] = {}
        self.smoothed_centroids: Dict[int, np.ndarray] = {}
        self.max_lost: int = max_lost
        self.trail_size: int = trail_size

    @staticmethod
    def _centroid(box: np.ndarray) -> np.ndarray:
        x, y, w, h = box
        return np.array([x + w / 2.0, y + h / 2.0], dtype=float)

    def _register(self, box: np.ndarray) -> None:
        tid = self.next_id
        self.next_id += 1

        self.boxes[tid] = box.astype(float)
        self.lost[tid] = 0
        self.trails[tid] = deque(maxlen=self.trail_size)

        c = self._centroid(box)
        self.trails[tid].append(c)
        self.smoothed_centroids[tid] = c.copy()

    def _deregister(self, tid: int) -> None:
        self.boxes.pop(tid, None)
        self.lost.pop(tid, None)
        self.trails.pop(tid, None)
        self.smoothed_centroids.pop(tid, None)

    def _cleanup(self) -> None:
        to_delete = [tid for tid, lost in self.lost.items() if lost > self.max_lost]
        for tid in to_delete:
            self._deregister(tid)

    def update(self, detections: List[np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, deque]]:
        if len(self.boxes) == 0:
            for box in detections:
                self._register(box)
            return self.boxes, self.trails

        ids = list(self.boxes.keys())
        old_centroids = [self._centroid(self.boxes[i]) for i in ids]

        if len(detections) == 0:
            for tid in ids:
                self.lost[tid] += 1
            self._cleanup()
            return self.boxes, self.trails

        new_centroids = [self._centroid(b) for b in detections]
        dist = np.zeros((len(old_centroids), len(new_centroids)), dtype=float)
        for r, oc in enumerate(old_centroids):
            for c, nc in enumerate(new_centroids):
                dist[r, c] = np.linalg.norm(oc - nc)

        used_rows, used_cols = set(), set()

        while True:
            r, c = divmod(int(dist.argmin()), dist.shape[1])
            if dist[r, c] == np.inf or dist[r, c] > 80:
                break

            tid = ids[r]
            box = detections[c].astype(float)
            self.boxes[tid] = box
            self.lost[tid] = 0

            centroid = new_centroids[c]
            self.trails[tid].append(centroid)
            self.smoothed_centroids[tid] = smooth(self.smoothed_centroids.get(tid), centroid)

            used_rows.add(r)
            used_cols.add(c)
            dist[r, :] = np.inf
            dist[:, c] = np.inf

        for c, box in enumerate(detections):
            if c not in used_cols:
                self._register(box)

        for r, tid in enumerate(ids):
            if r not in used_rows:
                self.lost[tid] += 1

        self._cleanup()
        return self.boxes, self.trails
