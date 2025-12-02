from __future__ import annotations

from collections import deque
from typing import Dict, Tuple, List

import cv2
import numpy as np


def _create_mosse_like_tracker():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
        return cv2.legacy.TrackerMOSSE_create()
    if hasattr(cv2, "TrackerMOSSE_create"):
        return cv2.TrackerMOSSE_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()

    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMIL_create"):
        return cv2.legacy.TrackerMIL_create()
    if hasattr(cv2, "TrackerMIL_create"):
        return cv2.TrackerMIL_create()

    raise RuntimeError("No suitable OpenCV tracker (MOSSE/KCF/MIL) is available in this OpenCV build.")


class CorrelationTrackerManager:
    def __init__(self, max_lost: int = 15, trail_size: int = 60) -> None:
        self.next_id: int = 0
        self.trackers: Dict[int, cv2.Tracker] = {}
        self.boxes: Dict[int, np.ndarray] = {}
        self.trails: Dict[int, deque[np.ndarray]] = {}
        self.lost: Dict[int, int] = {}
        self.max_lost: int = max_lost
        self.trail_size: int = trail_size

    def _register(self, frame: np.ndarray, box: np.ndarray) -> None:
        tracker = _create_mosse_like_tracker()
        x, y, w, h = box
        bbox = (float(x), float(y), float(w), float(h))
        ok = tracker.init(frame, bbox)
        if not ok:
            return
        tid = self.next_id
        self.next_id += 1

        self.trackers[tid] = tracker
        self.boxes[tid] = np.array([float(x), float(y), float(w), float(h)], dtype=float)
        self.lost[tid] = 0
        self.trails[tid] = deque(maxlen=self.trail_size)

        cx, cy = x + w / 2.0, y + h / 2.0
        self.trails[tid].append(np.array([cx, cy], dtype=float))

    def _deregister(self, tid: int) -> None:
        self.trackers.pop(tid, None)
        self.boxes.pop(tid, None)
        self.trails.pop(tid, None)
        self.lost.pop(tid, None)

    def _cleanup(self) -> None:
        to_delete = [tid for tid, l in self.lost.items() if l > self.max_lost]
        for tid in to_delete:
            self._deregister(tid)

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1, inter_y1 = max(ax, bx), max(ay, by)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = aw * ah + bw * bh - inter_area
        if union_area <= 0:
            return 0.0
        return float(inter_area / union_area)

    def update_with_detections(self, frame: np.ndarray, detections: List[np.ndarray]):
        if len(self.trackers) == 0:
            for box in detections:
                self._register(frame, box)
            return self.boxes, self.trails

        ids = list(self.trackers.keys())
        existing_boxes = [self.boxes[i] for i in ids]

        if len(detections) == 0:
            for tid in ids:
                self.lost[tid] += 1
            self._cleanup()
            return self.boxes, self.trails

        iou_mat = np.zeros((len(existing_boxes), len(detections)), dtype=float)
        for r, eb in enumerate(existing_boxes):
            for c, db in enumerate(detections):
                iou_mat[r, c] = self._iou(eb, db)

        used_rows, used_cols = set(), set()

        while True:
            flat_idx = int(iou_mat.argmax())
            r, c = divmod(flat_idx, iou_mat.shape[1])
            if iou_mat[r, c] < 0.2:
                break

            tid = ids[r]
            box = detections[c].astype(float)
            x, y, w, h = box
            bbox = (float(x), float(y), float(w), float(h))

            tracker = _create_mosse_like_tracker()
            ok = tracker.init(frame, bbox)
            if ok:
                self.trackers[tid] = tracker
                self.boxes[tid] = np.array([float(x), float(y), float(w), float(h)], dtype=float)
                self.lost[tid] = 0

                cx, cy = x + w / 2.0, y + h / 2.0
                self.trails[tid].append(np.array([cx, cy], dtype=float))
            else:
                self.lost[tid] += 1

            used_rows.add(r)
            used_cols.add(c)
            iou_mat[r, :] = -1
            iou_mat[:, c] = -1

        for c, box in enumerate(detections):
            if c not in used_cols:
                self._register(frame, box)

        for r, tid in enumerate(ids):
            if r not in used_rows:
                self.lost[tid] += 1

        self._cleanup()
        return self.boxes, self.trails

    def predict_only(self, frame: np.ndarray):
        to_delete = []
        for tid, tracker in list(self.trackers.items()):
            ok, new_box = tracker.update(frame)
            if not ok:
                self.lost[tid] += 1
                if self.lost[tid] > self.max_lost:
                    to_delete.append(tid)
                continue

            x, y, w, h = new_box
            self.boxes[tid] = np.array([float(x), float(y), float(w), float(h)], dtype=float)
            x, y, w, h = self.boxes[tid]
            cx, cy = x + w / 2.0, y + h / 2.0
            self.trails[tid].append(np.array([cx, cy], dtype=float))
            self.lost[tid] = 0

        for tid in to_delete:
            self._deregister(tid)

        self._cleanup()
        return self.boxes, self.trails
