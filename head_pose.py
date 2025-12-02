from __future__ import annotations

from typing import Tuple

import numpy as np
from mediapipe.framework.formats import detection_pb2


def estimate_head_angles(
    detection: detection_pb2.Detection,
    image_width: int,
    image_height: int,
) -> Tuple[float, float]:
    kps = detection.location_data.relative_keypoints
    if kps is None or len(kps) < 4:
        return 0.0, 0.0

    right_eye = kps[0]
    left_eye = kps[1]
    nose_tip = kps[2]
    mouth = kps[3]

    re = np.array([right_eye.x * image_width, right_eye.y * image_height])
    le = np.array([left_eye.x * image_width, left_eye.y * image_height])
    nose = np.array([nose_tip.x * image_width, nose_tip.y * image_height])
    mouth = np.array([mouth.x * image_width, mouth.y * image_height])

    eye_center = 0.5 * (re + le)

    eye_vec = le - re
    eye_width = np.linalg.norm(eye_vec) + 1e-5
    proj = (nose - eye_center)[0]
    yaw_norm = proj / eye_width
    yaw_deg = float(40.0 * yaw_norm)

    up_vec = nose[1] - eye_center[1]
    down_vec = mouth[1] - nose[1]
    denom = (abs(up_vec) + abs(down_vec) + 1e-5)
    pitch_norm = (down_vec - up_vec) / denom
    pitch_deg = float(35.0 * pitch_norm)

    return yaw_deg, pitch_deg
