from __future__ import annotations

import csv
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from centroid_tracker import CentroidTracker
from correlation_tracker import CorrelationTrackerManager
from effects import (
    BeatRipple,
    FloatingText,
    draw_face_aura,
    draw_floating_texts,
    draw_prompt_lane,
    draw_ripples,
    draw_xp_bar,
)
from head_pose import estimate_head_angles
from utils import MotionState, combo_multiplier, compute_motion

# ----------------- CONFIG -----------------

USE_CORRELATION_TRACKER = True
DETECTION_INTERVAL = 5

LOGGING_ENABLED = True
SAVE_VIDEO = True
ENABLE_HAND_GESTURES = True

BPM = 120.0
BEAT_INTERVAL = 60.0 / BPM
PROMPT_BEATS = 4  # beats before auto-changing prompt if idle

MASTER_VOLUME = 0.8
BGM_VOLUME = 0.45
BGM_ULT_VOLUME = 0.85
SFX_VOLUME = 0.9
VOICE_VOLUME = 0.9

# Weighted prompt pool for better gameplay feel
WEIGHTED_PROMPTS = [
    ("MOVE LEFT", 10),
    ("MOVE RIGHT", 10),
    ("LEAN IN", 10),
    ("LEAN BACK", 10),
    ("WAVE", 4),
    ("HANDS UP", 3),
    ("CLAP", 3),
    ("TURN LEFT", 1),
    ("TURN RIGHT", 1),
    ("TILT UP", 1),
    ("TILT DOWN", 1),
]

PROMPT_LIST: List[str] = []
for _prompt, _w in WEIGHTED_PROMPTS:
    PROMPT_LIST.extend([_prompt] * _w)


def choose_weighted_prompt() -> str:
    return random.choice(PROMPT_LIST)


# ----------------- VIDEO -----------------


def init_video_capture() -> Tuple[cv2.VideoCapture, int, int]:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, width, height


# ----------------- AUDIO -----------------


class AudioManager:
    def __init__(self) -> None:
        self.ok = False
        self._init_pygame()

        self.music_channel = None
        self.ult_channel = None
        self.voice_channel = None
        self.sfx_channel = None

        self.bgm_intro = None
        self.bgm_loop = None
        self.bgm_ult = None

        self.sfx: Dict[str, object] = {}
        self.voice: Dict[str, object] = {}

        self.voice_cooldown: float = 0.25
        self.last_voice_time: float = 0.0

        self.music_mode: str = "idle"
        self.intro_end_time: float = 0.0
        self.ult_mode: bool = False

        if self.ok:
            self._setup_channels()
            self._load_all_sounds()

    def _init_pygame(self) -> None:
        try:
            import pygame

            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
            self.ok = True
        except Exception as e:
            print("[Audio] Failed to init pygame.mixer:", e)
            self.ok = False

    def _setup_channels(self) -> None:
        import pygame

        pygame.mixer.set_num_channels(8)
        self.music_channel = pygame.mixer.Channel(0)
        self.ult_channel = pygame.mixer.Channel(1)
        self.sfx_channel = pygame.mixer.Channel(2)
        self.voice_channel = pygame.mixer.Channel(3)

        self.music_channel.set_volume(BGM_VOLUME * MASTER_VOLUME)
        self.ult_channel.set_volume(BGM_ULT_VOLUME * MASTER_VOLUME)
        self.sfx_channel.set_volume(SFX_VOLUME * MASTER_VOLUME)
        self.voice_channel.set_volume(VOICE_VOLUME * MASTER_VOLUME)

    def _load_sound(self, base_name: str):
        if not self.ok:
            return None
        import pygame

        exts = [".wav", ".mp3", ".ogg"]
        for ext in exts:
            path = os.path.join("assets", "sfx", base_name + ext)
            if os.path.exists(path):
                try:
                    snd = pygame.mixer.Sound(path)
                    return snd
                except Exception as e:
                    print(f"[Audio] Failed to load {path}: {e}")
        return None

    def _load_all_sounds(self) -> None:
        # layered bgm: intro -> loop, optional ult
        self.bgm_intro = self._load_sound("bgm_intro")
        self.bgm_loop = self._load_sound("bgm_loop")
        self.bgm_ult = self._load_sound("bgm_ult")

        # core sfx
        for name in ["sfx_perfect", "sfx_miss", "sfx_ult", "sfx_combo_break"]:
            snd = self._load_sound(name)
            if snd is not None:
                self.sfx[name] = snd

        # voice prompts
        voice_keys = [
            "voice_move_left",
            "voice_move_right",
            "voice_lean_in",
            "voice_lean_back",
            "voice_turn_left",
            "voice_turn_right",
            "voice_tilt_up",
            "voice_tilt_down",
            "voice_wave",
            "voice_clap",
            "voice_hands_up",
        ]
        for key in voice_keys:
            snd = self._load_sound(key)
            if snd is not None:
                self.voice[key] = snd

    def maybe_start_bgm(self, now: float) -> None:
        if not self.ok or self.music_mode != "idle":
            return
        if self.bgm_intro is not None:
            self.music_channel.play(self.bgm_intro)
            self.intro_end_time = now + float(self.bgm_intro.get_length())
            self.music_mode = "intro"
        elif self.bgm_loop is not None:
            self.music_channel.play(self.bgm_loop, loops=-1)
            self.music_channel.set_volume(BGM_VOLUME * MASTER_VOLUME)
            self.music_mode = "loop"

    def update(self, now: float, ult_active: bool) -> None:
        if not self.ok:
            return

        # transition intro -> loop
        if self.music_mode == "intro" and self.bgm_loop is not None:
            if now >= self.intro_end_time or not self.music_channel.get_busy():
                self.music_channel.play(self.bgm_loop, loops=-1)
                self.music_channel.set_volume(BGM_VOLUME * MASTER_VOLUME)
                self.music_mode = "loop"

        # ult layering
        if ult_active and not self.ult_mode:
            self._enter_ult_mode()
        elif not ult_active and self.ult_mode:
            self._exit_ult_mode()

    def _enter_ult_mode(self) -> None:
        if not self.ok:
            return
        self.ult_mode = True
        self.music_channel.set_volume(BGM_VOLUME * MASTER_VOLUME * 0.4)
        if self.bgm_ult is not None:
            self.ult_channel.play(self.bgm_ult, loops=-1)
            self.ult_channel.set_volume(BGM_ULT_VOLUME * MASTER_VOLUME)

    def _exit_ult_mode(self) -> None:
        if not self.ok:
            return
        self.ult_mode = False
        self.music_channel.set_volume(BGM_VOLUME * MASTER_VOLUME)
        if self.ult_channel is not None:
            self.ult_channel.stop()

    def play_sfx(self, name: str) -> None:
        if not self.ok or self.sfx_channel is None:
            return
        snd = self.sfx.get(name)
        if snd is None:
            return
        self.sfx_channel.play(snd)

    def play_voice_for_prompt(self, prompt: str, now: float) -> None:
        if not self.ok or self.voice_channel is None:
            return
        if now - self.last_voice_time < self.voice_cooldown:
            return

        mapping = {
            "MOVE LEFT": "voice_move_left",
            "MOVE RIGHT": "voice_move_right",
            "LEAN IN": "voice_lean_in",
            "LEAN BACK": "voice_lean_back",
            "TURN LEFT": "voice_turn_left",
            "TURN RIGHT": "voice_turn_right",
            "TILT UP": "voice_tilt_up",
            "TILT DOWN": "voice_tilt_down",
            "WAVE": "voice_wave",
            "CLAP": "voice_clap",
            "HANDS UP": "voice_hands_up",
        }
        key = mapping.get(prompt)
        if key is None:
            return
        snd = self.voice.get(key)
        if snd is None:
            return

        self.voice_channel.play(snd)
        self.voice_channel.set_volume(VOICE_VOLUME * MASTER_VOLUME)
        self.last_voice_time = now


# ----------------- MAIN GAME LOOP -----------------


def main() -> None:
    mp_face = mp.solutions.face_detection
    mp_hands = mp.solutions.hands if ENABLE_HAND_GESTURES else None

    cap, W, H = init_video_capture()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    centroid_tracker = CentroidTracker(max_lost=15, trail_size=60)
    corr_tracker = CorrelationTrackerManager(max_lost=15, trail_size=60) if USE_CORRELATION_TRACKER else None

    motion_states: Dict[int, MotionState] = defaultdict(MotionState)

    xp: int = 0
    max_xp: int = 200
    level: int = 1
    combos: Dict[int, int] = defaultdict(int)
    ult_active: bool = False
    ult_end_time: float = 0.0

    current_prompt: str = choose_weighted_prompt()
    prompt_started: float = time.time()

    floating_texts: List[FloatingText] = []
    beat_ripples: List[BeatRipple] = []

    frame_idx = 0
    phase = 0.0

    if LOGGING_ENABLED:
        logfp = open("face_logs.csv", "w", newline="")
        writer = csv.writer(logfp)
        writer.writerow(
            [
                "timestamp_ms",
                "frame_idx",
                "face_id",
                "tracker_type",
                "x",
                "y",
                "w",
                "h",
                "smooth_cx",
                "smooth_cy",
                "dx",
                "dz",
                "prompt",
                "action_success",
                "reaction_time_ms",
            ]
        )
    else:
        logfp = None
        writer = None

    hand_state = {
        "last_wave_time": 0.0,
        "last_clap_time": 0.0,
        "last_handsup_time": 0.0,
        "gesture_cooldown_until": 0.0,
        "hand_history": [],
    }

    beat_timer = time.time()
    beats_since_prompt = 0
    beat_pop = 0.0

    audio_mgr = AudioManager()

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6) as fd:
        if mp_hands:
            hands_ctx = mp_hands.Hands(
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            hands_ctx = None

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_idx += 1
                now = time.time()

                # audio engine
                audio_mgr.maybe_start_bgm(now)
                audio_mgr.update(now, ult_active)

                # beat timing
                dt = now - beat_timer
                if dt >= BEAT_INTERVAL:
                    beats = int(dt // BEAT_INTERVAL)
                    beat_timer += beats * BEAT_INTERVAL
                    beats_since_prompt += beats
                    beat_pop = 1.0
                else:
                    beat_pop = max(0.0, beat_pop - 0.15)

                phase = (phase + 0.03) % 1.0

                # mirror camera for natural UX
                frame = cv2.flip(frame, 1)

                if out is None and SAVE_VIDEO:
                    out = cv2.VideoWriter("output_gameplay.mp4", fourcc, 20.0, (W, H))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # keep the prompt stable for the entire frame
                prompt_for_frame = current_prompt

                detections: List[np.ndarray] = []
                raw_detections = []
                run_detection_this_frame = (not USE_CORRELATION_TRACKER) or frame_idx % DETECTION_INTERVAL == 0

                if run_detection_this_frame:
                    results = fd.process(frame_rgb)
                    if results.detections:
                        for det in results.detections:
                            raw_detections.append(det)
                            bbox = det.location_data.relative_bounding_box
                            x = int(bbox.xmin * W)
                            y = int(bbox.ymin * H)
                            w = int(bbox.width * W)
                            h = int(bbox.height * H)
                            x = max(0, min(x, W - 1))
                            y = max(0, min(y, H - 1))
                            w = max(1, min(w, W - x))
                            h = max(1, min(h, H - y))
                            # Anti-ghost filtering: ignore tiny or far-away faces
                            if w < 80 or h < 80:
                                continue
                            if x < 40 or x + w > W - 40:
                                continue
                            detections.append(np.array([x, y, w, h], dtype=float))


                # Keep only the largest detected face to avoid ghost/background boxes
                if len(detections) > 1:
                    detections = [max(detections, key=lambda b: b[2] * b[3])]

                if USE_CORRELATION_TRACKER and corr_tracker is not None:
                    if run_detection_this_frame:
                        boxes, trails = corr_tracker.update_with_detections(frame, detections)
                    else:
                        boxes, trails = corr_tracker.predict_only(frame)
                    tracker_type = "correlation"
                else:
                    boxes, trails = centroid_tracker.update(detections)
                    tracker_type = "centroid"

                # ---- hand gestures (global per frame) ----
                gesture_performed = None
                if ENABLE_HAND_GESTURES and mp_hands is not None and hands_ctx is not None:
                    hand_results = hands_ctx.process(frame_rgb)
                    if hand_results.multi_hand_landmarks:
                        gesture_performed = detect_hand_gestures(
                            hand_results.multi_hand_landmarks, W, H, hand_state, now, boxes
                        )

                # HUD
                max_combo = max(combos.values()) if combos else 0
                draw_xp_bar(frame, xp, max_xp, level, max_combo, prompt_for_frame)

                if ult_active:
                    if now > ult_end_time:
                        ult_active = False
                    else:
                        cv2.putText(
                            frame,
                            "ULT MODE!",
                            (W // 2 - 110, 80),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1.0,
                            (0, 255, 255),
                            3,
                            cv2.LINE_AA,
                        )

                # ---- per-face logic ----
                for fid, box in list(boxes.items()):
                    x, y, w, h = box
                    cx, cy = x + w / 2.0, y + h / 2.0
                    size_scalar = w + h

                    motion_state = motion_states[fid]
                    dx, dz = compute_motion(
                        np.array([cx, cy], dtype=float),
                        float(size_scalar),
                        motion_state,
                    )

                    combo = combos[fid]
                    draw_face_aura(frame, box, combo=combo, ult=ult_active, show_zone=True)
                    cv2.putText(
                        frame,
                        f"ID {fid}",
                        (int(x), int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    pts = list(trails[fid])
                    for i in range(1, len(pts)):
                        p1 = tuple(pts[i - 1].astype(int))
                        p2 = tuple(pts[i].astype(int))
                        cv2.line(frame, p1, p2, (255, 200, 200), 2)

                    performed = None

                    # ---- action lock: ignore inputs briefly after success ----
                    lock_until = getattr(motion_state, "lock_until", 0.0)
                    locked = now < lock_until

                    # ---- motion-based gestures (only if not locked) ----
                    yaw_deg = 0.0
                    pitch_deg = 0.0

                    norm_dx = 0.0
                    norm_dz = 0.0
                    if not locked:
                        norm_dx = dx / max(40.0, w)
                        norm_dz = dz / max(40.0, size_scalar)

                        # face motion thresholds (more stable)
                        if norm_dx < -0.14:
                            performed = "MOVE LEFT"
                        elif norm_dx > 0.14:
                            performed = "MOVE RIGHT"
                        elif norm_dz > 0.18:
                            performed = "LEAN IN"
                        elif norm_dz < -0.18:
                            performed = "LEAN BACK"

                        # head pose only if not already classified and not moving too much
                        if performed is None and run_detection_this_frame and raw_detections:
                            yaw_deg, pitch_deg = estimate_head_angles(raw_detections[0], W, H)
                            pitch_deg = -pitch_deg  # flip for mirrored UX

                            if abs(norm_dx) < 0.08 and abs(norm_dz) < 0.08:
                                if yaw_deg < -22:
                                    performed = "TURN LEFT"
                                elif yaw_deg > 22:
                                    performed = "TURN RIGHT"
                                elif pitch_deg < -22:
                                    performed = "TILT UP"
                                elif pitch_deg > 22:
                                    performed = "TILT DOWN"

                        # hand gestures override if detected this frame
                        if gesture_performed is not None:
                            performed = gesture_performed

                    success = 0
                    reaction_time_ms = ""

                    # ---- scoring logic ----
                    if performed is not None and performed == prompt_for_frame and not locked:
                        success = 1

                        # lock out further gestures for a short window to avoid MISS after success
                        motion_state.lock_until = now + 0.45

                        rt = (now - prompt_started) * 1000.0
                        reaction_time_ms = f"{rt:.1f}"

                        combo = combos[fid] + 1
                        combos[fid] = combo
                        mult = combo_multiplier(combo)
                        gained = 20 * mult
                        if ult_active:
                            gained *= 2
                        xp += gained

                        floating_texts.append(
                            FloatingText(
                                text=f"PERFECT +{gained} XP (x{mult})",
                                position=(int(cx - 80), int(cy - 20)),
                                ttl=1.0,
                                created_at=now,
                                color=(0, 255, 255),
                                scale=0.85,
                            )
                        )
                        beat_ripples.append(
                            BeatRipple(
                                center=(int(cx), int(cy)),
                                radius=10.0,
                                max_radius=120.0,
                                thickness=2,
                                color=(0, 255, 255),
                            )
                        )
                        audio_mgr.play_sfx("sfx_perfect")

                        if combos[fid] >= 10 and not ult_active:
                            ult_active = True
                            ult_end_time = now + 8.0
                            audio_mgr.play_sfx("sfx_ult")

                        if xp >= max_xp:
                            xp -= max_xp
                            level += 1
                            floating_texts.append(
                                FloatingText(
                                    text=f"LEVEL UP! {level}",
                                    position=(W // 2 - 130, H // 2),
                                    ttl=1.5,
                                    created_at=now,
                                    color=(0, 255, 255),
                                    scale=1.2,
                                )
                            )

                        # new weighted prompt
                        current_prompt = choose_weighted_prompt()
                        prompt_started = now
                        beats_since_prompt = 0
                        audio_mgr.play_voice_for_prompt(current_prompt, now)

                    elif performed is not None and performed != prompt_for_frame and not locked:
                        # only penalize if we actually had a combo running
                        if combos[fid] > 0:
                            floating_texts.append(
                                FloatingText(
                                    text="MISS",
                                    position=(int(cx - 20), int(cy - 20)),
                                    ttl=0.7,
                                    created_at=now,
                                    color=(0, 0, 255),
                                    scale=1.0,
                                )
                            )
                            audio_mgr.play_sfx("sfx_miss")
                        combos[fid] = 0

                    # logging per face
                    if writer is not None:
                        timestamp_ms = int(time.time() * 1000)
                        writer.writerow(
                            [
                                timestamp_ms,
                                frame_idx,
                                fid,
                                tracker_type,
                                int(x),
                                int(y),
                                int(w),
                                int(h),
                                float(cx),
                                float(cy),
                                float(dx),
                                float(dz),
                                prompt_for_frame,
                                success,
                                reaction_time_ms,
                            ]
                        )

                # draw global effects
                floating_texts[:] = draw_floating_texts(frame, floating_texts, now)
                beat_ripples[:] = draw_ripples(frame, beat_ripples)

                draw_prompt_lane(frame, prompt_for_frame, phase=phase, beat_pop=beat_pop)

                # auto-change prompt if it's been around too long and nobody hit it
                if beats_since_prompt >= PROMPT_BEATS and (now - prompt_started) > 1.0:
                    current_prompt = choose_weighted_prompt()
                    prompt_started = now
                    beats_since_prompt = 0
                    audio_mgr.play_voice_for_prompt(current_prompt, now)

                if out is not None and SAVE_VIDEO:
                    out.write(frame)

                cv2.imshow("Face Hero V5 - Rhythm Tracker", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        finally:
            if mp_hands is not None and hands_ctx is not None:
                hands_ctx.close()

    if logfp is not None:
        logfp.close()
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


# ----------------- HAND GESTURES -----------------


def detect_hand_gestures(
    multi_hand_landmarks,
    W: int,
    H: int,
    state: dict,
    now: float,
    boxes: Dict[int, np.ndarray],
):
    """Detect WAVE, CLAP, HANDS UP with cooldown to avoid spam + MISS after action."""
    # global cooldown after a gesture
    if now < state.get("gesture_cooldown_until", 0.0):
        return None

    hands = []
    for lm in multi_hand_landmarks:
        coords = [(p.x * W, p.y * H) for p in lm.landmark]
        hands.append(coords)

    # approximate face zone
    face_boxes = list(boxes.values())
    face_y_top = None
    face_y_bottom = None
    if face_boxes:
        largest = max(face_boxes, key=lambda b: b[2] * b[3])
        fx, fy, fw, fh = largest
        face_y_top = fy
        face_y_bottom = fy + fh

    # ---- CLAP + HANDS UP: need two hands ----
    if len(hands) >= 2:
        l_idx = hands[0][8]
        r_idx = hands[1][8]
        dist = math.hypot(l_idx[0] - r_idx[0], l_idx[1] - r_idx[1])

        # CLAP: fingertips close together horizontally & vertically
        if dist < W * 0.08 and now - state.get("last_clap_time", 0.0) > 0.8:
            state["last_clap_time"] = now
            state["gesture_cooldown_until"] = now + 0.45
            return "CLAP"

        l_wrist = hands[0][0]
        r_wrist = hands[1][0]
        if face_y_top is not None and face_y_bottom is not None:
            zone_top = max(0, face_y_top - (face_y_bottom - face_y_top) * 0.4)
        else:
            zone_top = H * 0.45

        # HANDS UP: both wrists clearly above face zone
        if (
            l_wrist[1] < zone_top
            and r_wrist[1] < zone_top
            and now - state.get("last_handsup_time", 0.0) > 0.8
        ):
            state["last_handsup_time"] = now
            state["gesture_cooldown_until"] = now + 0.45
            return "HANDS UP"

    # ---- WAVE: strong horizontal motion of any hand over short time ----
    history = state["hand_history"]
    for coords in hands:
        center = coords[9]  # middle finger MCP-ish
        history.append((now, center[0]))
    history[:] = [h for h in history if now - h[0] < 0.5]

    if len(history) >= 2:
        xs = [h[1] for h in history]
        min_x, max_x = min(xs), max(xs)
        if max_x - min_x > W * 0.18 and now - state.get("last_wave_time", 0.0) > 0.8:
            state["last_wave_time"] = now
            state["gesture_cooldown_until"] = now + 0.45
            return "WAVE"

    return None


if __name__ == "__main__":
    main()
