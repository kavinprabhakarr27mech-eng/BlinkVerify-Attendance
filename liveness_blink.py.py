"""
Blink-based liveness check (AIML-style feature extraction + decision logic)

What it does:
- Uses MediaPipe FaceMesh to get facial landmarks.
- Computes Eye Aspect Ratio (EAR) from eye landmarks.
- Detects blinks by counting consecutive frames where EAR falls below a threshold.
- Marks "LIVE" if user blinks N times within a time window.

This is intended for demos / attendance anti-cheat flows.
It's not a full anti-deepfake spoof model, but it blocks common QR-photo/replay cheats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

# IMPORTANT: Your mediapipe build doesn't expose mp.solutions.
# So we import FaceMesh directly from mediapipe's solutions package.
from mediapipe.python.solutions import face_mesh as mp_face_mesh




# -----------------------------
# Config you can tune quickly
# -----------------------------
@dataclass(frozen=True)
class LivenessConfig:
    ear_threshold: float = 0.21       # If EAR drops below this => eye likely closed
    min_closed_frames: int = 2        # Closed eye must persist this many frames to count as blink
    required_blinks: int = 2          # How many blinks needed for verification
    timeout_seconds: int = 8          # Must complete within this many seconds
    camera_index: int = 0             # Change if you have multiple cameras


# Eye landmark indices (MediaPipe FaceMesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# -----------------------------
# Math helpers
# -----------------------------
def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _ear(eye_pts: np.ndarray) -> float:
    """
    Eye Aspect Ratio:
      (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)

    eye_pts must be 6 points in this order: p1..p6.
    """
    p1, p2, p3, p4, p5, p6 = eye_pts
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = 2.0 * _dist(p1, p4)
    return float(vertical / (horizontal + 1e-6))


def _lm_xy(face_landmarks, idx: int, w: int, h: int) -> np.ndarray:
    lm = face_landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def compute_ear(face_landmarks, w: int, h: int) -> float:
    left = np.array([_lm_xy(face_landmarks, i, w, h) for i in LEFT_EYE])
    right = np.array([_lm_xy(face_landmarks, i, w, h) for i in RIGHT_EYE])
    return float((_ear(left) + _ear(right)) / 2.0)


# -----------------------------
# Blink-based liveness state machine
# -----------------------------
class BlinkLivenessChecker:
    def __init__(self, cfg: LivenessConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.started_at = time.time()
        self.blinks = 0
        self.closed_frames = 0
        self.verified = False
        self.timed_out = False

    def update(self, ear_value: float | None) -> None:
        """
        Call this every frame with current EAR.
        If ear_value is None (no face), we just don't update blink counters.
        """
        if self.verified or self.timed_out:
            return

        elapsed = time.time() - self.started_at
        if elapsed > self.cfg.timeout_seconds:
            self.timed_out = True
            return

        if ear_value is None:
            # No face => don't guess; just wait.
            return

        # Eye closed?
        if ear_value < self.cfg.ear_threshold:
            self.closed_frames += 1
            return

        # Eye opened again: if it was closed long enough => blink.
        if self.closed_frames >= self.cfg.min_closed_frames:
            self.blinks += 1

        self.closed_frames = 0

        if self.blinks >= self.cfg.required_blinks:
            self.verified = True


# -----------------------------
# UI helpers
# -----------------------------
def put_text(img, text: str, y: int, color=(255, 255, 255), scale: float = 0.75) -> None:
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main() -> None:
    cfg = LivenessConfig()

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam (index={cfg.camera_index}). "
            "Try camera_index=1 or 2 if you have multiple cameras."
        )

    face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

    checker = BlinkLivenessChecker(cfg)

    print("\nBlink Liveness Demo")
    print("- Look at the camera and blink twice within the timer.")
    print("- Press R to reset, Q to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        ear_value = None
        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0].landmark
            ear_value = compute_ear(face, w, h)

        checker.update(ear_value)

        # Draw status
        put_text(frame, "Liveness Check (Blink-based)", 30)
        if ear_value is None:
            put_text(frame, "Face: not detected", 60, (0, 255, 255))
        else:
            put_text(frame, f"EAR: {ear_value:.3f}", 60)

        remaining = max(0, int(cfg.timeout_seconds - (time.time() - checker.started_at)))
        put_text(frame, f"Blinks: {checker.blinks}/{cfg.required_blinks}", 95, (255, 255, 0))
        put_text(frame, f"Time left: {remaining}s", 125, (255, 255, 0))

        if checker.verified:
            put_text(frame, "STATUS: LIVE (Verified ✅)", 165, (0, 255, 0))
        elif checker.timed_out:
            put_text(frame, "STATUS: FAIL (Timeout ❌)", 165, (0, 0, 255))
        else:
            put_text(frame, "Instruction: Blink naturally to verify", 165, (255, 255, 0))

        put_text(frame, "Keys: [R] Reset   [Q] Quit", h - 20, (200, 200, 200), scale=0.6)

        cv2.imshow("Blink Liveness Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("r"), ord("R")):
            checker.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
