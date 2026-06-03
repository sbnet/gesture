"""MediaPipe Tasks wrapper: converts results into our HandLandmarks type.

Uses the modern mp.tasks API (mediapipe >= 0.10.20).
The hand landmarker model (~1 MB) is downloaded once to ~/.cache/hand-wire-demo/.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError("mediapipe is required. Run: uv sync") from exc

from models import HandLandmarks, Landmark

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_CACHE = Path.home() / ".cache" / "hand-wire-demo" / "hand_landmarker.task"


def _ensure_model() -> Path:
    """Download the hand landmarker model on first use."""
    if not _MODEL_CACHE.exists():
        _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand landmarker model -> {_MODEL_CACHE} (first run only)")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)
    return _MODEL_CACHE


class HandTracker:
    """Detects hand landmarks in BGR frames using the MediaPipe Tasks API."""

    def __init__(
        self,
        max_hands: int = 1,
        model_complexity: int = 1,  # kept for API compatibility, unused by Tasks API
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = _ensure_model()

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._start_ms = int(time.monotonic() * 1_000)

    def process(self, bgr_frame: np.ndarray) -> list[HandLandmarks]:
        """Detect hands in *bgr_frame* and return a list of HandLandmarks."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1_000) - self._start_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return []

        hands: list[HandLandmarks] = []
        for lm_list, handedness_cats in zip(result.hand_landmarks, result.handedness, strict=True):
            landmarks = [Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in lm_list]
            label = handedness_cats[0].category_name  # "Left" or "Right"
            score = handedness_cats[0].score
            hands.append(HandLandmarks(landmarks=landmarks, handedness=label, score=score))

        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
