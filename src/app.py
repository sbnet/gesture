"""Main real-time loop: capture -> detect -> recognise -> draw -> display."""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from camera import Camera, CameraError
from config import FPS_WINDOW, AppConfig
from drawing import draw_gesture_label, draw_hud, draw_wireframe
from gestures import GestureRecognizer
from hand_tracking import HandTracker
from models import GestureResult


def _make_no_hand_result() -> GestureResult:
    return GestureResult(name="unknown", score=0.0)


def run(config: AppConfig) -> None:
    """Open the camera and run the recognition loop until the user quits."""
    try:
        camera = Camera(config.camera_index, config.frame_width, config.frame_height)
    except CameraError as exc:
        raise SystemExit(f"[camera] {exc}") from exc

    tracker = HandTracker(
        max_hands=config.max_hands,
        model_complexity=config.model_complexity,
        min_detection_confidence=config.min_detection_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    recognizer = GestureRecognizer()

    fps_times: deque[float] = deque(maxlen=FPS_WINDOW)
    last_result = _make_no_hand_result()
    consecutive_failures = 0
    MAX_FAILURES = 30  # ~1 s at 30 fps before giving up

    with camera, tracker:
        while True:
            t0 = time.perf_counter()

            frame: np.ndarray | None = camera.read()
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    raise SystemExit(
                        "[camera] Too many consecutive read failures. "
                        "Check that the camera is streaming."
                    )
                continue
            consecutive_failures = 0

            if config.mirror:
                frame = cv2.flip(frame, 1)

            hands = tracker.process(frame)

            if hands:
                hand = hands[0]
                last_result = recognizer.recognize(hand)
                draw_wireframe(frame, hand)
            else:
                last_result = _make_no_hand_result()

            draw_gesture_label(frame, last_result)

            fps_times.append(time.perf_counter() - t0)
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0

            draw_hud(
                frame,
                fps=fps,
                num_hands=len(hands),
                debug=config.debug,
                debug_data=last_result.debug if config.debug else None,
            )

            cv2.imshow("hand-wire-demo  [q / Esc = quit]", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 27 = Esc
                break

    cv2.destroyAllWindows()
