"""OpenCV drawing helpers: wireframe, HUD, gesture label."""

from __future__ import annotations

import cv2
import numpy as np

import config as cfg
from models import (
    HAND_CONNECTIONS,
    GestureResult,
    HandLandmarks,
)

# Palm connections (subset of HAND_CONNECTIONS) drawn in a different colour
_PALM_PAIRS = {(0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)}


def _to_pixel(lm_x: float, lm_y: float, w: int, h: int) -> tuple[int, int]:
    return int(lm_x * w), int(lm_y * h)


def draw_wireframe(frame: np.ndarray, hand: HandLandmarks) -> None:
    """Draw skeleton connections and landmark dots onto *frame* in-place."""
    h, w = frame.shape[:2]
    lms = hand.landmarks

    for a, b in HAND_CONNECTIONS:
        pt_a = _to_pixel(lms[a].x, lms[a].y, w, h)
        pt_b = _to_pixel(lms[b].x, lms[b].y, w, h)
        color = cfg.PALM_COLOR if (a, b) in _PALM_PAIRS else cfg.CONNECTION_COLOR
        cv2.line(frame, pt_a, pt_b, color, cfg.CONNECTION_THICKNESS, cv2.LINE_AA)

    for lm in lms:
        pt = _to_pixel(lm.x, lm.y, w, h)
        cv2.circle(frame, pt, cfg.LANDMARK_RADIUS, cfg.LANDMARK_COLOR, -1, cv2.LINE_AA)


def draw_gesture_label(frame: np.ndarray, result: GestureResult) -> None:
    """Draw the gesture name in the top-left corner."""
    color = cfg.GESTURE_COLOR_KNOWN if result.name != "unknown" else cfg.GESTURE_COLOR_UNKNOWN
    text = result.name.upper().replace("_", " ")
    cv2.putText(
        frame,
        text,
        (20, 60),
        cv2.FONT_HERSHEY_DUPLEX,
        cfg.GESTURE_FONT_SCALE,
        color,
        cfg.GESTURE_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    score_text = f"{result.score:.0%}"
    cv2.putText(frame, score_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    num_hands: int,
    debug: bool = False,
    debug_data: dict[str, float | bool | str] | None = None,
) -> None:
    """Draw FPS, hand count, and optional debug info in the top-right area."""
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Hands: {num_hands}",
    ]
    if debug and debug_data:
        for k, v in list(debug_data.items())[:8]:
            val = f"{v:.2f}" if isinstance(v, float) else str(v)
            lines.append(f"{k}: {val}")

    for i, line in enumerate(lines):
        y = 25 + i * 20
        cv2.putText(
            frame,
            line,
            (w - 180, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            cfg.HUD_FONT_SCALE,
            cfg.HUD_COLOR,
            1,
            cv2.LINE_AA,
        )
