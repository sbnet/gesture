"""Configurable constants for the hand wireframe demo."""

from dataclasses import dataclass


@dataclass
class AppConfig:
    """Runtime configuration, typically provided via CLI."""

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    max_hands: int = 1
    model_complexity: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    mirror: bool = True
    debug: bool = False


# Drawing constants
LANDMARK_RADIUS = 5
LANDMARK_COLOR = (0, 220, 100)  # green
CONNECTION_COLOR = (200, 200, 200)  # light grey
CONNECTION_THICKNESS = 2
PALM_COLOR = (100, 180, 255)  # blue tint for palm connections

GESTURE_FONT_SCALE = 1.2
GESTURE_FONT_THICKNESS = 2
GESTURE_COLOR_KNOWN = (0, 255, 128)
GESTURE_COLOR_UNKNOWN = (160, 160, 160)
HUD_FONT_SCALE = 0.6
HUD_COLOR = (220, 220, 220)

# Gesture smoothing window (frames)
SMOOTHING_WINDOW = 7

# Minimum score to report a gesture (0-1)
GESTURE_SCORE_THRESHOLD = 0.75

# FPS averaging window
FPS_WINDOW = 30
