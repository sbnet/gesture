"""Core data types for hand wireframe demo."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Landmark:
    """A single 3D landmark point, normalized to [0, 1]."""

    x: float
    y: float
    z: float


# MediaPipe hand landmark indices (0-20)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Connections that form the hand skeleton
HAND_CONNECTIONS: list[tuple[int, int]] = [
    # Palm
    (WRIST, THUMB_CMC),
    (WRIST, INDEX_MCP),
    (WRIST, PINKY_MCP),
    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, PINKY_MCP),
    # Thumb
    (THUMB_CMC, THUMB_MCP),
    (THUMB_MCP, THUMB_IP),
    (THUMB_IP, THUMB_TIP),
    # Index
    (INDEX_MCP, INDEX_PIP),
    (INDEX_PIP, INDEX_DIP),
    (INDEX_DIP, INDEX_TIP),
    # Middle
    (MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_DIP, MIDDLE_TIP),
    # Ring
    (RING_MCP, RING_PIP),
    (RING_PIP, RING_DIP),
    (RING_DIP, RING_TIP),
    # Pinky
    (PINKY_MCP, PINKY_PIP),
    (PINKY_PIP, PINKY_DIP),
    (PINKY_DIP, PINKY_TIP),
]


@dataclass
class HandLandmarks:
    """All 21 landmarks of a detected hand plus metadata."""

    landmarks: list[Landmark]
    handedness: str  # "Left" or "Right"
    score: float


@dataclass
class GestureResult:
    """Result of a gesture recognition pass."""

    name: str
    score: float
    debug: dict[str, float | bool | str] = field(default_factory=dict)
