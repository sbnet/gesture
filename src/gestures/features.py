"""Feature extraction from 21 MediaPipe landmarks.

All distances are normalised by the wrist->middle-MCP distance so they are
scale-invariant (independent of how close the hand is to the camera).
"""

from __future__ import annotations

import math

from models import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    HandLandmarks,
    Landmark,
)


def _dist(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def hand_scale(lms: list[Landmark]) -> float:
    """Reference length: wrist to middle-finger MCP."""
    d = _dist(lms[WRIST], lms[MIDDLE_MCP])
    return d if d > 1e-6 else 1.0


def finger_extended(lms: list[Landmark], tip: int, pip: int, mcp: int) -> bool:
    """Return True if a finger is reasonably extended (tip above pip above mcp in Y).

    MediaPipe Y increases downward, so a raised finger has a *smaller* Y value at the tip.
    """
    # Primary test: tip is above pip
    tip_above_pip = lms[tip].y < lms[pip].y
    # Secondary: pip is above mcp (helps distinguish partially bent fingers)
    pip_above_mcp = lms[pip].y < lms[mcp].y
    return tip_above_pip and pip_above_mcp


def thumb_extended(lms: list[Landmark], handedness: str) -> bool:
    """Thumb extension uses X-axis because the thumb is oriented sideways.

    For a right hand the thumb tip is to the left of the IP joint when extended.
    Handedness is the MediaPipe label ("Left"/"Right") which is mirrored when
    the image is not flipped (we flip it, so "Right" means the user's right hand).
    """
    tip = lms[THUMB_TIP]
    ip = lms[THUMB_IP]
    mcp = lms[THUMB_MCP]
    # Extended thumb: tip is farther from index MCP than the IP joint
    if handedness == "Right":
        return tip.x < ip.x and ip.x < mcp.x
    else:
        return tip.x > ip.x and ip.x > mcp.x


def fingers_up(lms: list[Landmark], handedness: str) -> tuple[bool, bool, bool, bool, bool]:
    """Return (thumb, index, middle, ring, pinky) extension booleans."""
    return (
        thumb_extended(lms, handedness),
        finger_extended(lms, INDEX_TIP, INDEX_PIP, INDEX_MCP),
        finger_extended(lms, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        finger_extended(lms, RING_TIP, RING_PIP, RING_MCP),
        finger_extended(lms, PINKY_TIP, PINKY_PIP, PINKY_MCP),
    )


def pinch_distance(lms: list[Landmark]) -> float:
    """Normalised thumb-tip to index-tip distance (pinch metric)."""
    scale = hand_scale(lms)
    return _dist(lms[THUMB_TIP], lms[INDEX_TIP]) / scale


def extract_features(hand: HandLandmarks) -> dict[str, float | bool]:
    """Compute all features for *hand* in one call."""
    lms = hand.landmarks
    thumb, index, middle, ring, pinky = fingers_up(lms, hand.handedness)
    return {
        "thumb": thumb,
        "index": index,
        "middle": middle,
        "ring": ring,
        "pinky": pinky,
        "pinch": pinch_distance(lms),
        "extended_count": int(thumb) + int(index) + int(middle) + int(ring) + int(pinky),
    }
