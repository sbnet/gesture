"""Shared test fixtures: synthetic hand landmark builders."""

from __future__ import annotations

import pytest

from models import HandLandmarks, Landmark


def _flat_hand(*, x: float = 0.5, y: float = 0.5, z: float = 0.0) -> Landmark:
    return Landmark(x=x, y=y, z=z)


def make_landmarks(positions: list[tuple[float, float, float]]) -> list[Landmark]:
    return [Landmark(x=x, y=y, z=z) for x, y, z in positions]


def build_hand(
    positions: list[tuple[float, float, float]], handedness: str = "Right"
) -> HandLandmarks:
    return HandLandmarks(
        landmarks=make_landmarks(positions),
        handedness=handedness,
        score=1.0,
    )


# ── Canonical synthetic hands ──────────────────────────────────────────────────
# Landmark order follows MediaPipe index 0-20.
# Y increases downward; a raised fingertip has lower Y than its MCP.
# We model fingers as two-segment chains: MCP(low Y=0.7) -> PIP(0.5) -> TIP(0.3)
# A curled finger: MCP(0.7) -> PIP(0.65) -> TIP(0.68)

# x positions per finger (spaced across image width)
_X = {
    "wrist": 0.5,
    "thumb_cmc": 0.35,
    "thumb_mcp": 0.28,
    "thumb_ip": 0.22,
    "thumb_tip": 0.16,
    "index_mcp": 0.42,
    "index_pip": 0.42,
    "index_dip": 0.42,
    "index_tip": 0.42,
    "middle_mcp": 0.50,
    "middle_pip": 0.50,
    "middle_dip": 0.50,
    "middle_tip": 0.50,
    "ring_mcp": 0.58,
    "ring_pip": 0.58,
    "ring_dip": 0.58,
    "ring_tip": 0.58,
    "pinky_mcp": 0.65,
    "pinky_pip": 0.65,
    "pinky_dip": 0.65,
    "pinky_tip": 0.65,
}


# Helper: build a finger column as (x, y, z) for [mcp, pip, dip, tip]
def _finger_up(x: float) -> list[tuple[float, float, float]]:
    return [(x, 0.70, 0), (x, 0.55, 0), (x, 0.40, 0), (x, 0.25, 0)]


def _finger_down(x: float) -> list[tuple[float, float, float]]:
    return [(x, 0.70, 0), (x, 0.68, 0), (x, 0.72, 0), (x, 0.75, 0)]


def _thumb_up_right() -> list[tuple[float, float, float]]:
    # For right hand: thumb tip x < ip x (extended left)
    return [
        (0.35, 0.75, 0),  # cmc
        (0.28, 0.65, 0),  # mcp
        (0.20, 0.55, 0),  # ip
        (0.12, 0.45, 0),  # tip
    ]


def _thumb_down_right() -> list[tuple[float, float, float]]:
    # Tucked under: tip x > ip x
    return [
        (0.35, 0.75, 0),  # cmc
        (0.38, 0.70, 0),  # mcp
        (0.42, 0.68, 0),  # ip
        (0.45, 0.67, 0),  # tip
    ]


def _build_positions(
    thumb: list[tuple[float, float, float]],
    index_up: bool,
    middle_up: bool,
    ring_up: bool,
    pinky_up: bool,
) -> list[tuple[float, float, float]]:
    """Assemble all 21 landmark positions."""

    def pick(x: float, up: bool) -> list[tuple[float, float, float]]:
        return _finger_up(x) if up else _finger_down(x)

    # 0: wrist
    pts: list[tuple[float, float, float]] = [(0.5, 0.90, 0)]
    # 1-4: thumb (cmc, mcp, ip, tip)
    pts.extend(thumb)
    # 5-8: index (mcp, pip, dip, tip)
    pts.extend(pick(_X["index_mcp"], index_up))
    # 9-12: middle
    pts.extend(pick(_X["middle_mcp"], middle_up))
    # 13-16: ring
    pts.extend(pick(_X["ring_mcp"], ring_up))
    # 17-20: pinky
    pts.extend(pick(_X["pinky_mcp"], pinky_up))
    return pts


@pytest.fixture
def open_palm_hand() -> HandLandmarks:
    pts = _build_positions(_thumb_up_right(), True, True, True, True)
    return build_hand(pts)


@pytest.fixture
def fist_hand() -> HandLandmarks:
    pts = _build_positions(_thumb_down_right(), False, False, False, False)
    return build_hand(pts)


@pytest.fixture
def peace_hand() -> HandLandmarks:
    pts = _build_positions(_thumb_down_right(), True, True, False, False)
    return build_hand(pts)


@pytest.fixture
def point_hand() -> HandLandmarks:
    pts = _build_positions(_thumb_down_right(), True, False, False, False)
    return build_hand(pts)


@pytest.fixture
def thumbs_up_hand() -> HandLandmarks:
    pts = _build_positions(_thumb_up_right(), False, False, False, False)
    return build_hand(pts)


@pytest.fixture
def rock_hand() -> HandLandmarks:
    pts = _build_positions(_thumb_down_right(), True, False, False, True)
    return build_hand(pts)
