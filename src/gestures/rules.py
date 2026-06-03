"""Rule-based gesture definitions.

Each rule is a function that takes a feature dict and returns a score in [0, 1].
A score of 1.0 means the gesture matches perfectly; 0.0 means it does not match.
Using continuous scores (rather than hard booleans) allows the recognizer to pick
the best match when multiple rules partially fire.
"""

from __future__ import annotations

from collections.abc import Callable

Features = dict[str, float | bool]
RuleFn = Callable[[Features], float]


def _b(f: Features, key: str) -> bool:
    return bool(f[key])


def _f(f: Features, key: str) -> float:
    return float(f[key])


# ── individual rules ──────────────────────────────────────────────────────────


def rule_open_palm(f: Features) -> float:
    """All five fingers extended."""
    up = [_b(f, k) for k in ("thumb", "index", "middle", "ring", "pinky")]
    return sum(up) / 5.0


def rule_fist(f: Features) -> float:
    """All fingers closed."""
    up = [_b(f, k) for k in ("thumb", "index", "middle", "ring", "pinky")]
    closed = sum(1 for v in up if not v)
    return closed / 5.0


def rule_peace(f: Features) -> float:
    """Index + middle up, ring + pinky + thumb down."""
    score = 0.0
    if _b(f, "index"):
        score += 0.35
    if _b(f, "middle"):
        score += 0.35
    if not _b(f, "ring"):
        score += 0.1
    if not _b(f, "pinky"):
        score += 0.1
    if not _b(f, "thumb"):
        score += 0.1
    return score


def rule_point(f: Features) -> float:
    """Index up only (thumb optional but others down)."""
    score = 0.0
    if _b(f, "index"):
        score += 0.5
    if not _b(f, "middle"):
        score += 0.2
    if not _b(f, "ring"):
        score += 0.15
    if not _b(f, "pinky"):
        score += 0.15
    return score


def rule_thumbs_up(f: Features) -> float:
    """Thumb up, all other fingers down."""
    score = 0.0
    if _b(f, "thumb"):
        score += 0.5
    if not _b(f, "index"):
        score += 0.15
    if not _b(f, "middle"):
        score += 0.15
    if not _b(f, "ring"):
        score += 0.1
    if not _b(f, "pinky"):
        score += 0.1
    return score


def rule_ok_sign(f: Features) -> float:
    """Thumb + index pinched (small distance), middle/ring/pinky extended."""
    pinch = _f(f, "pinch")
    # Pinch threshold: normalised distance < 0.3 -> tight pinch
    pinch_score = max(0.0, 1.0 - pinch / 0.3) * 0.5
    score = pinch_score
    if _b(f, "middle"):
        score += 0.2
    if _b(f, "ring"):
        score += 0.15
    if _b(f, "pinky"):
        score += 0.15
    return score


def rule_rock(f: Features) -> float:
    """Index + pinky up, middle + ring + thumb down."""
    score = 0.0
    if _b(f, "index"):
        score += 0.35
    if _b(f, "pinky"):
        score += 0.35
    if not _b(f, "middle"):
        score += 0.1
    if not _b(f, "ring"):
        score += 0.1
    if not _b(f, "thumb"):
        score += 0.1
    return score


# ── registry ──────────────────────────────────────────────────────────────────

GESTURE_RULES: dict[str, RuleFn] = {
    "open_palm": rule_open_palm,
    "fist": rule_fist,
    "peace": rule_peace,
    "point": rule_point,
    "thumbs_up": rule_thumbs_up,
    "ok_sign": rule_ok_sign,
    "rock": rule_rock,
}
