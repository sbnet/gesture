"""Gesture recognizer: runs rules, scores, smooths, returns GestureResult."""

from __future__ import annotations

from collections import deque

from config import GESTURE_SCORE_THRESHOLD, SMOOTHING_WINDOW
from models import GestureResult, HandLandmarks

from .features import extract_features
from .rules import GESTURE_RULES


class GestureRecognizer:
    """Wraps rule evaluation and temporal smoothing."""

    def __init__(
        self,
        threshold: float = GESTURE_SCORE_THRESHOLD,
        smoothing_window: int = SMOOTHING_WINDOW,
    ) -> None:
        self._threshold = threshold
        self._history: deque[str] = deque(maxlen=smoothing_window)

    def _best_gesture(self, features: dict[str, float | bool]) -> tuple[str, float]:
        """Return (gesture_name, score) for the highest-scoring rule."""
        scores = {name: fn(features) for name, fn in GESTURE_RULES.items()}
        best_name = max(scores, key=lambda k: scores[k])
        best_score = scores[best_name]
        if best_score < self._threshold:
            return "unknown", best_score
        return best_name, best_score

    def _smoothed_name(self, raw_name: str) -> str:
        """Return the most common name in the recent history window."""
        self._history.append(raw_name)
        return max(set(self._history), key=lambda n: list(self._history).count(n))

    def recognize(self, hand: HandLandmarks) -> GestureResult:
        """Detect the gesture for a single hand and return a GestureResult."""
        features = extract_features(hand)
        raw_name, score = self._best_gesture(features)
        smoothed = self._smoothed_name(raw_name)

        # Cast features to the right type for GestureResult.debug
        debug: dict[str, float | bool | str] = dict(features.items())
        debug["raw"] = raw_name

        return GestureResult(name=smoothed, score=score, debug=debug)
