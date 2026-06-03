"""Unit tests for gesture rules using synthetic hands."""

from __future__ import annotations

from gestures.features import extract_features
from gestures.rules import (
    rule_fist,
    rule_open_palm,
    rule_peace,
    rule_point,
    rule_rock,
    rule_thumbs_up,
)
from models import HandLandmarks


def _score(rule_fn, hand: HandLandmarks) -> float:
    return rule_fn(extract_features(hand))


def test_open_palm_scores_high(open_palm_hand: HandLandmarks) -> None:
    assert _score(rule_open_palm, open_palm_hand) >= 0.8


def test_fist_scores_high(fist_hand: HandLandmarks) -> None:
    assert _score(rule_fist, fist_hand) >= 0.7


def test_peace_scores_high(peace_hand: HandLandmarks) -> None:
    assert _score(rule_peace, peace_hand) >= 0.7


def test_point_scores_high(point_hand: HandLandmarks) -> None:
    assert _score(rule_point, point_hand) >= 0.7


def test_thumbs_up_scores_high(thumbs_up_hand: HandLandmarks) -> None:
    assert _score(rule_thumbs_up, thumbs_up_hand) >= 0.7


def test_rock_scores_high(rock_hand: HandLandmarks) -> None:
    assert _score(rule_rock, rock_hand) >= 0.7


# Cross-gesture sanity: a fist should not score high as open_palm
def test_fist_not_open_palm(fist_hand: HandLandmarks) -> None:
    assert _score(rule_open_palm, fist_hand) < 0.3


def test_open_palm_not_fist(open_palm_hand: HandLandmarks) -> None:
    assert _score(rule_fist, open_palm_hand) < 0.3


def test_scores_in_range(open_palm_hand: HandLandmarks) -> None:
    feats = extract_features(open_palm_hand)
    for fn in (rule_open_palm, rule_fist, rule_peace, rule_point, rule_thumbs_up, rule_rock):
        s = fn(feats)
        assert 0.0 <= s <= 1.0, f"{fn.__name__} returned {s}"
