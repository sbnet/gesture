"""Unit tests for gesture feature extraction."""

from __future__ import annotations

from gestures.features import extract_features
from models import HandLandmarks


def test_open_palm_all_fingers_up(open_palm_hand: HandLandmarks) -> None:
    feats = extract_features(open_palm_hand)
    assert feats["thumb"] is True
    assert feats["index"] is True
    assert feats["middle"] is True
    assert feats["ring"] is True
    assert feats["pinky"] is True
    assert int(feats["extended_count"]) == 5  # type: ignore[arg-type]


def test_fist_no_fingers_up(fist_hand: HandLandmarks) -> None:
    feats = extract_features(fist_hand)
    assert feats["index"] is False
    assert feats["middle"] is False
    assert feats["ring"] is False
    assert feats["pinky"] is False


def test_peace_index_middle_up(peace_hand: HandLandmarks) -> None:
    feats = extract_features(peace_hand)
    assert feats["index"] is True
    assert feats["middle"] is True
    assert feats["ring"] is False
    assert feats["pinky"] is False


def test_point_index_only(point_hand: HandLandmarks) -> None:
    feats = extract_features(point_hand)
    assert feats["index"] is True
    assert feats["middle"] is False
    assert feats["ring"] is False
    assert feats["pinky"] is False


def test_rock_index_and_pinky_up(rock_hand: HandLandmarks) -> None:
    feats = extract_features(rock_hand)
    assert feats["index"] is True
    assert feats["middle"] is False
    assert feats["ring"] is False
    assert feats["pinky"] is True


def test_extended_count_matches_booleans(open_palm_hand: HandLandmarks) -> None:
    feats = extract_features(open_palm_hand)
    keys = ("thumb", "index", "middle", "ring", "pinky")
    expected = sum(1 for k in keys if feats[k])
    assert feats["extended_count"] == expected
