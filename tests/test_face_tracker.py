"""Tests for sticky main-face selection and dropout hold."""

from dataclasses import dataclass, field

from vertical_video_converter.face_tracker import FaceTracker


@dataclass
class Face:
    bbox: list[float] = field(default_factory=list)
    det_score: float = 0.9


def face_at(cx, cy, size=100, score=0.9):
    half = size / 2
    return Face(bbox=[cx - half, cy - half, cx + half, cy + half], det_score=score)


class TestSelection:
    def test_picks_strongest_face_first(self):
        tracker = FaceTracker(1920, 1080)
        small = face_at(400, 400, size=80)
        large = face_at(1200, 400, size=160)
        assert tracker.update([small, large], now=0.0) == (1200, 400)

    def test_sticks_with_incumbent_over_similar_challenger(self):
        tracker = FaceTracker(1920, 1080)
        host_a = face_at(600, 400, size=120)
        host_b = face_at(1300, 400, size=110)
        assert tracker.update([host_a, host_b], now=0.0) == (600, 400)
        # Host B wobbles slightly larger: camera must not flip
        host_b_bigger = face_at(1300, 400, size=130)
        assert tracker.update([host_a, host_b_bigger], now=0.1) == (600, 400)

    def test_switches_to_decisively_stronger_challenger(self):
        tracker = FaceTracker(1920, 1080)
        host_a = face_at(600, 400, size=120)
        assert tracker.update([host_a], now=0.0) == (600, 400)
        closeup = face_at(1300, 400, size=400)
        assert tracker.update([host_a, closeup], now=0.1) == (1300, 400)

    def test_incumbent_tracked_by_position_as_it_moves(self):
        tracker = FaceTracker(1920, 1080)
        tracker.update([face_at(600, 400)], now=0.0)
        moved = face_at(650, 410)
        other = face_at(1400, 400, size=105)
        assert tracker.update([moved, other], now=0.1) == (650, 410)


class TestDropoutHold:
    def test_holds_last_position_during_brief_dropout(self):
        tracker = FaceTracker(1920, 1080, hold_seconds=2.0)
        tracker.update([face_at(700, 300)], now=0.0)
        assert tracker.update([], now=1.5) == (700, 300)

    def test_gives_up_after_hold_expires(self):
        tracker = FaceTracker(1920, 1080, hold_seconds=2.0)
        tracker.update([face_at(700, 300)], now=0.0)
        assert tracker.update([], now=2.5) is None
        assert not tracker.has_face

    def test_reset_clears_hold(self):
        tracker = FaceTracker(1920, 1080)
        tracker.update([face_at(700, 300)], now=0.0)
        tracker.reset()
        assert tracker.update([], now=0.1) is None

    def test_no_faces_ever_returns_none(self):
        tracker = FaceTracker(1920, 1080)
        assert tracker.update([], now=0.0) is None
