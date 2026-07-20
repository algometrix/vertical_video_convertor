"""Tests for hard scene-cut detection."""

import numpy as np

from vertical_video_converter.scene_detector import SceneCutDetector


def solid_frame(value, w=640, h=360):
    return np.full((h, w, 3), value, dtype=np.uint8)


def noisy_frame(base, rng, amplitude=6, w=640, h=360):
    noise = rng.integers(-amplitude, amplitude + 1, size=(h, w, 3))
    return np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)


class TestSceneCutDetector:
    def test_first_frame_is_never_a_cut(self):
        detector = SceneCutDetector()
        assert detector.update(solid_frame(128)) is False

    def test_static_shot_with_noise_is_not_a_cut(self):
        detector = SceneCutDetector()
        rng = np.random.default_rng(42)
        base = solid_frame(128)
        detector.update(base)
        for _ in range(10):
            assert detector.update(noisy_frame(base, rng)) is False

    def test_hard_cut_is_detected(self):
        detector = SceneCutDetector()
        detector.update(solid_frame(30))
        assert detector.update(solid_frame(200)) is True

    def test_reset_forgets_previous_frame(self):
        detector = SceneCutDetector()
        detector.update(solid_frame(30))
        detector.reset()
        assert detector.update(solid_frame(200)) is False

    def test_rejects_invalid_threshold(self):
        import pytest

        with pytest.raises(ValueError):
            SceneCutDetector(threshold=0)
