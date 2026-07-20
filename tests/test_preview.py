"""Tests for preview scaling and the side-by-side compare composition."""

import numpy as np

from vertical_video_converter.video_writer import (
    PREVIEW_GAP_PX,
    PREVIEW_MAX_HEIGHT,
    compose_side_by_side,
    scale_to_max_height,
)


def frame(width, height, value=128):
    return np.full((height, width, 3), value, dtype=np.uint8)


class TestScaleToMaxHeight:
    def test_tall_frame_scaled_to_max_height(self):
        out = scale_to_max_height(frame(1280, 720))
        assert out.shape[0] == PREVIEW_MAX_HEIGHT
        # Width follows the aspect ratio: 1280 * 400/720
        assert out.shape[1] == round(1280 * PREVIEW_MAX_HEIGHT / 720)

    def test_small_frame_untouched(self):
        small = frame(300, 200)
        assert scale_to_max_height(small) is small

    def test_exact_height_untouched(self):
        exact = frame(500, PREVIEW_MAX_HEIGHT)
        assert scale_to_max_height(exact) is exact

    def test_custom_max_height(self):
        out = scale_to_max_height(frame(1920, 1080), max_height=200)
        assert out.shape[:2] == (200, round(1920 * 200 / 1080))


class TestComposeSideBySide:
    def test_both_sides_capped_at_max_height(self):
        out = compose_side_by_side(frame(1280, 720), frame(404, 720))
        assert out.shape[0] == PREVIEW_MAX_HEIGHT
        expected_width = (
            round(1280 * PREVIEW_MAX_HEIGHT / 720) + PREVIEW_GAP_PX + round(404 * PREVIEW_MAX_HEIGHT / 720)
        )
        assert out.shape[1] == expected_width

    def test_shorter_side_padded_to_common_height(self):
        out = compose_side_by_side(frame(1280, 720), frame(200, 300))
        # Left scales to 400, right stays 300 and is padded
        assert out.shape[0] == PREVIEW_MAX_HEIGHT
        assert out.shape[1] == round(1280 * PREVIEW_MAX_HEIGHT / 720) + PREVIEW_GAP_PX + 200

    def test_gap_between_panels_is_black(self):
        left = frame(400, 400, value=200)
        right = frame(400, 400, value=200)
        out = compose_side_by_side(left, right)
        gap_column = out[:, 400 : 400 + PREVIEW_GAP_PX]
        assert (gap_column == 0).all()
