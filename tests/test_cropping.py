"""Tests for sub-pixel cropping.

getRectSubPix samples starting at ``center - (size - 1) / 2``: an odd-size
patch copies exactly at integer centers, an even-size patch at half-pixel
centers. The tests use both parities.
"""

import numpy as np
import pytest

from vertical_video_converter.cropping import SubpixelCropper


def gradient_frame(width=100, height=60):
    """Horizontal gradient: pixel value == x coordinate (fits in uint8)."""
    row = np.arange(width, dtype=np.uint8)
    return np.tile(row, (height, 1))


class TestSubpixelCropper:
    def test_odd_crop_integer_center_is_exact_copy(self):
        frame = gradient_frame()
        cropper = SubpixelCropper(100, 60, 21, 21)
        out = cropper.crop(frame, 50.0, 30.0)
        np.testing.assert_array_equal(out, frame[20:41, 40:61])

    def test_even_crop_half_pixel_center_is_exact_copy(self):
        frame = gradient_frame()
        cropper = SubpixelCropper(100, 60, 20, 20)
        out = cropper.crop(frame, 50.5, 30.5)
        np.testing.assert_array_equal(out, frame[21:41, 41:61])

    def test_fractional_center_interpolates(self):
        frame = gradient_frame()
        cropper = SubpixelCropper(100, 60, 21, 21)
        out = cropper.crop(frame, 50.25, 30.0)
        # A quarter-pixel shift right: every value grows by ~0.25
        expected = frame[20:41, 40:61].astype(np.float64) + 0.25
        np.testing.assert_allclose(out.astype(np.float64), expected, atol=0.51)

    def test_center_clamped_to_frame(self):
        frame = gradient_frame()
        cropper = SubpixelCropper(100, 60, 21, 21)
        # Way off the left edge: window parks at the frame's left border
        out = cropper.crop(frame, -500.0, -500.0)
        assert out.shape[:2] == (21, 21)
        assert float(out[10, 0]) <= 1.0  # leftmost source columns
        # Way off the right edge: window parks at the right border
        out = cropper.crop(frame, 500.0, 500.0)
        assert out.shape[:2] == (21, 21)
        assert float(out[10, -1]) >= 98.0  # rightmost source columns

    def test_stationary_camera_snaps_to_exact_pixels_odd(self):
        frame = gradient_frame()
        cropper = SubpixelCropper(100, 60, 21, 21)
        cropper.crop(frame, 50.4, 30.0)
        out = cropper.crop(frame, 50.4, 30.0)  # unchanged position: snapped
        np.testing.assert_array_equal(out, frame[20:41, 40:61])

    def test_stationary_camera_snaps_to_exact_pixels_even(self):
        frame = gradient_frame()
        cropper = SubpixelCropper(100, 60, 20, 20)
        cropper.crop(frame, 50.4, 30.4)
        out = cropper.crop(frame, 50.4, 30.4)  # snapped to the half-pixel grid
        np.testing.assert_array_equal(out, frame[21:41, 41:61])

    def test_moving_camera_stays_subpixel(self):
        frame = gradient_frame()
        cropper = SubpixelCropper(100, 60, 21, 21)
        cropper.crop(frame, 40.0, 30.0)
        out = cropper.crop(frame, 41.5, 30.0)  # moving: no snapping
        expected = frame[20:41, 31:52].astype(np.float64) + 0.5
        np.testing.assert_allclose(out.astype(np.float64), expected, atol=0.51)

    def test_output_size_matches_crop(self):
        cropper = SubpixelCropper(100, 60, 34, 22)
        out = cropper.crop(gradient_frame(), 50.0, 30.0)
        assert out.shape[:2] == (22, 34)

    def test_rejects_crop_larger_than_frame(self):
        with pytest.raises(ValueError):
            SubpixelCropper(100, 60, 200, 20)
