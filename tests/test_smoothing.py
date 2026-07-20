"""Tests for TargetSmoother and CropSmoother."""

from vertical_video_converter.smoothing import CropSmoother, TargetSmoother


def make_smoother(frame_w=1920, frame_h=1080, crop_w=608, fps=30.0):
    return TargetSmoother(frame_w, frame_h, crop_w, fps)


class TestTargetSmoother:
    def test_starts_at_frame_center(self):
        s = make_smoother()
        assert (s.x, s.y) == (960.0, 540.0)

    def test_moves_toward_target(self):
        s = make_smoother()
        x1, _ = s.update(1500, 540)
        x2, _ = s.update(1500, 540)
        assert 960 < x1 < x2 < 1500

    def test_jitter_within_deadband_is_ignored(self):
        s = make_smoother()
        s.reset(960, 540)
        # Detector wobble of ~1px must not move the camera at all
        for wobble in (961, 959, 960.5, 959.5):
            x, y = s.update(wobble, 540)
            assert (x, y) == (960.0, 540.0)

    def test_sway_within_refocus_band_never_moves_camera(self):
        s = make_smoother()
        s.reset(960, 540)
        band = s.refocus_band_px
        # Speaker sways well inside the band: camera stays parked
        for offset in (0.9 * band, -0.8 * band, 0.5 * band, -0.9 * band):
            x, y = s.update(960 + offset, 540)
            assert (x, y) == (960.0, 540.0)

    def test_return_inside_band_causes_no_walk_back(self):
        s = make_smoother()
        s.reset(960, 540)
        band = s.refocus_band_px
        # Face escapes the band; let the camera settle on the new aim
        for _ in range(100):
            s.update(960 + band + 20, 540)
        settled = s.x
        # Face returns inside the band: camera must not walk back toward it
        # (sub-pixel soft-zone creep toward the held aim is fine; it vanishes
        # at the integer crop stage)
        for _ in range(10):
            x, _ = s.update(960, 540)
            assert abs(x - settled) < 0.5
            assert x >= settled - 1e-9

    def test_sustained_move_tracks_with_band_lag(self):
        s = make_smoother()
        s.reset(960, 540)
        band = s.refocus_band_px
        target = 960.0
        positions = []
        for _ in range(200):
            target += 4.0  # slow walk across the frame
            x, _ = s.update(target, 540)
            positions.append(x)
        # Steady state: camera moves at target speed...
        tail_speed = (positions[-1] - positions[-31]) / 30
        assert 3.5 <= tail_speed <= 4.5
        # ...lagging by at least the band but by a bounded amount
        assert positions[-1] <= target - band
        assert target - positions[-1] <= band + 150

    def test_zero_band_follows_target_directly(self):
        s = TargetSmoother(1920, 1080, 608, 30.0, refocus_band_fraction=0.0)
        s.reset(960, 540)
        x1, _ = s.update(1000, 540)
        assert x1 > 960

    def test_speed_capped_to_one_crop_width_per_second(self):
        s = make_smoother(crop_w=608, fps=30.0)
        max_step = 608 / 30.0
        previous = s.x
        for _ in range(10):
            x, _ = s.update(1900, 540)
            assert x - previous <= max_step + 1e-9
            previous = x

    def test_snaps_when_very_close(self):
        s = make_smoother()
        s.reset(1000, 500)
        x, y = s.update(1000.5, 500)
        assert (x, y) == (1000.5, 500.0) or (x, y) == (1000.0, 500.0)

    def test_reset_is_instant(self):
        s = make_smoother()
        s.update(1500, 700)
        s.reset(200, 300)
        assert (s.x, s.y) == (200.0, 300.0)

    def test_no_target_keeps_easing_toward_held_target(self):
        s = make_smoother()
        x1, _ = s.update(1500, 540)
        x2, _ = s.update()  # non-detection frame
        assert x2 > x1

    def test_rejects_invalid_dimensions(self):
        import pytest

        with pytest.raises(ValueError):
            TargetSmoother(0, 1080, 608, 30.0)
        with pytest.raises(ValueError):
            TargetSmoother(1920, 1080, 608, 0.0)


class TestCropSmoother:
    def test_first_position_passes_through(self):
        c = CropSmoother()
        assert c.update(100, 50) == (100, 50)

    def test_one_px_oscillation_is_fully_suppressed(self):
        c = CropSmoother()
        c.update(100, 50)
        positions = [c.update(v, 50)[0] for v in (101, 100, 101, 100, 101, 100)]
        assert positions == [100] * 6

    def test_slow_pan_still_tracks(self):
        c = CropSmoother()
        c.update(100, 50)
        outputs = [c.update(v, 50)[0] for v in range(101, 121)]
        # Lags a little but must clearly follow the pan
        assert outputs[-1] >= 118

    def test_large_moves_pass_almost_through(self):
        c = CropSmoother()
        c.update(100, 50)
        left, _ = c.update(200, 50)
        assert left >= 195

    def test_reset_allows_snap(self):
        c = CropSmoother()
        c.update(100, 50)
        c.reset()
        assert c.update(900, 400) == (900, 400)
