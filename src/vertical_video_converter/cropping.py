"""Sub-pixel cropping.

Integer-pixel cropping makes panning visibly choppy on low-resolution
sources: 1px of a 360p frame is a large fraction of the crop, so every pan
advances in coarse steps. Sampling the crop at the float camera position
with bilinear interpolation (cv2.getRectSubPix) makes panning equally
smooth at any resolution.

Crispness guard: bilinear sampling at a fractional position softens the
image slightly, which would blur every static shot for no benefit. When the
camera has not moved since the previous frame, the center is snapped to the
nearest position where getRectSubPix samples exactly on the pixel grid,
making it an exact (unfiltered) copy. Note the grid depends on crop-size
parity: sampling starts at ``center - (size - 1) / 2``, so even crop sizes
need a half-pixel center and odd sizes an integer center.
"""

from __future__ import annotations

import cv2
import numpy as np


class SubpixelCropper:
    """Crops a fixed-size window from frames at a float center position."""

    STATIONARY_EPSILON_PX = 0.05

    def __init__(self, frame_width: int, frame_height: int, crop_width: int, crop_height: int) -> None:
        if crop_width > frame_width or crop_height > frame_height:
            raise ValueError("crop must fit inside the frame")
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.crop_width = crop_width
        self.crop_height = crop_height
        # Center offsets that put getRectSubPix sampling exactly on the
        # pixel grid: 0.5 for even crop sizes, 0.0 for odd ones.
        self._grid_offset_x = ((crop_width - 1) / 2) % 1
        self._grid_offset_y = ((crop_height - 1) / 2) % 1
        self._last_center: tuple[float, float] | None = None

    def reset(self) -> None:
        self._last_center = None

    def crop(self, frame: np.ndarray, center_x: float, center_y: float) -> np.ndarray:
        # Keep the window fully inside the frame
        center_x = min(max(center_x, self.crop_width / 2), self.frame_width - self.crop_width / 2)
        center_y = min(max(center_y, self.crop_height / 2), self.frame_height - self.crop_height / 2)

        last = self._last_center
        stationary = (
            last is not None
            and abs(center_x - last[0]) <= self.STATIONARY_EPSILON_PX
            and abs(center_y - last[1]) <= self.STATIONARY_EPSILON_PX
        )
        if stationary:
            # Static shot: snap to the sampling grid = exact copy, no softening
            center_x = round(center_x - self._grid_offset_x) + self._grid_offset_x
            center_y = round(center_y - self._grid_offset_y) + self._grid_offset_y
        self._last_center = (center_x, center_y)

        return cv2.getRectSubPix(frame, (self.crop_width, self.crop_height), (center_x, center_y))
