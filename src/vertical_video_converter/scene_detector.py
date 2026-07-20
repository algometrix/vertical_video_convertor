"""Hard scene-cut detection.

Podcast and stage footage switches camera angles constantly. Easing the crop
across a cut looks like a smeared pan; the correct behavior is to detect the
cut and snap. Detection here is deliberately simple - mean absolute
difference between consecutive downscaled grayscale frames - which is
reliable for hard cuts between fixed camera angles.

On a detected cut the caller must reset the face tracker and both smoothers
so no state from the dead shot leaks into the new one (easing from a stale
target drags the camera backward for several frames after the snap).
"""

from __future__ import annotations

import cv2
import numpy as np


class SceneCutDetector:
    """Detects hard cuts between consecutive frames."""

    DOWNSCALE_SIZE = (96, 54)  # (width, height): plenty for a global cut

    def __init__(self, threshold: float = 28.0) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def update(self, frame_bgr: np.ndarray) -> bool:
        """Feed the next frame; returns True when it starts a new shot."""
        small = cv2.resize(frame_bgr, self.DOWNSCALE_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        previous = self._previous
        self._previous = gray
        if previous is None:
            return False

        mean_diff = float(np.mean(cv2.absdiff(gray, previous)))
        return mean_diff >= self.threshold
