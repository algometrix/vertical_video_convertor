"""Main-face selection with persistence.

Two findings ported from a production tracker, both aimed at multi-person
frames (two podcast hosts, a panel on stage):

- Sticky selection: re-picking "the best face" fresh every frame flips the
  camera between similar-sized faces as detection scores wobble. Keep the
  incumbent unless a challenger is decisively (50%) stronger.
- Grace hold: brief detection dropouts (a turned head, a hand in front of
  the face) must not move the camera. Hold the last position for a grace
  window before giving up and recentering.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol


class DetectedFace(Protocol):
    """Anything with an insightface-style bbox and detection score."""

    bbox: Sequence[float]  # x1, y1, x2, y2
    det_score: float


def _center(face: DetectedFace) -> tuple[float, float]:
    x1, y1, x2, y2 = face.bbox[:4]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _score(face: DetectedFace) -> float:
    x1, y1, x2, y2 = face.bbox[:4]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area * float(face.det_score)


class FaceTracker:
    """Tracks one primary face across frames."""

    STICKY_TOLERANCE_FRACTION = 0.15  # of frame width: matching the incumbent
    STICKY_SWITCH_RATIO = 1.5  # challenger needs 50% higher score to take over

    def __init__(self, frame_width: int, frame_height: int, hold_seconds: float = 2.0) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if hold_seconds < 0:
            raise ValueError("hold_seconds must be >= 0")
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.hold_seconds = hold_seconds
        self._last_center: tuple[float, float] | None = None
        self._last_seen_at: float | None = None

    def reset(self) -> None:
        """Forget the tracked face (scene cut)."""
        self._last_center = None
        self._last_seen_at = None

    @property
    def has_face(self) -> bool:
        return self._last_center is not None

    def update(self, faces: Sequence[DetectedFace], now: float) -> tuple[float, float] | None:
        """Return the camera target for this frame.

        Returns the tracked face center, the held position during a brief
        dropout, or None once the hold expires (caller decides the fallback,
        typically the frame center).
        """
        if faces:
            face = self._select(faces)
            self._last_center = _center(face)
            self._last_seen_at = now
            return self._last_center

        if self._last_center is not None and self._last_seen_at is not None:
            if now - self._last_seen_at <= self.hold_seconds:
                return self._last_center
            self.reset()
        return None

    def _select(self, faces: Sequence[DetectedFace]) -> DetectedFace:
        best = max(faces, key=_score)
        if self._last_center is None:
            return best

        tolerance = self.frame_width * self.STICKY_TOLERANCE_FRACTION
        incumbent = None
        incumbent_distance = tolerance
        for face in faces:
            cx, cy = _center(face)
            distance = math.hypot(cx - self._last_center[0], cy - self._last_center[1])
            if distance <= incumbent_distance:
                incumbent = face
                incumbent_distance = distance

        if incumbent is None or incumbent is best:
            return best
        if _score(best) >= self.STICKY_SWITCH_RATIO * _score(incumbent):
            return best
        return incumbent
