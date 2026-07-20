"""Camera-target smoothing.

Design notes, distilled from long A/B-measured tuning on a production tracker:

- Always use ``round()`` for float -> int pixel conversions. ``int()``
  truncates toward zero and causes a visible +-1px oscillation whenever the
  smooth position hovers near a ``.5`` boundary.
- Ignore target changes smaller than a jitter deadband. Face detectors wobble
  by a few pixels every frame even on a perfectly still subject.
- Scale responsiveness with distance (slow when close, fast when far) so the
  camera neither vibrates on a seated speaker nor lags a walking one.
- Cap camera speed in absolute terms: one crop-width per second, regardless
  of source fps. Uncapped smoothing reads as whip-pans on 50/60fps sources.
- Keep a final anti-jitter filter on the integer crop coordinates. Rounding
  the smooth position can still alternate between adjacent integers.
"""

from __future__ import annotations

import math


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


class TargetSmoother:
    """Eases the camera center toward the tracked target.

    ``update(x, y)`` feeds a new target; ``update()`` (no target) keeps easing
    toward the last known target, so frames without a detection continue the
    motion instead of freezing and then jumping (the detection-cadence
    sawtooth).

    Refocus band: the camera aim only changes when the raw target escapes a
    band around the current aim. A seated speaker sways and gestures by tens
    of pixels constantly; without the band the crop chases every shift.
    While the target moves inside the band the camera is perfectly still;
    a sustained real move (speaker walks) is followed 1:1 with a fixed lag
    of one band radius.
    """

    REFOCUS_BAND_FRACTION = 0.03  # of frame width: no re-aim inside this radius
    SOFT_ZONE_RADIUS = 0.005  # of frame width: extra damping when this close
    MIN_ALPHA = 0.06  # gentle pursuit when near the target
    MAX_ALPHA = 0.40  # responsive when far
    FAR_THRESHOLD = 0.15  # normalized distance where alpha reaches MAX
    SNAP_DISTANCE_PX = 0.75  # close enough: land exactly on the target

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        crop_width: int,
        fps: float,
        refocus_band_fraction: float | None = None,
    ) -> None:
        if frame_width <= 0 or frame_height <= 0 or crop_width <= 0 or fps <= 0:
            raise ValueError("frame dimensions, crop width and fps must be positive")
        if refocus_band_fraction is None:
            refocus_band_fraction = self.REFOCUS_BAND_FRACTION
        if refocus_band_fraction < 0:
            raise ValueError("refocus_band_fraction must be >= 0")
        self.frame_width = frame_width
        self.frame_height = frame_height
        # One crop-width per second, expressed per frame.
        self.max_step_px = max(1.0, crop_width / fps)
        self.refocus_band_px = frame_width * refocus_band_fraction
        self.x = frame_width / 2.0
        self.y = frame_height / 2.0
        self._held_target: tuple[float, float] = (self.x, self.y)

    def reset(self, x: float, y: float) -> None:
        """Hard-snap the camera (scene cut): no easing across a cut."""
        self.x = float(x)
        self.y = float(y)
        self._held_target = (self.x, self.y)

    def update(self, target_x: float | None = None, target_y: float | None = None) -> tuple[float, float]:
        if target_x is not None and target_y is not None:
            held_x, held_y = self._held_target
            dx = float(target_x) - held_x
            dy = float(target_y) - held_y
            distance = math.hypot(dx, dy)
            if distance > self.refocus_band_px:
                # Pull the aim just enough that the target sits on the band
                # edge: wobble inside the band never moves it, sustained
                # motion tracks with band-radius lag.
                pull = (distance - self.refocus_band_px) / distance
                self._held_target = (held_x + dx * pull, held_y + dy * pull)

        tx, ty = self._held_target
        dx = tx - self.x
        dy = ty - self.y
        distance = math.hypot(dx, dy)

        if distance <= self.SNAP_DISTANCE_PX:
            self.x, self.y = tx, ty
            return self.x, self.y

        normalized = distance / self.frame_width
        ease = 1.0 - (1.0 - min(1.0, normalized / self.FAR_THRESHOLD)) ** 3
        alpha = self.MIN_ALPHA + (self.MAX_ALPHA - self.MIN_ALPHA) * ease
        if normalized < self.SOFT_ZONE_RADIUS:
            alpha *= _smoothstep(normalized / self.SOFT_ZONE_RADIUS)

        step_x = dx * alpha
        step_y = dy * alpha
        step = math.hypot(step_x, step_y)
        if step > self.max_step_px:
            scale = self.max_step_px / step
            step_x *= scale
            step_y *= scale

        self.x += step_x
        self.y += step_y
        return self.x, self.y


class CropSmoother:
    """Final anti-jitter filter on the integer crop origin.

    Even a well-smoothed float position rounds to alternating integers at
    times. Two defenses: small integer moves (<= 2px) are blended gently
    while larger moves pass almost straight through, and the emitted integer
    only changes once the blended position has moved a real fraction of a
    pixel away from it (hysteresis), so an input alternating across a ``.5``
    boundary produces a rock-solid output. Reset on scene cuts so the crop
    can snap.
    """

    SMALL_MOVE_PX = 2
    SMALL_MOVE_BLEND = 0.4  # fraction of the way toward the new position
    LARGE_MOVE_BLEND = 0.97
    OUTPUT_HYSTERESIS_PX = 0.75

    def __init__(self) -> None:
        self._left: float | None = None
        self._top: float | None = None
        self._out_left = 0
        self._out_top = 0

    def reset(self) -> None:
        self._left = None
        self._top = None

    def update(self, left: int, top: int) -> tuple[int, int]:
        if self._left is None or self._top is None:
            self._left, self._top = float(left), float(top)
            self._out_left, self._out_top = left, top
            return left, top

        def blend(previous: float, new: int) -> float:
            delta = new - previous
            factor = self.SMALL_MOVE_BLEND if abs(delta) <= self.SMALL_MOVE_PX else self.LARGE_MOVE_BLEND
            return previous + factor * delta

        self._left = blend(self._left, left)
        self._top = blend(self._top, top)
        if abs(self._left - self._out_left) > self.OUTPUT_HYSTERESIS_PX:
            self._out_left = round(self._left)
        if abs(self._top - self._out_top) > self.OUTPUT_HYSTERESIS_PX:
            self._out_top = round(self._top)
        return self._out_left, self._out_top
