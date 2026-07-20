"""Face-tracked horizontal-to-vertical video conversion."""

from .cropping import SubpixelCropper
from .face_tracker import FaceTracker
from .scene_detector import SceneCutDetector
from .smoothing import CropSmoother, TargetSmoother

__version__ = "1.0.0"
__all__ = [
    "CropSmoother",
    "FaceTracker",
    "SceneCutDetector",
    "SubpixelCropper",
    "TargetSmoother",
    "VerticalVideoConverter",
]


def __getattr__(name: str):
    # Lazy import: insightface takes seconds to import and is not needed
    # for the pure tracking/smoothing utilities.
    if name == "VerticalVideoConverter":
        from .converter import VerticalVideoConverter

        return VerticalVideoConverter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
