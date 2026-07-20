"""Background video writing thread with optional live preview."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Thread

import cv2
import numpy as np

PREVIEW_MAX_HEIGHT = 400
PREVIEW_GAP_PX = 2


def scale_to_max_height(frame: np.ndarray, max_height: int = PREVIEW_MAX_HEIGHT) -> np.ndarray:
    """Downscale to at most max_height, width following the aspect ratio."""
    height, width = frame.shape[:2]
    if height <= max_height:
        return frame
    scale = max_height / height
    return cv2.resize(frame, (round(width * scale), max_height), interpolation=cv2.INTER_AREA)


def compose_side_by_side(
    left: np.ndarray, right: np.ndarray, max_height: int = PREVIEW_MAX_HEIGHT
) -> np.ndarray:
    """Original-vs-converted comparison image, each capped at max_height."""
    a = scale_to_max_height(left, max_height)
    b = scale_to_max_height(right, max_height)
    height = max(a.shape[0], b.shape[0])

    def pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == height:
            return img
        out = np.zeros((height, img.shape[1], 3), dtype=img.dtype)
        top = (height - img.shape[0]) // 2
        out[top : top + img.shape[0]] = img
        return out

    gap = np.zeros((height, PREVIEW_GAP_PX, 3), dtype=a.dtype)
    return np.hstack([pad(a), gap, pad(b)])


@dataclass
class OutputItem:
    frame: np.ndarray  # written to the video file
    preview: np.ndarray | None = None  # shown in the preview window, if enabled


class VideoWriter(Thread):
    """Writes frames from a queue; a None item ends the stream."""

    def __init__(
        self,
        video_path: Path,
        queue: Queue,
        frame_size: tuple[int, int],
        fps: float,
        stop_event: Event,
        show_preview: bool = False,
        preview_max_height: int = PREVIEW_MAX_HEIGHT,
    ) -> None:
        super().__init__(daemon=True, name="video-writer")
        self.video_path = str(video_path)
        self.queue = queue
        self.frame_size = frame_size
        self.fps = fps
        self.stop_event = stop_event
        self.show_preview = show_preview
        self.preview_max_height = preview_max_height
        self.frames_written = 0

    def run(self) -> None:
        writer = cv2.VideoWriter(
            self.video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self.frame_size,
        )
        if not writer.isOpened():
            self.stop_event.set()
            raise RuntimeError(f"Could not open video writer for {self.video_path}")
        try:
            while True:
                item: OutputItem | None = self.queue.get()
                if item is None:
                    break
                writer.write(item.frame)
                self.frames_written += 1
                if self.show_preview and item.preview is not None and not self._show(item.preview):
                    self.stop_event.set()
                    break
        finally:
            writer.release()
            if self.show_preview:
                cv2.destroyAllWindows()

    def _show(self, frame: np.ndarray) -> bool:
        frame = scale_to_max_height(frame, self.preview_max_height)
        cv2.imshow("vertical-video-converter (q to quit)", frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")
