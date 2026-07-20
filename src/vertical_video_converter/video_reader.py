"""Background video reading thread."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event, Thread

import cv2
import numpy as np


@dataclass
class VideoFrame:
    frame: np.ndarray
    frame_number: int


class VideoReader(Thread):
    """Reads frames into a bounded queue; puts None when finished."""

    def __init__(self, video_path: Path, queue: Queue, stop_event: Event) -> None:
        super().__init__(daemon=True, name="video-reader")
        self.video_path = str(video_path)
        self.queue = queue
        self.stop_event = stop_event

    def run(self) -> None:
        capture = cv2.VideoCapture(self.video_path)
        try:
            frame_number = 0
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                self.queue.put(VideoFrame(frame, frame_number))
                frame_number += 1
        finally:
            capture.release()
            self.queue.put(None)
