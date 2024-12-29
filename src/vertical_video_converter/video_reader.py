from pathlib import Path
from queue import Queue
from threading import Thread
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class VideoFrame:
    frame: np.ndarray
    frame_number: int

class VideoReader(Thread):
    def __init__(self, video_path: Path, queue: Queue):
        super(VideoReader, self).__init__()
        self.video_path: str = str(video_path)
        self.queue = queue
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_skip = 1

    def run(self):
        cap = self.cap
        current_frame_number = 0
        max_frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # For debugging. Set the current frame number to the given value
        if current_frame_number > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
        while True and current_frame_number <= max_frame_number:
            ret, frame = cap.read()
            if ret:
                self.queue.put(VideoFrame(frame, current_frame_number))
                current_frame_number += self.frame_skip
                if self.frame_skip > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
            else:
                break

        self.stop()

    def set_frame_skip(self, frame_skip: int):
        self.frame_skip = frame_skip

    def stop(self):
        # Clear the queue
        while not self.queue.empty():
            self.queue.get()
        self.queue.put(None)
        self.cap.release()