from queue import Queue
from threading import Thread, Event
from pathlib import Path
from typing import List, Tuple
import cv2
from dataclasses import dataclass
import numpy as np
from typing import Optional

def convert_milliseconds_to_hmsx(ms: int) -> str:
    # Convert milliseconds to HH:MM:SS.xxx format
    total_seconds = ms // 1000
    # Remaining milliseconds after converting to seconds
    remaining_ms = ms % 1000
    # Hours calculation
    hours = total_seconds // 3600
    # Remaining seconds after hours calculation for minutes calculation
    remaining_seconds = total_seconds % 3600
    # Minutes calculation
    minutes = remaining_seconds // 60
    # Remaining seconds for seconds display
    seconds = remaining_seconds % 60
    # Format to HH:MM:SS.MS
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(remaining_ms):03}"


@dataclass
class DisplayFrame:
    frame: np.ndarray
    frame_number: int
    time_ms: int = 0


class VideoWriter(Thread):
    def __init__(
        self,
        video_path: Path,
        queue: Queue,
        frame_size: tuple,
        fps: float,
        stopped: Event,
        display_max_height=360,
        show_video=True,
        temp_dir: Optional[Path] = None,

    ):
        super(VideoWriter, self).__init__()
        self.video_path = video_path
        self.queue = queue
        self.frame_size = frame_size
        self.fps = fps
        self.stopped = stopped
        self.display_max_height = display_max_height
        self.show_video = show_video
        self.frame_ranges: List[Tuple[int, int]] = [(-999, -999)]
        self.error_frame_ranges: List[Tuple[int, int]] = [(-999, -999)]
        if temp_dir:
            self.temp_dir = temp_dir
        else:
            self.temp_dir = video_path.parent.parent / "temp_conversion"

    def run(self):
        video_path = str(self.video_path)
        prev_valid_read = None
        out = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_size[0], self.frame_size[1]),
        )
        while True:
            read: DisplayFrame = self.queue.get()
            if read is None:
                break
            frame = read.frame
            try:
                out.write(frame)
                self.add_frame(read.frame_number)
                prev_valid_read = read
            except Exception:
                try:
                    out.write(prev_valid_read.frame)
                    self.add_error_frames(read.frame_number)
                    self.add_frame(read.frame_number)
                    frame = prev_valid_read.frame
                except Exception:
                    pass
                pass

            w, h = frame.shape[1], frame.shape[0]
            if self.show_video:
                if h > self.display_max_height:
                    scale = self.display_max_height / h
                    w = int(w * scale)
                    h = int(h * scale)
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)

                if self.show_video:
                    cv2.imshow("Frame", frame)
                    key_pressed = cv2.waitKey(1) & 0xFF
                    if key_pressed == ord("q"):
                        self.stopped.set()
                        break
        out.release()
        self.stop()

    def stop(self):
        time_ranges = [
            f"{convert_milliseconds_to_hmsx(start*1000/self.fps)}-{convert_milliseconds_to_hmsx(end*1000/self.fps)}"
            for start, end in self.frame_ranges[1:]
            if start < end
        ]
        txt_file = self.video_path.with_suffix(".txt")
        txt_temp = self.temp_dir / f"frame_ranges_{txt_file.name}.txt"
        txt_error_tmp = self.temp_dir / f"error_frame_ranges_{txt_file.name}.txt"
        total_secs = 0

        with open(str(txt_temp), "w") as f:
            for start, end in self.frame_ranges[1:]:
                duraction_sec = (end - start) / self.fps
                total_secs += duraction_sec
                f.write(f"{start}-{end} => {duraction_sec:.2f}\n")

        with open(str(txt_temp), "a") as f:
            f.write(f"Total Duration: {total_secs:.2f}")

        with open(str(txt_error_tmp), "w") as f:
            for start, end in self.error_frame_ranges[1:]:
                duraction_sec = (end - start) / self.fps
                total_secs += duraction_sec
                f.write(
                    f"{start}-{end} => {duraction_sec:.2f} | "
                    f"{convert_milliseconds_to_hmsx(start*1000/self.fps)}-{convert_milliseconds_to_hmsx(end*1000/self.fps)}\n"
                )

        with open(str(txt_file), "w") as f:
            f.write("\n".join(time_ranges))
        # Clear the queue
        while not self.queue.empty():
            self.queue.get()
        self.queue.put(None)

    def add_frame(self, time_ms: int):
        # Add the frame number to the list of video audio ranges
        if self.frame_ranges[-1][1] == time_ms - 1:
            self.frame_ranges[-1] = (self.frame_ranges[-1][0], time_ms)
        else:
            self.frame_ranges.append((time_ms, time_ms))

    def add_error_frames(self, time_ms: int):
        if self.error_frame_ranges[-1][1] == time_ms - 1:
            self.error_frame_ranges[-1] = (self.error_frame_ranges[-1][0], time_ms)
        else:
            self.error_frame_ranges.append((time_ms, time_ms))