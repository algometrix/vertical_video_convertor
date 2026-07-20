"""Horizontal-to-vertical video conversion driven by face tracking.

Pipeline:

    VideoReader (thread) -> face detection (InsightFace) -> FaceTracker
        -> TargetSmoother -> crop -> CropSmoother -> VideoWriter (thread)
        -> ffmpeg audio mux

Built for talking-head content (podcasts, interviews, stage talks): one
primary subject per shot, mostly sitting or standing, hard cuts between
camera angles.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event

from insightface.app import FaceAnalysis

from .cropping import SubpixelCropper
from .face_tracker import FaceTracker
from .scene_detector import SceneCutDetector
from .smoothing import TargetSmoother
from .video_reader import VideoFrame, VideoReader
from .video_writer import OutputItem, VideoWriter, compose_side_by_side

QUEUE_SIZE = 50
PROGRESS_EVERY_FRAMES = 30


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    has_audio: bool


def _require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise RuntimeError(
                f"{tool} not found on PATH. Install ffmpeg:\n"
                "- Ubuntu/Debian: sudo apt install ffmpeg\n"
                "- macOS:         brew install ffmpeg\n"
                "- Windows:       winget install Gyan.FFmpeg"
            )


def _probe_video(input_path: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(input_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {input_path}: {result.stderr.strip()}")

    streams = json.loads(result.stdout).get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"No video stream found in {input_path}")

    numerator, _, denominator = video.get("avg_frame_rate", "0/1").partition("/")
    fps = float(numerator) / float(denominator or 1) if float(denominator or 1) else 0.0
    if fps <= 0:
        raise ValueError(f"Could not determine fps for {input_path}")

    try:
        frame_count = int(video.get("nb_frames", 0))
    except (TypeError, ValueError):
        frame_count = 0

    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        frame_count=frame_count,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def _even(value: float) -> int:
    return max(2, round(value / 2) * 2)


class VerticalVideoConverter:
    """Converts horizontal video to a face-tracked vertical crop."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        det_size: int = 640,
        use_gpu: bool = True,
    ) -> None:
        _require_ffmpeg()
        providers = ["CPUExecutionProvider"]
        if use_gpu:
            providers.insert(0, "CUDAExecutionProvider")
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection"],
            providers=providers,
        )
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))

    def create_vertical_video(
        self,
        input_path: str | Path,
        output_dir: str | Path | None = None,
        aspect_ratio: str = "9/16",
        height_ratio: float = 1.0,
        headroom: float = 0.42,
        hold_seconds: float = 2.0,
        scene_cut_threshold: float = 28.0,
        refocus_band: float | None = None,
        show_preview: bool = False,
        compare_preview: bool = False,
    ) -> Path:
        """Convert one video; returns the output file path.

        Args:
            input_path: Source video.
            output_dir: Output directory (defaults to the source's directory).
            aspect_ratio: Output aspect as "W/H", e.g. "9/16".
            height_ratio: Crop height as a fraction of source height.
            headroom: Vertical position of the face in the crop
                (0.5 = centered, smaller = closer to the top).
            hold_seconds: How long to hold the last face position when
                detection drops out before recentering.
            scene_cut_threshold: Sensitivity of hard-cut detection
                (lower = more sensitive).
            refocus_band: The camera does not re-aim while the face center
                stays within this fraction of frame width of the current
                aim (default 0.03). Larger = calmer camera; 0 disables.
            show_preview: Show a live preview window (press q to stop).
            compare_preview: Preview the original and the converted frame
                side by side (implies show_preview).
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        if not 0.0 < headroom <= 1.0:
            raise ValueError("headroom must be in (0, 1]")
        if not 0.0 < height_ratio <= 1.0:
            raise ValueError("height_ratio must be in (0, 1]")

        output_dir = Path(output_dir) if output_dir is not None else input_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        ratio_w, _, ratio_h = aspect_ratio.partition("/")
        try:
            ratio = int(ratio_w) / int(ratio_h)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(f'aspect_ratio must look like "9/16", got "{aspect_ratio}"') from error

        info = _probe_video(input_path)
        crop_h = _even(min(info.height, info.height * height_ratio))
        crop_w = _even(crop_h * ratio)
        if crop_w > info.width:
            raise ValueError(
                f"Crop {crop_w}x{crop_h} is wider than the source ({info.width}px); "
                "lower height_ratio or pick a narrower aspect_ratio"
            )

        output_path = output_dir / f"{input_path.stem}_vertical_{ratio_w}x{ratio_h}.mp4"
        if output_path.exists():
            raise FileExistsError(f"Output already exists: {output_path}")

        with tempfile.TemporaryDirectory(prefix="vvc_", dir=output_dir) as temp_dir:
            temp_video = Path(temp_dir) / f"video_only{output_path.suffix}"
            self._process_frames(
                input_path,
                temp_video,
                info,
                crop_w,
                crop_h,
                headroom,
                hold_seconds,
                scene_cut_threshold,
                refocus_band,
                show_preview or compare_preview,
                compare_preview,
            )
            self._mux_audio(temp_video, input_path, output_path, info.has_audio)

        print(f"[*] Done: {output_path}")
        return output_path

    def _process_frames(
        self,
        input_path: Path,
        temp_video: Path,
        info: VideoInfo,
        crop_w: int,
        crop_h: int,
        headroom: float,
        hold_seconds: float,
        scene_cut_threshold: float,
        refocus_band: float | None,
        show_preview: bool,
        compare_preview: bool,
    ) -> None:
        stop_event = Event()
        input_queue: Queue = Queue(maxsize=QUEUE_SIZE)
        output_queue: Queue = Queue(maxsize=QUEUE_SIZE)
        reader = VideoReader(input_path, input_queue, stop_event)
        writer = VideoWriter(
            temp_video,
            output_queue,
            (crop_w, crop_h),
            info.fps,
            stop_event,
            show_preview=show_preview,
        )

        tracker = FaceTracker(info.width, info.height, hold_seconds=hold_seconds)
        smoother = TargetSmoother(
            info.width, info.height, crop_w, info.fps, refocus_band_fraction=refocus_band
        )
        cropper = SubpixelCropper(info.width, info.height, crop_w, crop_h)
        scene_detector = SceneCutDetector(threshold=scene_cut_threshold)
        # Face sits at `headroom` of crop height; camera center compensates.
        headroom_shift = (0.5 - headroom) * crop_h
        frame_center = (info.width / 2.0, info.height / 2.0)

        reader.start()
        writer.start()
        started_at = time.perf_counter()
        processed = 0
        try:
            while not stop_event.is_set():
                item: VideoFrame | None = input_queue.get()
                if item is None:
                    break

                now = item.frame_number / info.fps
                faces = self.app.get(item.frame)
                target = tracker.update(faces, now)

                if scene_detector.update(item.frame):
                    # Hard cut: drop all state from the dead shot and snap.
                    tracker.reset()
                    target = tracker.update(faces, now) or frame_center
                    smoother.reset(target[0], target[1] + headroom_shift)
                    cropper.reset()
                elif target is None:
                    target = frame_center

                smooth_x, smooth_y = smoother.update(target[0], target[1] + headroom_shift)
                # Sub-pixel crop: pans glide smoothly even on low-res sources
                cropped = cropper.crop(item.frame, smooth_x, smooth_y)
                preview = None
                if show_preview:
                    preview = compose_side_by_side(item.frame, cropped) if compare_preview else cropped
                out_item = OutputItem(cropped, preview)
                while not stop_event.is_set():
                    try:
                        output_queue.put(out_item, timeout=1.0)
                        break
                    except Full:
                        continue
                processed += 1
                if processed % PROGRESS_EVERY_FRAMES == 0:
                    self._print_progress(processed, info.frame_count, started_at)
        except KeyboardInterrupt:
            print("\n[!] Interrupted, finishing the frames written so far...")
        finally:
            stop_event.set()
            # Writer may already be gone (preview quit); don't block on it
            with contextlib.suppress(Full):
                output_queue.put(None, timeout=1.0)
            writer.join()
            # Drain the input queue so a reader blocked on a full queue can exit
            while reader.is_alive():
                with contextlib.suppress(Empty):
                    input_queue.get(timeout=0.1)
            reader.join()
            print(
                f"\n[*] Processed {processed} frames "
                f"({processed / max(1e-6, time.perf_counter() - started_at):.1f} fps)"
            )
        if writer.frames_written == 0:
            raise RuntimeError("No frames were written; conversion failed")

    @staticmethod
    def _print_progress(processed: int, total: int, started_at: float) -> None:
        rate = processed / max(1e-6, time.perf_counter() - started_at)
        if total > 0:
            message = f"[*] {processed}/{total} frames ({100 * processed / total:.1f}%) at {rate:.1f} fps"
        else:
            message = f"[*] {processed} frames at {rate:.1f} fps"
        print(message, end="\r", file=sys.stderr)

    @staticmethod
    def _mux_audio(temp_video: Path, source: Path, output_path: Path, has_audio: bool) -> None:
        if not has_audio:
            shutil.move(str(temp_video), str(output_path))
            return
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_video),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio mux failed: {result.stderr.strip()[-500:]}")
