# Same content as previous analyzer.py, just renamed the file """Main module for video conversion functionality."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List
import cv2
import numpy as np
import subprocess
import shutil
import os
from insightface.app import FaceAnalysis
from threading import Event
from collections import deque
import re

from queue import Queue
from .video_writer import VideoWriter
from .video_reader import VideoReader
from .video_reader import VideoFrame
from .video_writer import DisplayFrame

@dataclass
class VideoInfo:
    """Video information container."""
    width: int
    height: int
    fps: float


class WeighedAverage:
    # Calculate the weighted average of the given values using np.average
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.values = deque(maxlen=5)

    def add(self, value: tuple) -> None:
        self.values.append(value)

    def average(self) -> tuple:
        if len(self.values) == 0:
            return (0, 0)
        x, y = zip(*self.values)
        return (
            int(np.average(x, weights=np.linspace(1, self.alpha, len(x)))),
            int(np.average(y, weights=np.linspace(1, self.alpha, len(y)))),
        )
    

class VerticalVideo:
    def __init__(
        self,
        video_path: Path,
        app: FaceAnalysis,
        output_path: Path,
        face_threshold=70,
        show_video=True,
    ) -> None:
        self.video_path: Path = video_path
        self.queue = None
        self.video_reader_thread = None
        self.video_writer_thread = None
        self.fa = app
        self.face_threshold = face_threshold
        self.output_queue = None
        self.stop_event = Event()
        self.output_path = output_path
        self.show_video = show_video

    def __del__(self) -> None:
        self.stop_event.set()
        if self.video_reader_thread:
            self.video_reader_thread.join()
        if self.video_writer_thread:
            self.video_writer_thread.join()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass



class VerticalVideoConverter:
    """Class to handle video conversion to vertical format."""

    def __init__(self, model_name: str = "buffalo_l") -> None:
        """Initialize the video converter.

        Args:
            model_name: Name of the face detection model. Defaults to "buffalo_l".
        """
        self._check_ffmpeg()
        self.app = FaceAnalysis(name=model_name, allowed_modules=["detection"], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(256, 256))
        self.temp_dir = None

    def _check_ffmpeg(self) -> None:
        """Check if ffmpeg is installed and accessible."""
        if not shutil.which('ffmpeg'):
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg:\n"
                "- Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "- MacOS: brew install ffmpeg\n"
                "- Windows: download from https://ffmpeg.org/download.html"
            )

    def _get_video_info(self, input_path: Path) -> VideoInfo:
        """Get video information using ffmpeg.
        
        Args:
            input_path: Path to input video
            
        Returns:
            VideoInfo containing width, height, and fps
        """
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-hide_banner'
        ]
        
        try:
            output = subprocess.check_output(
                cmd, 
                stderr=subprocess.STDOUT,
                text=True
            )
        except subprocess.CalledProcessError as e:
            output = e.output

        width = height = fps = None

        # Parse video information
        for line in output.split('\n'):
            if 'Stream' in line and 'Video' in line:
                video_data = line[line.find('Video:'):]
                # Video: h264 (Main) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], 247 kb/s, 25 fps, 25 tbr, 12800 tbn (default)
                resolution_match = re.search(r'(\d{2,})x(\d{2,})(?!\[0x[0-9a-fA-F]+\])', video_data)
                if resolution_match:
                    width = int(resolution_match.group(1))
                    height = int(resolution_match.group(2))

                # Extract fps from various formats
                fps_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:fps|tbr)', video_data) 
                if fps_match:
                    fps = float(fps_match.group(1))

        if not all([width, height, fps]):
            raise ValueError("Could not extract video information")
                        
        return VideoInfo(width=width, height=height, fps=fps)

    def stop(self) -> None:
        self.stop_event.set()
        if self.video_reader_thread:
            self.video_reader_thread.stop()
            self.video_reader_thread.join()
        if self.video_writer_thread:
            self.video_writer_thread.stop()
            self.video_writer_thread.join()
    
    def create_vertical_video(
        self, 
        input_path: str | Path, 
        output_path: str | Path = None, 
        vertical_video_ratio: str = "9/16",
        height_ratio: float = 1.0,         
    ) -> None:
        """Convert horizontal video to vertical format focusing on faces.

        Args:
            input_path: Path to input video file
            output_path: Path to save output video
            target_width: Width of the output vertical video
        """

        input_path = Path(input_path)
        self.input_path = input_path
        video_suffix = input_path.suffix
        sanitized_file_name = input_path.stem.replace("'", "")
        aspect_ratio = vertical_video_ratio.split("/")
        aspect_ratio_width = int(aspect_ratio[0])
        aspect_ratio_height = int(aspect_ratio[1])

        if output_path is None:
            output_path = input_path.parent

        if isinstance(output_path, str):
            output_path = Path(output_path)

        if not output_path.is_dir():
            raise ValueError(f"Output path is not a directory: {output_path}")

        video_output_path = Path(output_path) / f"{sanitized_file_name}_vertical_{aspect_ratio_width}x{aspect_ratio_height}{video_suffix}"
        
        if not input_path.exists():
            raise ValueError(f"Input video not found: {input_path}")
        if not output_path.exists():
            raise ValueError(f"Output directory not found: {output_path.parent}")
        if video_output_path.exists():
            raise ValueError(f"Output file already exists: {output_path}")

        

        video_info = self._get_video_info(input_path)
        target_height = int(video_info.height * height_ratio)
        target_width = int(target_height * aspect_ratio_width / aspect_ratio_height)

        
        if not input_path.exists():
            raise ValueError(f"Input video not found: {input_path}")

        # Create temporary directory for intermediate files
        temp_dir = output_path / "temp_conversion"
        temp_dir.mkdir(exist_ok=True)
        temp_video = temp_dir / f"temp_video{video_suffix}"
        self.temp_dir = temp_dir
        try:
            # Get video properties
            try:
                video_info = self._get_video_info(input_path)
                
                input_queue = Queue(maxsize=50)
                output_queue = Queue(maxsize=50)
                self.stop_event = Event()    
                self.video_reader_thread = VideoReader(input_path, input_queue)
                self.video_writer_thread = VideoWriter(temp_video, output_queue, (target_width, target_height), video_info.fps, self.stop_event, show_video=True)

                self.video_reader_thread.start()
                self.video_writer_thread.start()

                # For smooth tracking
                smooth_x = video_info.width // 2
                smooth_y = video_info.height // 2
                smooth_factor = 0.2
                # Track when we last saw a face
                last_face_time = 0
                face_timeout = 10  # seconds
                
                while True:
                    read:VideoFrame = input_queue.get()
                    if not read:
                        break

                    current_time = read.frame_number / video_info.fps
                    # Detect faces
                    faces = self.app.get(read.frame)
                    
                    # Get main face center
                    if not faces:
                        has_face = False
                    else:
                        main_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                        center_x = int((main_face.bbox[0] + main_face.bbox[2]) // 2)
                        center_y = int((main_face.bbox[1] + main_face.bbox[3]) // 2)
                        has_face = True
                        last_face_time = current_time

                    # Process frame if we have a face or saw one recently
                    if has_face or (current_time - last_face_time) <= face_timeout:
                        if has_face:
                            smooth_x = int(smooth_x * (1 - smooth_factor) + center_x * smooth_factor)
                            smooth_y = int(smooth_y * (1 - smooth_factor) + center_y * smooth_factor)
                        
                        # Calculate crop region
                        crop_width = target_width
                        crop_height = target_height

                        left = max(0, min(video_info.width - crop_width, smooth_x - crop_width // 2))
                        top = max(0, min(video_info.height - crop_height, smooth_y - crop_height // 2))

                        # Crop and resize
                        vertical = read.frame[top:top + crop_height, left:left + crop_width, :]
                        # vertical = cv2.resize(cropped, (target_width, target_height))
                        output_queue.put(DisplayFrame(vertical, read.frame_number, read.frame_number * 1000 / video_info.fps))

                
                self.stop()
            except KeyboardInterrupt:
                print("\nStopping video processing...")
                self.stop()
            except Exception as e:
                print(f"Error in video conversion: {e}")
            
            audio_parts = self._extract_audio()
            self._combine_audio_parts(audio_parts)
            vertical_video_output_path = output_path / f"{self.input_path.stem}_vertical_{vertical_video_ratio.replace('/', '_')}{video_suffix}"
            self._add_audio_to_video(vertical_video_output_path)

            

        finally:
            # Clean up temporary files and directory
            for file in self.temp_dir.glob('*'):
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {file}: {e}")
            
            try:
                self.temp_dir.rmdir()
            except Exception as e:
                print(f"Warning: Could not remove temporary directory {self.temp_dir}: {e}")
            

    def _extract_audio(self) -> List[Path]:
        audio_data_txt = self.temp_dir / "temp_video.txt"
        with open(str(audio_data_txt), "r") as f:
            time_ranges = f.readlines()
        # Convert the time ranges to list of tuples
        time_ranges = [
            tuple(map(lambda x: x.strip(), time.split("-"))) for time in time_ranges
        ]
        print(f"[*] Extracting audio fragments. Total Fragments : {len(time_ranges)}")
        extracted_audio: List[Path] = []
        # Create the command to extract the audio for the time ranges
        for index, time_range in enumerate(time_ranges):
            start, end = time_range
            sanitized_filename = self.input_path.stem.replace("'", "")
            audio_output = self.temp_dir / f"{sanitized_filename}_{index}.mp3"
            command = [
                "ffmpeg",
                "-i",
                str(self.input_path),
                "-ss",
                start,
                "-to",
                end,
                "-avoid_negative_ts",
                "1",
                str(audio_output),
            ]
            extracted_audio.append(audio_output)
            # print(" ".join(command))
            ret = subprocess.run(command, capture_output=True, text=True)
            if ret.returncode != 0:
                print(f"[-] Error extracting audio fragment {index}")
                print(ret.stderr)
            else:
                print(
                    f"[*] Extracted audio for fragment : {index+1}/{len(time_ranges)}",
                    end="\r",
                )

        print("[*] Extracted audio fragments.")
        return extracted_audio
    
    def _combine_audio_parts(self, audio_parts: List[Path]) -> Path:
        output_path = self.temp_dir / "temp_audio.mp3"
        temp_audio_merge_list = self.temp_dir / "temp_audio_merge_list.txt"
        with open(str(temp_audio_merge_list), "w") as f:
            f.write("\n".join([f"file '{audio_path}'" for audio_path in audio_parts]))
        # Join the audio files using ffmpeg
        # ffmpeg -f concat -safe 0 -i filelist.txt -c copy output_audio.mp3
        command = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(temp_audio_merge_list),
            "-c",
            "copy",
            str(output_path),
        ]
        # print(" ".join(command))
        ret = subprocess.run(command, capture_output=True, text=True)
        if ret.returncode != 0:
            print("[-] Error joining audio")
            print(f"[!] Command: {' '.join(command)}")
        else:
            print("[*] Joined audio fragments to single file.")
        # Remove the audio list file
        return output_path
    
    def _add_audio_to_video(self, output_path: Path) -> Path:
        video_path = self.temp_dir / f"temp_video{self.input_path.suffix}"
        audio_path = self.temp_dir / "temp_audio.mp3"
        command = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            str(output_path),
        ]
        ret = subprocess.run(command, capture_output=True, text=True)
        if ret.returncode != 0:
            print("[-] Error merging video and audio")
            print(ret.stderr)
        else:
            print("[*] Merged video and audio")
        return output_path


if __name__ == "__main__":
    # Example usage
    converter = VerticalVideoConverter()
    # Add your test code here 