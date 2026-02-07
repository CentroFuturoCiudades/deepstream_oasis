"""Analytics helpers: FPS tracking and recording state management."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Optional

from gi.repository import Gst


class FPSTracker:
    """Tracks instantaneous and average FPS for a stream."""

    def __init__(self, stream_id: int):
        """Initialize counters and locks for the specified stream."""
        self.stream_id = stream_id
        self.start_time = time.time()
        self.frame_count = 0
        self.total_frames = 0
        self.total_time = 0.0
        self.initialized = False
        self.lock = Lock()

    def update(self) -> None:
        """Accumulate a processed frame to update instantaneous FPS."""
        with self.lock:
            if not self.initialized:
                self.start_time = time.time()
                self.initialized = True
            else:
                self.frame_count += 1

    def get_fps(self) -> tuple[float, float]:
        """Compute the current and average FPS metrics."""
        with self.lock:
            elapsed = time.time() - self.start_time
            if elapsed <= 0:
                return 0.0, 0.0
            current = self.frame_count / elapsed
            self.total_time += elapsed
            self.total_frames += self.frame_count
            average = (
                self.total_frames / self.total_time if self.total_time > 0 else 0.0
            )
            self.start_time = time.time()
            self.frame_count = 0
            return current, average

    def print_callback(self) -> bool:
        """Log FPS metrics on a GLib timeout signal."""
        if self.initialized:
            current, average = self.get_fps()
            print(f"[Stream {self.stream_id}] FPS: {current:.2f} (avg: {average:.2f})")
        return True


@dataclass
class RecordingConfig:
    max_recording_frames: int
    max_no_detection_frames: int


class RecordingManager:
    """Handles clip creation and termination based on detections."""

    def __init__(
        self,
        camera_id: str,
        splitmuxsink: Gst.Element,
        config: RecordingConfig,
    ) -> None:
        """Prepare recording state for a camera stream."""
        self.camera_id = camera_id
        self.splitmuxsink = splitmuxsink
        self.config = config
        self.recording = False
        self.video_id: Optional[str] = None
        self.video_path: Optional[str] = None
        self.frames_without_detections = 0
        self.recording_frames = 0

    def on_detections(self, timestamp: str) -> None:
        """Update recording state when detections are present."""
        self.frames_without_detections = 0
        self.recording_frames += 1

        if not self.recording:
            self._start_new_clip(timestamp)

    def on_no_detections(self) -> None:
        """Track consecutive empty frames and finish clips if needed."""
        self.frames_without_detections += 1
        if self.frames_without_detections < self.config.max_no_detection_frames:
            return
        if not self.recording:
            return

        # print("end clip")
        self.splitmuxsink.set_property("location", "null/dummy.mp4")
        self.splitmuxsink.emit("split-now")
        self.recording = False
        if self.video_path:
            final_path = self.video_path.replace("temp_", "")
            os.rename(self.video_path, final_path)
        self.video_path = None
        self.video_id = None

    def _start_new_clip(self, timestamp: str) -> None:
        """Start a fresh recording segment anchored to the timestamp."""
        #print("new clip")
        self.video_id = str(uuid.uuid4())
        directory = os.path.join("output", str(self.camera_id))
        self.video_path = os.path.join(directory, f"temp_{timestamp}.mp4")
        self.splitmuxsink.set_property("location", self.video_path)
        self.splitmuxsink.emit("split-now")
        self.recording = True
        self.recording_frames = 0

    def _split_clip(self, timestamp: str) -> None:
        """Rotate to a new segment once the max clip length is exceeded."""
        #print("splitting long clip")
        if self.video_path:
            final_path = self.video_path.replace("temp_", "")
            os.rename(self.video_path, final_path)
        self.video_id = str(uuid.uuid4())
        directory = os.path.join("output", str(self.camera_id))
        self.video_path = os.path.join(directory, f"temp_{timestamp}.mp4")
        self.splitmuxsink.set_property("location", self.video_path)
        self.splitmuxsink.emit("split-now")
        self.recording_frames = 0

    def finalize_detection_window(self, timestamp: str) -> None:
        """Check whether the active clip should be split due to length."""
        if self.recording and self.recording_frames > self.config.max_recording_frames:
            self._split_clip(timestamp)
