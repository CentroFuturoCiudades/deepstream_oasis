"""Analytics helpers: FPS tracking and recording state management."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Optional, TYPE_CHECKING

from gi.repository import Gst

if TYPE_CHECKING:  # pragma: no cover
    from .metadata import ClipMetadataWriter


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
        metadata_writer: "ClipMetadataWriter" | None = None,
    ) -> None:
        """Prepare recording state for a camera stream."""
        self.camera_id = camera_id
        self.splitmuxsink = splitmuxsink
        self.config = config
        self.metadata_writer = metadata_writer
        self.recording = False
        self.video_id: Optional[str] = None
        self.video_path: Optional[str] = None
        self.frames_without_detections = 0
        self.recording_frames = 0
        self._clip_basename: Optional[str] = None

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
            if self.metadata_writer:
                self.metadata_writer.finalize_clip(self.video_id, final_path)
        self.video_path = None
        self.video_id = None
        self._clip_basename = None

    def _start_new_clip(self, timestamp: str) -> None:
        """Start a fresh recording segment anchored to the timestamp."""
        # print("new clip")
        self.video_id = str(uuid.uuid4())
        directory = os.path.join("output", str(self.camera_id))
        os.makedirs(directory, exist_ok=True)
        self._clip_basename = timestamp
        self.video_path = os.path.join(directory, f"temp_{timestamp}.mp4")
        self.splitmuxsink.set_property("location", self.video_path)
        self.splitmuxsink.emit("split-now")
        self.recording = True
        self.recording_frames = 0
        if self.metadata_writer and self.video_id:
            self.metadata_writer.start_clip(
                video_id=self.video_id,
                clip_basename=self._clip_basename,
                clip_start_ts=timestamp,
            )

    def _split_clip(self, timestamp: str) -> None:
        """Rotate to a new segment once the max clip length is exceeded."""
        # print("splitting long clip")
        previous_video_id = self.video_id
        previous_path = self.video_path
        if previous_path:
            final_path = previous_path.replace("temp_", "")
            os.rename(previous_path, final_path)
            if self.metadata_writer:
                self.metadata_writer.finalize_clip(previous_video_id, final_path)
        self.video_id = str(uuid.uuid4())
        directory = os.path.join("output", str(self.camera_id))
        self.video_path = os.path.join(directory, f"temp_{timestamp}.mp4")
        self._clip_basename = timestamp
        self.splitmuxsink.set_property("location", self.video_path)
        self.splitmuxsink.emit("split-now")
        self.recording_frames = 0
        if self.metadata_writer and self.video_id:
            self.metadata_writer.start_clip(
                video_id=self.video_id,
                clip_basename=self._clip_basename,
                clip_start_ts=timestamp,
            )

    def finalize_detection_window(self, timestamp: str) -> None:
        """Check whether the active clip should be split due to length."""
        if self.recording and self.recording_frames > self.config.max_recording_frames:
            self._split_clip(timestamp)

    def finalize_clip(self) -> None:
        """Force a clip to close and persist metadata if recording is active."""
        if not self.recording or not self.video_path:
            return
        final_path = self.video_path.replace("temp_", "")
        self.splitmuxsink.set_property("location", "null/dummy.mp4")
        self.splitmuxsink.emit("split-now")
        os.rename(self.video_path, final_path)
        if self.metadata_writer:
            self.metadata_writer.finalize_clip(self.video_id, final_path)
        self.recording = False
        self.video_path = None
        self.video_id = None
        self._clip_basename = None
