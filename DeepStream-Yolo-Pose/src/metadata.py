"""Clip metadata writer for local sidecar JSON files."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _round_bbox(bbox: Sequence[float]) -> List[float]:
    return [float(round(coord, 4)) for coord in bbox]


@dataclass
class ClipMetadata:
    clip_basename: str
    clip_start_ts: str
    directory: Path
    fps: float
    camera_id: str
    tracks: Dict[str, List[dict]] = field(
        default_factory=lambda: defaultdict(list)
    )  # track_id -> samples


class ClipMetadataWriter:
    """Accumulates per-clip detections and writes a JSON sidecar on completion."""

    def __init__(
        self, camera_id: str, output_root: str = "output", fps: float = 30.0
    ) -> None:
        self._camera_id = camera_id
        self._output_root = Path(output_root)
        self._fps = fps
        self._clips: Dict[str, ClipMetadata] = {}

    def start_clip(self, video_id: str, clip_basename: str, clip_start_ts: str) -> None:
        """Allocate an in-memory clip slot when recording begins."""
        directory = self._output_root / str(self._camera_id)
        directory.mkdir(parents=True, exist_ok=True)
        self._clips[video_id] = ClipMetadata(
            clip_basename=clip_basename,
            clip_start_ts=clip_start_ts,
            directory=directory,
            fps=self._fps,
            camera_id=str(self._camera_id),
        )

    def record_detection(
        self,
        video_id: Optional[str],
        track_id: Optional[int],
        person_id: Optional[str],
        timestamp: str,
        bbox: Sequence[float],
    ) -> None:
        """Append a detection sample for the active clip."""
        if not video_id or track_id is None:
            return
        clip = self._clips.get(video_id)
        if not clip:
            return
        clip.tracks[str(track_id)].append(
            {
                "person_id": person_id,
                "timestamp": timestamp,
                "bbox": _round_bbox(bbox),
            }
        )

    def finalize_clip(
        self, video_id: Optional[str], final_video_path: Optional[str]
    ) -> Optional[Path]:
        """Persist accumulated metadata into a sidecar JSON file."""
        if not video_id:
            return None
        clip = self._clips.pop(video_id, None)
        if not clip:
            return None

        final_path = (
            Path(final_video_path)
            if final_video_path
            else clip.directory / f"{clip.clip_basename}.mp4"
        )
        clip_name = final_path.stem.replace("ready_", "")
        payload = {
            "clip_name": clip_name,
            "clip_start_ts": clip.clip_start_ts,
            "camera_id": clip.camera_id,
            "fps": clip.fps,
            "tracks": clip.tracks,
        }
        json_path = clip.directory / f"{clip_name}.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return json_path

    def discard_clip(self, video_id: Optional[str]) -> None:
        """Drop any cached metadata for the in-flight clip."""
        if not video_id:
            return
        self._clips.pop(video_id, None)
