"""Background worker that classifies local tracklets before upload."""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .age_group_classifier.src.inference.api import AgeGroupClassifier
from .send_helper import SendHelper

VIDEO_FPS = 30.0
READY_PREFIX = "ready_"
MODEL_VERSION = "mean_std_v1.0"
CLASSIFIER_DEBUG = False
IMAGE_DEBUG = False


class TrackletClassificationWorker:
    """Polls clip folders, classifies tracklets, and emits Event Hub updates."""

    def __init__(
        self,
        camera_ids: Sequence[str],
        base_output_dir: str = "output",
        poll_interval: float = 2.0,
        demo_cfg_path: Optional[str] = None,
        model_cfg_path: Optional[str] = None,
        classifier_root: Optional[str] = None,
        debug: Optional[bool] = None,
    ) -> None:
        self._camera_ids = list(camera_ids)
        self._base_output = Path(base_output_dir)
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._threads: Dict[str, threading.Thread] = {}
        self._debug = debug if debug is not None else CLASSIFIER_DEBUG
        self._image_debug = IMAGE_DEBUG
        self._classifier_lock = threading.Lock()
        self._send_lock = threading.Lock()

        classifier_base = (
            Path(classifier_root).expanduser().resolve()
            if classifier_root
            else Path(__file__).resolve().parent / "age_group_classifier"
        )
        configs_dir = classifier_base / "configs"
        demo_cfg = demo_cfg_path or str(configs_dir / "demo.yaml")
        model_cfg = model_cfg_path or str(configs_dir / "model.yaml")

        self._classifier = AgeGroupClassifier(
            demo_cfg_path=demo_cfg,
            model_cfg_path=model_cfg,
            classifier_path=classifier_base,
        )
        self._send_helper = SendHelper()

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._threads:
            return
        self._stop_event.clear()
        for camera_id in self._camera_ids:
            thread = threading.Thread(
                target=self._run_camera,
                args=(camera_id,),
                daemon=True,
                name=f"TrackletClassifier-{camera_id}",
            )
            thread.start()
            self._threads[camera_id] = thread

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._threads.values():
            if thread.is_alive():
                thread.join(timeout=5.0)
        self._threads.clear()
        self._send_helper.close()

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    def _run_camera(self, camera_id: str) -> None:
        while not self._stop_event.is_set():
            processed = self._process_camera(camera_id)
            if not processed:
                self._stop_event.wait(self._poll_interval)

    def _process_camera(self, camera_id: str) -> bool:
        camera_dir = self._base_output / str(camera_id)
        if not camera_dir.exists():
            return False

        processed_any = False
        for json_path in sorted(camera_dir.glob("*.json")):
            clip_base = json_path.stem
            video_path = camera_dir / f"{clip_base}.mp4"
            if not video_path.exists():
                continue
            print(f"[Classifier] Found clip '{video_path.name}' for camera {camera_id}")
            try:
                self._handle_clip(camera_id, video_path, json_path)
                processed_any = True
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Classification worker error for {json_path}: {exc}")
        return processed_any

    # ------------------------------------------------------------------
    # Clip processing
    # ------------------------------------------------------------------
    def _handle_clip(self, camera_id: str, video_path: Path, json_path: Path) -> None:
        with open(json_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        clip_start_str = metadata.get("clip_start_ts") or metadata.get("clip_name")
        if not clip_start_str:
            raise ValueError(f"Missing clip_start_ts in {json_path}")
        clip_start_ts = self._parse_timestamp(clip_start_str)

        tracks: Dict[str, List[dict]] = metadata.get("tracks", {})
        frame_plan, track_person = self._prepare_frame_plan(tracks, clip_start_ts)
        if not frame_plan:
            print(f"[Classifier] No usable detections in '{video_path.name}', marking ready")
            self._finalize_clip(video_path, json_path)
            return

        open_start = time.time()
        cap = cv2.VideoCapture(str(video_path))
        if self._debug:
            open_ms = (time.time() - open_start) * 1000
            print(
                f"[Classifier][Debug] Open video {video_path.name} in {open_ms:.1f} ms"
            )
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video {video_path}")

        try:
            track_crops = self._extract_crops_for_tracks(
                cap, frame_plan, camera_id, debug=self._debug
            )
            for track_id, crops in track_crops.items():
                if not crops:
                    continue
                person_id = track_person.get(track_id)
                if not person_id:
                    continue
                track_start = time.time()
                if self._debug:
                    print(
                        f"[Classifier][Debug] Track {track_id} prepared {len(crops)} crops"
                    )
                try:
                    with self._classifier_lock:
                        label, confidence = self._classifier.classify_from_crops(
                            crops, debug=self._debug
                        )
                except ValueError as exc:
                    print(f"Track {track_id} skipped: {exc}")
                    continue
                print(
                    f"[Classifier] Track {track_id} ({person_id}) -> {label}"
                    f" ({confidence:.2f})"
                )
                with self._send_lock:
                    self._send_helper.send_person_observed(
                        person_id=person_id,
                        age_group=label.lower(),
                        confidence=confidence,
                        model_version=MODEL_VERSION,
                    )
                if self._debug:
                    total_ms = (time.time() - track_start) * 1000
                    print(
                        f"[Classifier][Debug] Track {track_id} total time {total_ms:.1f} ms"
                    )
        finally:
            cap.release()

        self._finalize_clip(video_path, json_path)

    def _finalize_clip(self, video_path: Path, json_path: Path) -> None:
        try:
            json_path.unlink()
        except FileNotFoundError:
            pass

        ready_path = video_path.with_name(f"{READY_PREFIX}{video_path.name}")
        try:
            if ready_path.exists():
                ready_path.unlink()
            video_path.rename(ready_path)
            print(f"[Classifier] Clip '{video_path.name}' marked ready for upload")
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_person_id(detections: List[dict]) -> Optional[str]:
        for det in detections:
            person_id = det.get("person_id")
            if person_id:
                return str(person_id)
        return None

    def _prepare_frame_plan(
        self,
        tracks: Dict[str, List[dict]],
        clip_start_ts: datetime,
    ) -> Tuple[List[Tuple[int, List[Dict[str, Any]]]], Dict[str, Optional[str]]]:
        frame_map: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        track_person: Dict[str, Optional[str]] = {}
        for track_id, detections in tracks.items():
            sorted_dets = sorted(
                detections,
                key=lambda item: item.get("timestamp", ""),
            )
            track_person[track_id] = self._extract_person_id(sorted_dets)
            for det in sorted_dets:
                timestamp_str = det.get("timestamp")
                bbox = det.get("bbox")
                if not timestamp_str or not bbox:
                    continue
                frame_index = self._timestamp_to_frame(timestamp_str, clip_start_ts)
                if frame_index < 0:
                    continue
                frame_map[frame_index].append(
                    {
                        "track_id": track_id,
                        "bbox": bbox,
                    }
                )
        frame_plan = sorted(frame_map.items(), key=lambda item: item[0])
        return frame_plan, track_person

    def _extract_crops_for_tracks(
        self,
        cap: cv2.VideoCapture,
        frame_plan: List[Tuple[int, List[Dict[str, Any]]]],
        camera_id: str,
        debug: bool = False,
    ) -> Dict[str, List[np.ndarray]]:
        track_crops: Dict[str, List[np.ndarray]] = defaultdict(list)
        if not frame_plan:
            return {}

        start_time = time.time()
        frames_processed = 0
        current_index = -1
        last_frame: Optional[np.ndarray] = None

        first_frame = max(frame_plan[0][0], 0)
        if first_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        current_index = first_frame - 1

        for frame_index, requests in frame_plan:
            if frame_index <= current_index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                current_index = frame_index - 1
            while current_index < frame_index:
                success, last_frame = cap.read()
                if not success or last_frame is None:
                    if debug:
                        print("[Classifier][Debug] Failed to read frame during extraction")
                    return {track_id: list(crops) for track_id, crops in track_crops.items()}
                current_index += 1
            if last_frame is None:
                continue
            frames_processed += 1
            for req in requests:
                crop = self._crop_frame(last_frame, req["bbox"])
                if crop.size:
                    track_id = req["track_id"]
                    track_crops[track_id].append(crop)
                    if self._image_debug:
                        self._save_crop_image(
                            camera_id,
                            track_id,
                            crop,
                            len(track_crops[track_id]),
                        )

        if debug:
            duration_ms = (time.time() - start_time) * 1000
            total_crops = sum(len(crops) for crops in track_crops.values())
            print(
                f"[Classifier][Debug] Extracted {total_crops} crops from {frames_processed} frames in {duration_ms:.1f} ms"
            )

        return {track_id: list(crops) for track_id, crops in track_crops.items()}

    def _save_crop_image(
        self,
        camera_id: str,
        track_id: str,
        crop: np.ndarray,
        index: int,
    ) -> None:
        track_dir = self._base_output / str(camera_id) / track_id
        track_dir.mkdir(parents=True, exist_ok=True)
        filename = track_dir / f"crop_{index:04d}.jpg"
        cv2.imwrite(str(filename), crop)

    def _timestamp_to_frame(self, timestamp_str: str, clip_start: datetime) -> int:
        det_ts = self._parse_timestamp(timestamp_str)
        delta = det_ts - clip_start
        total_seconds = max(delta.total_seconds(), 0.0)
        return int(total_seconds * VIDEO_FPS)

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        return datetime.fromisoformat(value)

    @staticmethod
    def _crop_frame(frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        x, y, w, h = bbox
        height, width = frame.shape[:2]
        x1 = max(int(x), 0)
        y1 = max(int(y), 0)
        x2 = min(int(x + w), width - 1)
        y2 = min(int(y + h), height - 1)
        if x2 <= x1 or y2 <= y1:
            return np.array([])
        return frame[y1:y2, x1:x2].copy()
