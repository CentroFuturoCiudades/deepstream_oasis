"""Output handling: CSV metadata and Event Hub publishing."""

from __future__ import annotations

import csv
import os
import uuid
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .config import AppConfig, RuntimeConfig
from .vision import find_matching_pose
from .send_helper import SendHelper


class OutputManager:
    """Coordinates CSV writing and Event Hub publishing."""

    def __init__(self, app_config: AppConfig, runtime: RuntimeConfig) -> None:
        self._config = app_config
        self._runtime = runtime
        self._csv_file: Optional[object] = None
        self._csv_writer: Optional[csv.writer] = None
        self._send_helper: Optional[SendHelper] = None

    # ------------------------------------------------------------------
    # CSV handling
    # ------------------------------------------------------------------
    def init_csv(self) -> None:
        """Prepare the CSV writer when CSV export is enabled."""
        if not self._runtime.enable_csv:
            print("CSV output disabled")
            return

        self._csv_file = open(self._config.constants.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)

        header = ["frame", "track_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        for idx in range(self._config.constants.num_keypoints):
            header.extend([f"kp{idx}_x", f"kp{idx}_y", f"kp{idx}_conf"])

        self._csv_writer.writerow(header)
        print(f"CSV output enabled: {self._config.constants.csv_path}")

    def write_csv_rows(
        self,
        frame_number: int,
        normal_detections: Sequence[Mapping[str, object]],
        pose_detections: Iterable[Mapping[str, object]],
    ) -> None:
        """Append detection rows for the given frame to the CSV file."""
        if not self._runtime.enable_csv or not self._csv_writer:
            return

        pose_detections_list = list(pose_detections)
        for det in normal_detections:
            row = [
                frame_number,
                det["track_id"],
                round(det["bbox"][0], 2),
                round(det["bbox"][1], 2),
                round(det["bbox"][2], 2),
                round(det["bbox"][3], 2),
            ]

            match = find_matching_pose(det["bbox"], pose_detections_list)
            if match:
                for x, y, conf in match["keypoints"]:
                    row.extend([round(x, 2), round(y, 2), round(conf, 3)])

            self._csv_writer.writerow(row)

    def close_csv(self) -> None:
        """Close any open CSV file handles."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    # ------------------------------------------------------------------
    # Event Hub handling
    # ------------------------------------------------------------------
    def init_event_hub(self) -> None:
        """Instantiate the Event Hub helper when database output is active."""
        if not self._runtime.enable_db:
            print("Database output disabled")
            return

        try:
            self._send_helper = SendHelper()
            print("Event Hub connection initialized for database writes")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"ERROR: Failed to initialize Event Hub: {exc}")
            print("Database writes disabled.")
            self._runtime.enable_db = False
            self._send_helper = None

    def process_db_event(
        self,
        event: Sequence[object],
        track_to_person_id: MutableMapping[int, int],
    ) -> None:
        """Publish a processed detection batch to Azure Event Hub."""
        if not self._runtime.enable_db or self._send_helper is None:
            return

        (
            frame_num,
            video_id,
            normal_detections,
            pose_detections,
            timestamp,
            camera_id,
        ) = event

        pose_detections_list = list(pose_detections)

        if not normal_detections:
            return

        prepared_detections: List[dict] = []
        for det in normal_detections:
            bbox_dict = format_bbox_for_db(det["bbox"])
            skeleton = None
            match = find_matching_pose(det["bbox"], pose_detections_list)
            if match:
                skeleton = format_skeleton_for_db(match["keypoints"])

            prepared_detections.append(
                {
                    "track_id": det["track_id"],
                    "confidence": float(det.get("confidence", 0.0)),
                    "bbox": bbox_dict,
                    "skeleton": skeleton,
                }
            )

        self._send_helper.send_frame_with_detections(
            frame_id=str(uuid.uuid4()),
            video_id=video_id,
            camera_id=camera_id,
            timestamp=timestamp,
            width=self._runtime.width,
            height=self._runtime.height,
            detections=prepared_detections,
            track_to_person_id=track_to_person_id,
        )

    def cleanup_event_hub(self) -> None:
        """Flush pending Event Hub events and dispose of the helper."""
        if self._send_helper is None:
            return

        try:
            stats = self._send_helper.get_stats()
            print("\n[Event Hub Stats]")
            print(f"  Events queued:  {stats['events_queued']}")
            print(f"  Events sent:    {stats['events_sent']}")
            print(f"  Batches sent:   {stats['batches_sent']}")
            print(f"  Errors:         {stats['errors']}")
            print(f"  Queue pending:  {stats['queue_size']}")

            if stats["queue_size"] > 0:
                print(f"  Flushing {stats['queue_size']} pending events...")
                self._send_helper.flush()

            self._send_helper.close()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"WARNING: Error closing Event Hub: {exc}")
        finally:
            self._send_helper = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def summary_lines(self) -> List[str]:
        """Return human-readable status lines for enabled outputs."""
        rows = []
        rows.append(
            f"CSV:       {'enabled' if self._runtime.enable_csv else 'disabled'}"
        )
        rows.append(
            f"DATABASE:  {'enabled' if self._runtime.enable_db else 'disabled'}"
        )
        return rows

    def shutdown(self) -> None:
        """Release all output resources during application shutdown."""
        self.close_csv()
        self.cleanup_event_hub()


def format_bbox_for_db(bbox: Sequence[float]) -> dict:
    """Convert a detection bounding box to the Event Hub payload schema."""
    x, y, w, h = bbox
    return {
        "x": float(round(x, 2)),
        "y": float(round(y, 2)),
        "width": float(round(w, 2)),
        "height": float(round(h, 2)),
    }


def format_skeleton_for_db(keypoints: Sequence[Sequence[float]]) -> List[dict]:
    """Translate pose keypoints into the structured skeleton format."""
    if not keypoints:
        return []

    names = (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )

    skeleton: List[dict] = []
    for idx, (x, y, conf) in enumerate(keypoints):
        name = names[idx] if idx < len(names) else f"kp_{idx}"
        skeleton.append(
            {
                "name": name,
                "x": float(round(x, 2)),
                "y": float(round(y, 2)),
                "confidence": float(round(conf, 3)),
            }
        )

    return skeleton
