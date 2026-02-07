"""Vision-related helpers: keypoints extraction, matching, and geometry math."""

from __future__ import annotations

from ctypes import sizeof, c_float
from typing import Iterable, List, Sequence, Tuple

SKELETON: Sequence[Tuple[int, int]] = (
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),
    (12, 13),
    (6, 12),
    (7, 13),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (2, 3),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
)


def clamp(value: float, minimum: float, maximum: float) -> int:
    """Clamp a floating-point value into a bounded integer range."""
    return int(min(maximum, max(minimum, value)))


def extract_keypoints(
    mask_params, frame_width: int, frame_height: int
) -> List[Tuple[float, float, float]]:
    """Extract pose keypoints from raw mask parameters."""
    if not hasattr(mask_params, "size") or mask_params.size <= 0:
        return []

    num_joints = int(mask_params.size / (sizeof(c_float) * 3))
    if num_joints <= 0:
        return []

    gain = min(mask_params.width / frame_width, mask_params.height / frame_height)
    if gain <= 0:
        return []

    pad_x = (mask_params.width - frame_width * gain) * 0.5
    pad_y = (mask_params.height - frame_height * gain) * 0.5

    data = mask_params.get_mask_array()
    keypoints: List[Tuple[float, float, float]] = []
    for idx in range(num_joints):
        x = (data[idx * 3] - pad_x) / gain
        y = (data[idx * 3 + 1] - pad_y) / gain
        conf = data[idx * 3 + 2]
        keypoints.append((x, y, conf))

    return keypoints


def point_in_bbox(x: float, y: float, bbox: Sequence[float]) -> bool:
    """Return True when a point lies within the provided bounding box."""
    bx, by, bw, bh = bbox
    return bx <= x <= bx + bw and by <= y <= by + bh


def find_matching_pose(
    bbox: Sequence[float],
    pose_detections: Iterable[dict],
    threshold: float = 0.85,
) -> dict | None:
    """Find the pose whose keypoints mostly fall inside the bounding box."""
    bx, by, bw, bh = bbox
    for pose in pose_detections:
        keypoints = pose.get("keypoints", [])
        if not keypoints:
            continue
        inside = sum(
            1 for x, y, _ in keypoints if point_in_bbox(x, y, (bx, by, bw, bh))
        )
        if inside >= threshold * len(keypoints):
            return pose
    return None
