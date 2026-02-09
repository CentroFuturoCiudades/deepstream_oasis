"""Configuration loading and runtime settings for DeepStream YOLO Pose app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Sequence

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_FILE = "config/deepstream.yaml"

load_dotenv()

class ConfigError(Exception):
    """Raised when configuration values are invalid or missing."""


def _require(section: str, key: str, data: dict, path: str):
    """Fetch a required configuration value or raise ConfigError."""
    if section not in data or key not in data[section]:
        raise ConfigError(f"Missing '{key}' in '{section}' section of {path}")
    return data[section][key]


def _as_bool(value) -> bool:
    """Normalize miscellaneous truthy representations to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_list(value, label: str) -> List[str]:
    """Convert list or comma-separated values into a clean list of strings."""
    if isinstance(value, (list, tuple)):
        result = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        result = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raise ConfigError(f"{label} must be a list or comma-separated string")

    if not result:
        raise ConfigError(f"{label} must contain at least one value")
    return result


def _as_abs_path(path_value: str) -> str:
    """Return the absolute version of a configuration path."""
    return os.path.abspath(path_value)


@dataclass(frozen=True)
class ArgumentsConfig:
    sources: Sequence[str]
    camera_ids: Sequence[str]
    infer_config: str
    pose_config: str
    output_path: str
    width: int
    height: int
    gpu: int
    enable_csv: bool
    enable_db: bool


@dataclass(frozen=True)
class ConstantsConfig:
    tracker_config: str
    tracker_lib: str
    streammux_batch_size: int
    num_keypoints: int
    fps_interval_sec: int
    infer_stride: int
    max_recording_minutes: int
    max_no_det_frames: int
    mexico_tz_offset: int
    csv_path: str

    @property
    def max_recording_frames(self) -> int:
        return int((30 / self.infer_stride) * 60 * self.max_recording_minutes) # Assuming 30 FPS

    @property
    def mexico_timezone(self) -> timezone:
        return timezone(timedelta(hours=self.mexico_tz_offset))


@dataclass(frozen=True)
class AppConfig:
    arguments: ArgumentsConfig
    constants: ConstantsConfig
    raw: dict


@dataclass
class RuntimeConfig:
    width: int
    height: int
    gpu_id: int
    is_jetson: bool
    enable_csv: bool
    enable_db: bool


def load_app_config(path: str = DEFAULT_CONFIG_FILE) -> AppConfig:
    """Load the YAML configuration and project strongly-typed dataclasses."""
    if not os.path.isfile(path):
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as cfg_file:
        raw_config = yaml.safe_load(cfg_file) or {}

    if not isinstance(raw_config, dict):
        raise ConfigError(f"Config root must be a mapping in {path}")

    arguments = raw_config.get("arguments", {})
    constants = raw_config.get("constants", {})

    env_sources = os.getenv("CAMERA_SOURCES")
    if env_sources:
        sources = _as_list(env_sources, "CAMERA_SOURCES")
    else:
        try:
            sources_value = _require("arguments", "source", raw_config, path)
        except ConfigError as exc:
            raise ConfigError(
                "arguments.source missing in config and CAMERA_SOURCES not set"
            ) from exc
        sources = _as_list(sources_value, "arguments.source")
    camera_ids = _as_list(
        _require("arguments", "camera_ids", raw_config, path), "arguments.camera_ids"
    )

    arguments_config = ArgumentsConfig(
        sources=sources,
        camera_ids=camera_ids,
        infer_config=_as_abs_path(_require("arguments", "config", raw_config, path)),
        pose_config=_require("arguments", "pose_config", raw_config, path),
        output_path=_require("arguments", "output", raw_config, path),
        width=int(_require("arguments", "width", raw_config, path)),
        height=int(_require("arguments", "height", raw_config, path)),
        gpu=int(_require("arguments", "gpu", raw_config, path)),
        enable_csv=_as_bool(_require("arguments", "enable_csv", raw_config, path)),
        enable_db=_as_bool(_require("arguments", "enable_db", raw_config, path)),
    )

    constants_config = ConstantsConfig(
        tracker_config=_require("constants", "tracker_config", raw_config, path),
        tracker_lib=_require("constants", "tracker_lib", raw_config, path),
        streammux_batch_size=int(
            _require("constants", "streammux_batch_size", raw_config, path)
        ),
        num_keypoints=int(_require("constants", "num_keypoints", raw_config, path)),
        fps_interval_sec=int(
            _require("constants", "fps_interval_sec", raw_config, path)
        ),
        infer_stride=int(_require("constants", "infer_stride", raw_config, path)),
        max_recording_minutes=int(
            _require("constants", "max_recording_minutes", raw_config, path)
        ),
        max_no_det_frames=int(
            _require("constants", "max_no_det_frames", raw_config, path)
        ),
        mexico_tz_offset=int(
            _require("constants", "mexico_tz_offset", raw_config, path)
        ),
        csv_path=_require("constants", "csv_path", raw_config, path),
    )

    return AppConfig(
        arguments=arguments_config, constants=constants_config, raw=raw_config
    )


def create_runtime_config(app_config: AppConfig, is_jetson: bool) -> RuntimeConfig:
    """Build mutable runtime flags derived from the immutable app config."""
    return RuntimeConfig(
        width=app_config.arguments.width,
        height=app_config.arguments.height,
        gpu_id=app_config.arguments.gpu,
        is_jetson=is_jetson,
        enable_csv=app_config.arguments.enable_csv,
        enable_db=app_config.arguments.enable_db,
    )


def current_timestamp(constants: ConstantsConfig) -> str:
    """Generate a timezone-aware timestamp using the configured offset."""
    return datetime.now(constants.mexico_timezone).isoformat()
