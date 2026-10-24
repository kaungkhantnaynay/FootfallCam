"""Explicit, validated settings; no clip-specific timestamps or coordinates."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model: str = "models/yolo11n.pt"
    device: str = "cpu"
    image_size: int = 960
    person_confidence: float = 0.20
    detector_rotations: tuple[int, ...] = (0, 90, 180, 270)
    tag_stride: int = 3
    tag_threshold: float = 0.76
    verifier: str = "models/tag_verifier.joblib"
    verifier_margin: float = 0.0
    acquisition_motion_px: float = 25.0
    min_track_travel_px: float = 100.0
    tag_widths: tuple[int, ...] = (16, 20, 24, 28, 34, 40)
    rotation_step: int = 15
    min_tag_std: float = 8.0
    min_person_size: int = 35
    max_track_gap_s: float = 0.32
    confirm_hits: int = 3
    confirm_window_s: float = 0.8
    evidence_ttl_s: float = 1.5
    offline_backfill_s: float = 8.0

    @classmethod
    def load(cls, path: str | None = None, **overrides):
        values = json.loads(Path(path).read_text()) if path else {}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        values.update({k: v for k, v in overrides.items() if v is not None})
        for key in ("tag_widths", "detector_rotations"):
            if key in values:
                values[key] = tuple(values[key])
        config = cls(**values)
        for key in ("person_confidence", "tag_threshold"):
            if not 0 < getattr(config, key) <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
        if config.tag_stride < 1 or config.confirm_hits < 1:
            raise ValueError("Stride and confirm_hits must be positive")
        if not config.detector_rotations or any(x not in (0, 90, 180, 270) for x in config.detector_rotations):
            raise ValueError("Detector rotations must be quarter turns")
        if not 1 <= config.rotation_step <= 180 or config.image_size < 32:
            raise ValueError("Invalid image size or rotation step")
        if not config.tag_widths or min(config.tag_widths) < 4:
            raise ValueError("Tag widths must be at least 4 pixels")
        if min(config.max_track_gap_s, config.confirm_window_s, config.evidence_ttl_s) <= 0:
            raise ValueError("Time windows must be positive")
        if config.offline_backfill_s < 0:
            raise ValueError("Backfill must be non-negative")
        if min(config.acquisition_motion_px, config.min_track_travel_px) < 0:
            raise ValueError("Acquisition motion must be non-negative")
        return config

    def as_dict(self):
        return asdict(self)
