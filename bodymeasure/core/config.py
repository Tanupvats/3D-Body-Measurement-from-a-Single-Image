

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# Repository root — used to locate model weights, configs, etc.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_MODELS_DIR = Path.home() / ".cache" / "bodymeasure" / "models"


@dataclass
class CameraConfig:
    default_hfov_deg: float = 60.0   # phone-camera-ish default; overridden by EXIF
    sensor_diag_mm: float = 7.0      # ~1/2.55" sensor; used with EXIF focal_length_mm


@dataclass
class DetectionConfig:
    min_person_score: float = 0.4
    min_person_pixel_area_frac: float = 0.04   # below 4% of image -> reject


@dataclass
class ParsingConfig:
    # Whether to require a per-part parser (SCHP/Sapiens) or accept whole-body
    require_part_labels: bool = False


@dataclass
class PoseConfig:
    min_keypoint_confidence: float = 0.3
    convention: str = "coco_17"


@dataclass
class OptimizerConfig:
    """SMPLify-X-style optimization weights and schedule.

    These are the dials that have the largest effect on output quality.
    The schedule loosens shape regularization across stages — locking pose
    first then unlocking shape is more stable than joint optimization.
    """
    # Stage 1: coarse global alignment (pose only, fixed shape, large reg)
    stage1_lr: float = 0.05
    stage1_iters: int = 80
    stage1_w_kpt: float = 1.0
    stage1_w_silh: float = 0.0
    stage1_w_shape_reg: float = 100.0
    stage1_w_pose_reg: float = 5.0

    # Stage 2: shape unlocked, silhouette term enabled
    stage2_lr: float = 0.02
    stage2_iters: int = 120
    stage2_w_kpt: float = 1.0
    stage2_w_silh: float = 1.0
    stage2_w_shape_reg: float = 5.0
    stage2_w_pose_reg: float = 1.0

    # Stage 3: fine refinement, low reg
    stage3_lr: float = 0.005
    stage3_iters: int = 80
    stage3_w_kpt: float = 1.0
    stage3_w_silh: float = 2.0
    stage3_w_shape_reg: float = 1.0
    stage3_w_pose_reg: float = 0.5

    # Hard constraints
    height_constraint_weight: float = 50.0   # high weight, near-equality
    height_tol_cm: float = 0.5               # allowed slack

    # Renderer
    render_image_size: int = 256             # silhouette render size for diff. rasterizer

    # Numerical
    grad_clip_norm: float = 1.0


@dataclass
class MeasurementConfig:
    """Defines which measurements to extract and how."""
    # Extra slack added to convex-hull girths (cm). Tape measures don't lie
    # perfectly flat against highly concave regions; this is a population-
    # average adjustment that should be calibrated against ground truth
    # once available.
    convex_hull_slack_cm: float = 0.0  # 0 until calibrated; do not lie

    # Whether to compute geodesic measurements (slower, more accurate for
    # curved paths like across-back distance).
    enable_geodesic: bool = True


@dataclass
class QCConfig:
    min_image_resolution_px: int = 384       # below this, hard reject
    max_pose_deviation_rad: float = 0.4      # arms not roughly down? warn
    max_silhouette_iou_residual: float = 0.15  # poor fit -> warn
    require_full_body_visible: bool = True


@dataclass
class FineTuneConfig:
    """Hyperparameters for the shape-corrector fine-tune stage.

    The fine-tune learns a residual on betas to correct systematic bias
    between pretrained-model output and ground-truth measurements. We do
    NOT fine-tune the regressor backbone in v1 — too few samples make this
    a recipe for overfitting.
    """
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    val_split: float = 0.2
    measurement_loss_weight: float = 1.0
    beta_reg_weight: float = 0.05
    early_stop_patience: int = 25
    seed: int = 42


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    models_dir: Path = field(default_factory=lambda: DEFAULT_MODELS_DIR)

    # Adapter selection — strings instead of classes so YAML can override
    detector_name: str = "yolo"            # {yolo, mock}
    parser_name: str = "rembg"             # {sapiens, schp, sam2, rembg, mock}
    pose_name: str = "rtmpose"             # {sapiens, vitpose, rtmpose, mock}
    regressor_name: str = "multihmr"       # {multihmr, tokenhmr, mock}
    use_optimizer: bool = True
    body_model: str = "smplx"              # {smpl, smplx}

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        # Shallow construction; nested overrides handled by section.
        cfg = cls()
        for key, val in data.items():
            if hasattr(cfg, key):
                attr = getattr(cfg, key)
                if hasattr(attr, "__dataclass_fields__") and isinstance(val, dict):
                    for k2, v2 in val.items():
                        if hasattr(attr, k2):
                            setattr(attr, k2, v2)
                else:
                    setattr(cfg, key, val)
        if isinstance(cfg.models_dir, str):
            cfg.models_dir = Path(cfg.models_dir).expanduser()
        return cfg

    def model_path(self, *parts: str) -> Path:
        return self.models_dir.joinpath(*parts)


def default_config() -> Config:
    return Config()
