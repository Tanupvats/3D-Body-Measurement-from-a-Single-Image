"""Core dataclasses. Everything that flows between pipeline stages is typed.

Why dataclasses and not dicts: stages depend on each other's outputs and
dicts let bugs slip in silently. With dataclasses, a missing field is a
type error at construction time, not an undebuggable KeyError three stages
later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Gender(str, Enum):
    NEUTRAL = "neutral"
    MALE = "male"
    FEMALE = "female"


class ViewType(str, Enum):
    FRONT = "front"
    SIDE = "side"
    BACK = "back"
    UNKNOWN = "unknown"


@dataclass
class CameraIntrinsics:
    """Pinhole camera. fx/fy in pixels, cx/cy principal point in pixels.

    `is_estimated` tells downstream consumers whether to trust this.
    EXIF-derived focals are reliable; learned-prior focals are not.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    is_estimated: bool = False
    source: str = "unknown"  # "exif", "default_fov", "learned_prior", etc.

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = 60.0) -> "CameraIntrinsics":
        """Fallback constructor when no metadata is available.

        60 deg horizontal FoV is a reasonable phone-camera default but is
        explicitly marked as estimated so downstream uncertainty grows.
        """
        f = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                   width=width, height=height, is_estimated=True,
                   source="default_fov")


@dataclass
class PersonDetection:
    """A single detected person in image-pixel coordinates."""
    bbox_xyxy: np.ndarray   # shape (4,)
    score: float
    mask: Optional[np.ndarray] = None  # HxW uint8, 0/255


@dataclass
class HumanParsing:
    """Per-part segmentation. Optional; falls back to whole-body mask."""
    full_mask: np.ndarray            # HxW, 0/255
    part_labels: Optional[np.ndarray] = None  # HxW, int per-pixel class id
    label_names: Optional[list[str]] = None   # index -> name (e.g. "left_arm")


@dataclass
class Keypoints2D:
    """COCO-Wholebody convention preferred (133 kpts). Body subset is the
    first 17 indices (COCO standard)."""
    xy: np.ndarray         # (N, 2) in image pixels
    confidence: np.ndarray # (N,)
    convention: str = "coco_17"

    def __post_init__(self) -> None:
        assert self.xy.shape[0] == self.confidence.shape[0], \
            "keypoint count mismatch between xy and confidence"


@dataclass
class SMPLXParams:
    """Numpy-side SMPL-X parameter container.

    Shape (1,N) leading dim kept so a batch dimension exists for the
    optimizer; downstream code never assumes N>1.
    """
    betas: np.ndarray             # (1, 10) or (1, 16) — model-dependent
    body_pose: np.ndarray         # (1, 63) for SMPL-X (21 joints x 3)
    global_orient: np.ndarray     # (1, 3) axis-angle
    transl: np.ndarray            # (1, 3) translation in metres
    left_hand_pose: Optional[np.ndarray] = None   # (1, 45) or PCA-reduced
    right_hand_pose: Optional[np.ndarray] = None
    jaw_pose: Optional[np.ndarray] = None
    expression: Optional[np.ndarray] = None
    gender: Gender = Gender.NEUTRAL


@dataclass
class FittedBody:
    """Result of the reconstruction stage: parameters + posed mesh."""
    params: SMPLXParams
    vertices_m: np.ndarray  # (V, 3) in metres, posed
    faces: np.ndarray       # (F, 3) int triangle indices
    joints_m: np.ndarray    # (J, 3) in metres
    camera: CameraIntrinsics
    fit_loss: float = float("nan")  # final optimization loss; lower is tighter fit


@dataclass
class Measurement:
    """A single anthropometric measurement.

    `value_cm` and `uncertainty_cm` are both centimetres. Uncertainty is
    1-sigma. `method` documents how it was extracted so the report can
    explain itself."""
    name: str
    value_cm: float
    uncertainty_cm: float
    method: str  # e.g. "convex_hull_girth", "geodesic_path", "vertex_distance"
    notes: str = ""


@dataclass
class QCReport:
    """Quality gates. If `passed` is False the API may refuse to return
    measurements depending on policy. `warnings` are non-fatal."""
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ImageInput:
    """A single view going into the pipeline."""
    image_bgr: np.ndarray       # HxWx3, uint8, BGR (OpenCV convention)
    view: ViewType = ViewType.FRONT
    camera: Optional[CameraIntrinsics] = None  # if known a-priori
    exif: dict = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Top-level result returned by the pipeline."""
    measurements: list[Measurement]
    fitted_body: Optional[FittedBody]
    qc: QCReport
    metadata: dict = field(default_factory=dict)

    def measurements_dict(self) -> dict[str, dict]:
        return {
            m.name: {
                "value_cm": round(m.value_cm, 2),
                "uncertainty_cm": round(m.uncertainty_cm, 2),
                "method": m.method,
                "notes": m.notes,
            }
            for m in self.measurements
        }
