

from __future__ import annotations

import numpy as np

from ..core.config import CameraConfig
from ..core.interfaces import ICameraEstimator
from ..core.types import CameraIntrinsics
from ..io.image_io import estimate_intrinsics_from_exif


class ExifCameraEstimator(ICameraEstimator):
    """Pulls focal from EXIF when present, falls back to FoV prior."""

    def __init__(self, cfg: CameraConfig | None = None):
        self.cfg = cfg or CameraConfig()

    def estimate(self, image_bgr: np.ndarray, exif: dict) -> CameraIntrinsics:
        h, w = image_bgr.shape[:2]
        return estimate_intrinsics_from_exif(
            w, h, exif,
            sensor_diag_mm=self.cfg.sensor_diag_mm,
            default_hfov_deg=self.cfg.default_hfov_deg,
        )
