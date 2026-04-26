

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.types import Gender, Keypoints2D, PersonDetection, SMPLXParams
from .corrector import build_feature_vector


def shoulder_hip_ratio_from_kpts(kpts: Keypoints2D) -> float:
    """Width(shoulders) / width(hips) from 2D keypoints. ~1.0 if missing."""
    xy = kpts.xy
    cnf = kpts.confidence
    # shoulders 5,6 hips 11,12
    sh_ok = cnf[5] > 0.3 and cnf[6] > 0.3
    hp_ok = cnf[11] > 0.3 and cnf[12] > 0.3
    if not (sh_ok and hp_ok):
        return 1.0
    sh_w = abs(xy[5, 0] - xy[6, 0])
    hp_w = abs(xy[11, 0] - xy[12, 0])
    if hp_w < 1e-3:
        return 1.0
    return float(sh_w / hp_w)


def features_from_pipeline_state(
    init_params: SMPLXParams,
    height_cm: float,
    detection: PersonDetection,
    keypoints: Keypoints2D,
    gender: Gender,
) -> np.ndarray:
    bbox = detection.bbox_xyxy
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    bbox_wh_ratio = float(bw / max(bh, 1.0))
    sh_hip = shoulder_hip_ratio_from_kpts(keypoints)
    return build_feature_vector(
        betas=init_params.betas,
        height_cm=height_cm,
        bbox_w_h_ratio=bbox_wh_ratio,
        shoulder_hip_ratio=sh_hip,
        gender=gender,
    )
