

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.types import (
    FittedBody,
    HumanParsing,
    ImageInput,
    Keypoints2D,
    PersonDetection,
    QCReport,
)


def pre_pipeline_qc(
    image: ImageInput,
    detection: PersonDetection,
    keypoints: Keypoints2D,
    cfg: QCConfig,
) -> QCReport:
    failures: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, float] = {}

    h, w = image.image_bgr.shape[:2]
    metrics["image_h"] = float(h)
    metrics["image_w"] = float(w)
    if min(h, w) < cfg.min_image_resolution_px:
        failures.append(
            f"image too low resolution: min(h,w)={min(h,w)} < {cfg.min_image_resolution_px}"
        )

    if detection is None or detection.score < 0.5:
        failures.append("no confident person detected")
        return QCReport(passed=False, failures=failures, warnings=warnings, metrics=metrics)

    bbox = detection.bbox_xyxy
    metrics["bbox_score"] = float(detection.score)
    bbox_h_frac = (bbox[3] - bbox[1]) / float(h)
    metrics["bbox_height_fraction"] = float(bbox_h_frac)
    if bbox_h_frac < 0.4:
        failures.append(f"person too small: bbox covers only {bbox_h_frac:.0%} of image height")

    # Frame clipping: penalty if bbox is right at the edge
    margin = 4
    if cfg.require_full_body_visible:
        if bbox[1] < margin or bbox[3] > h - margin:
            failures.append("person appears clipped at top/bottom of frame")
        if bbox[0] < margin or bbox[2] > w - margin:
            warnings.append("person appears clipped at left/right of frame")

    # Pose check: are wrists below shoulders? (arms roughly down for measurement pose)
    # COCO indices: shoulders 5,6 ; wrists 9,10
    try:
        kp = keypoints.xy
        cnf = keypoints.confidence
        if cnf[5] > 0.3 and cnf[9] > 0.3 and kp[9, 1] < kp[5, 1]:
            warnings.append("left wrist above left shoulder — measurements degrade in raised-arm poses")
        if cnf[6] > 0.3 and cnf[10] > 0.3 and kp[10, 1] < kp[6, 1]:
            warnings.append("right wrist above right shoulder — measurements degrade in raised-arm poses")
    except Exception:
        pass

    # Keypoint coverage
    n_visible = int((keypoints.confidence > 0.3).sum())
    metrics["visible_keypoints"] = float(n_visible)
    if n_visible < 12:
        failures.append(f"only {n_visible}/17 body keypoints visible — pose unreliable")

    return QCReport(
        passed=len(failures) == 0,
        failures=failures,
        warnings=warnings,
        metrics=metrics,
    )


def post_fit_qc(
    fitted: FittedBody,
    keypoints_per_view: list[Keypoints2D],
    parsings_per_view: list[HumanParsing],
    height_target_cm: float | None,
    cfg: QCConfig,
) -> QCReport:
    failures: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, float] = {"fit_loss": float(fitted.fit_loss)}

    # Height check
    if height_target_cm is not None:
        v = fitted.vertices_m
        height_pred_cm = (v[:, 1].max() - v[:, 1].min()) * 100.0
        metrics["height_pred_cm"] = float(height_pred_cm)
        if abs(height_pred_cm - height_target_cm) > 2.0:
            warnings.append(
                f"reconstructed height {height_pred_cm:.1f}cm differs from "
                f"target {height_target_cm:.1f}cm by >2cm — fit may be miscalibrated"
            )

    if not np.isfinite(fitted.fit_loss):
        failures.append("optimizer returned non-finite loss")

    return QCReport(
        passed=len(failures) == 0,
        failures=failures,
        warnings=warnings,
        metrics=metrics,
    )
