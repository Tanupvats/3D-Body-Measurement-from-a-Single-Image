"""2D keypoint adapters.

Default to RTMPose (open-mmlab) — strong accuracy, fast, ONNX-shippable.
ViTPose / Sapiens-Pose adapters are stubbed.

We ALWAYS return COCO-17 body keypoints minimum. Wholebody (133) is
preferred when the underlying model supports it because the optimizer
can use foot, hand and face landmarks for tighter fits.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..core.config import PoseConfig
from ..core.interfaces import IPoseEstimator
from ..core.types import Keypoints2D
from ..utils.logging import get_logger

logger = get_logger(__name__)


# COCO-17 keypoint order (the canonical body subset).
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class UltralyticsPose(IPoseEstimator):
    """Uses ultralytics' yolo*-pose models. Quick to install, decent quality.

    Returns COCO-17. For higher accuracy or wholebody, swap to RTMPose.
    """

    def __init__(self, weights: str = "yolov8s-pose.pt", cfg: PoseConfig | None = None):
        self.cfg = cfg or PoseConfig()
        self.weights = weights
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from ultralytics import YOLO
        self._model = YOLO(self.weights)

    def estimate(self, image_bgr: np.ndarray, bbox_xyxy: Optional[np.ndarray] = None) -> Keypoints2D:
        self._load()
        results = self._model(image_bgr, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return Keypoints2D(
                xy=np.zeros((17, 2), dtype=np.float32),
                confidence=np.zeros(17, dtype=np.float32),
                convention="coco_17",
            )
        # Pick the detection closest to bbox center if bbox provided, else the largest.
        kpts = results[0].keypoints
        xys = kpts.xy.cpu().numpy()  # (N, 17, 2)
        confs = kpts.conf.cpu().numpy() if kpts.conf is not None else np.ones((xys.shape[0], 17))
        if bbox_xyxy is not None:
            cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
            cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
            centers = xys.mean(axis=1)
            d2 = ((centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2)
            best = int(np.argmin(d2))
        else:
            # Largest spread
            spreads = (xys.max(axis=1) - xys.min(axis=1)).sum(axis=1)
            best = int(np.argmax(spreads))
        return Keypoints2D(
            xy=xys[best].astype(np.float32),
            confidence=confs[best].astype(np.float32),
            convention="coco_17",
        )


class RTMPoseEstimator(IPoseEstimator):
    """RTMPose adapter via mmpose. Stubbed pending mmpose install.

    To enable: `pip install mmpose mmcv mmengine`, then download an RTMPose
    checkpoint (e.g. rtmpose-l_8xb256-420e_coco-256x192) and fill `_load`.
    """

    def __init__(self, config_path: str, checkpoint_path: str, cfg: PoseConfig | None = None):
        self.cfg = cfg or PoseConfig()
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

    def estimate(self, image_bgr: np.ndarray, bbox_xyxy: Optional[np.ndarray] = None) -> Keypoints2D:
        raise NotImplementedError(
            "RTMPoseEstimator is a stub. Install mmpose and wire `init_model`/`inference_topdown`."
        )


class MockPoseEstimator(IPoseEstimator):
    """Returns a plausible T-pose layout inside the bbox. Tests only."""

    def estimate(self, image_bgr: np.ndarray, bbox_xyxy: Optional[np.ndarray] = None) -> Keypoints2D:
        h, w = image_bgr.shape[:2]
        if bbox_xyxy is None:
            bbox_xyxy = np.array([w * 0.3, h * 0.05, w * 0.7, h * 0.95], dtype=np.float32)
        x0, y0, x1, y1 = bbox_xyxy
        bw = x1 - x0
        bh = y1 - y0
        cx = (x0 + x1) / 2
        # Rough fractions in COCO-17 order; not anatomically perfect, just non-degenerate.
        rel = np.array([
            (0.50, 0.07),  # nose
            (0.47, 0.06), (0.53, 0.06),  # eyes
            (0.45, 0.08), (0.55, 0.08),  # ears
            (0.40, 0.20), (0.60, 0.20),  # shoulders
            (0.30, 0.35), (0.70, 0.35),  # elbows
            (0.22, 0.50), (0.78, 0.50),  # wrists
            (0.43, 0.55), (0.57, 0.55),  # hips
            (0.43, 0.75), (0.57, 0.75),  # knees
            (0.43, 0.95), (0.57, 0.95),  # ankles
        ])
        xy = np.zeros((17, 2), dtype=np.float32)
        xy[:, 0] = x0 + rel[:, 0] * bw
        xy[:, 1] = y0 + rel[:, 1] * bh
        return Keypoints2D(xy=xy, confidence=np.full(17, 0.9, dtype=np.float32),
                           convention="coco_17")
