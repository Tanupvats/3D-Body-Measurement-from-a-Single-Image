"""Person-detection adapters.

We default to Ultralytics YOLO because it's pip-installable and reasonable
quality. RTMDet is the modern open replacement; adding it is a 30-line
adapter when needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..core.config import DetectionConfig
from ..core.interfaces import IPersonDetector
from ..core.types import PersonDetection
from ..utils.logging import get_logger

logger = get_logger(__name__)


class YoloPersonDetector(IPersonDetector):
    """Ultralytics YOLO instance-segmentation as a person detector.

    The seg variant gives us both a bbox and a tight mask in one pass,
    saving a separate matting step for users who don't need part labels.
    """

    def __init__(
        self,
        weights: str = "yolov8s-seg.pt",
        cfg: DetectionConfig | None = None,
        device: Optional[str] = None,
    ):
        self.cfg = cfg or DetectionConfig()
        self.weights = weights
        self.device = device
        self._model = None  # lazy

    def _load(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # noqa
        except ImportError as e:
            raise RuntimeError(
                "ultralytics is not installed. `pip install ultralytics` to use YoloPersonDetector."
            ) from e
        from ultralytics import YOLO
        self._model = YOLO(self.weights)
        if self.device:
            self._model.to(self.device)

    def detect(self, image_bgr: np.ndarray) -> Optional[PersonDetection]:
        self._load()
        results = self._model(image_bgr, verbose=False, classes=[0])  # 0 = person in COCO
        if not results:
            return None
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        # Pick best-scoring person.
        scores = r.boxes.conf.cpu().numpy()
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score < self.cfg.min_person_score:
            logger.debug(f"best person score {score:.3f} below threshold")
            return None

        bbox = r.boxes.xyxy[best].cpu().numpy().astype(np.float32)

        mask = None
        if r.masks is not None and best < len(r.masks):
            # YOLO returns mask at processed input size; .data is float (0..1)
            m = r.masks.data[best].cpu().numpy()
            # Resize back to original
            h, w = image_bgr.shape[:2]
            import cv2
            m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (m_resized > 0.5).astype(np.uint8) * 255

        # Image-area sanity check
        h, w = image_bgr.shape[:2]
        bbox_area_frac = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(h * w)
        if bbox_area_frac < self.cfg.min_person_pixel_area_frac:
            logger.debug(f"person too small: {bbox_area_frac:.3f} of image")
            return None

        return PersonDetection(bbox_xyxy=bbox, score=score, mask=mask)


class MockPersonDetector(IPersonDetector):
    """Returns a centered bbox covering most of the image. Tests only."""

    def detect(self, image_bgr: np.ndarray) -> Optional[PersonDetection]:
        h, w = image_bgr.shape[:2]
        pad = 0.05
        bbox = np.array([w * pad, h * pad, w * (1 - pad), h * (1 - pad)], dtype=np.float32)
        return PersonDetection(bbox_xyxy=bbox, score=1.0, mask=None)
