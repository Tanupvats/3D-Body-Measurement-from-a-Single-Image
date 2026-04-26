"""Human parsing / matting adapters.

Quality ranking for body measurement use:
  1. Sapiens-Seg (Meta, 2024) — best, but >2GB checkpoint
  2. SCHP / Self-Correction Human Parsing — per-part labels
  3. SAM2 with bbox prompt — strong matte, no part labels
  4. rembg (U2Net derivative) — light fallback, foreground vs background only
  5. Use detector mask if nothing else loads

All adapters return a unified `HumanParsing`. Part labels are optional;
the optimizer handles their absence gracefully.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..core.config import ParsingConfig
from ..core.interfaces import IHumanParser
from ..core.types import HumanParsing
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RembgParser(IHumanParser):
    """Background removal via rembg. No body-part labels."""

    def __init__(self, model_name: str = "u2net_human_seg", cfg: ParsingConfig | None = None):
        self.cfg = cfg or ParsingConfig()
        self.model_name = model_name
        self._session = None

    def _load(self):
        if self._session is not None:
            return
        try:
            from rembg import new_session  # noqa
        except ImportError as e:
            raise RuntimeError("rembg is not installed. `pip install rembg`") from e
        from rembg import new_session
        self._session = new_session(self.model_name)

    def parse(self, image_bgr: np.ndarray, bbox_xyxy: Optional[np.ndarray] = None) -> HumanParsing:
        self._load()
        from rembg import remove
        rgba = remove(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), session=self._session)
        alpha = np.array(rgba)[..., 3]
        mask = (alpha > 127).astype(np.uint8) * 255
        return HumanParsing(full_mask=mask, part_labels=None, label_names=None)


class DetectorMaskParser(IHumanParser):
    """Trivial parser that re-uses the detector's instance mask.

    Used when nothing better is available — the YOLO seg mask is okay-ish
    for silhouette loss but loses fine boundaries (hair, fingers).
    """

    def __init__(self, detector_mask: np.ndarray):
        self._mask = detector_mask

    def parse(self, image_bgr: np.ndarray, bbox_xyxy: Optional[np.ndarray] = None) -> HumanParsing:
        return HumanParsing(full_mask=self._mask, part_labels=None, label_names=None)


class MockParser(IHumanParser):
    """Approximates a body silhouette as an oval inside the bbox."""

    def parse(self, image_bgr: np.ndarray, bbox_xyxy: Optional[np.ndarray] = None) -> HumanParsing:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if bbox_xyxy is None:
            bbox_xyxy = np.array([w * 0.3, h * 0.05, w * 0.7, h * 0.95], dtype=np.float32)
        cx = int((bbox_xyxy[0] + bbox_xyxy[2]) / 2)
        cy = int((bbox_xyxy[1] + bbox_xyxy[3]) / 2)
        ax = int((bbox_xyxy[2] - bbox_xyxy[0]) / 2)
        ay = int((bbox_xyxy[3] - bbox_xyxy[1]) / 2)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
        return HumanParsing(full_mask=mask, part_labels=None, label_names=None)


# Sapiens-Seg adapter is stubbed. The interface is correct; the inference
# call needs the actual checkpoint loaded via torch.hub or a local file.
class SapiensSegParser(IHumanParser):
    """Sapiens-Seg adapter (Meta 2024). Stubbed pending checkpoint wiring.

    Once the checkpoint file is placed at `models/sapiens/seg/sapiens_0.3b.pt`
    or similar, fill in `_load` and `parse`. The output should populate
    `part_labels` with the 28-class Sapiens body-part labelmap.
    """

    def __init__(self, checkpoint_path: str, cfg: ParsingConfig | None = None):
        self.cfg = cfg or ParsingConfig()
        self.checkpoint_path = checkpoint_path

    def parse(self, image_bgr: np.ndarray, bbox_xyxy: Optional[np.ndarray] = None) -> HumanParsing:
        raise NotImplementedError(
            "SapiensSegParser is a stub. Provide the checkpoint and implement"
            " `parse` to produce a 28-class part label map."
        )
