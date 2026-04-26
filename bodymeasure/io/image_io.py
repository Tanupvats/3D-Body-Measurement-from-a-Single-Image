"""Image loading + EXIF parsing.

EXIF focal-length parsing is non-trivial because phones store the *physical*
focal length in mm, not pixels. To convert we need either the focal length
in 35mm-equivalent (preferred when present) or the sensor diagonal.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ExifTags

from ..core.types import CameraIntrinsics, ImageInput, ViewType
from ..utils.logging import get_logger

logger = get_logger(__name__)


_EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def load_image(path: Path | str) -> tuple[np.ndarray, dict]:
    """Load an image from disk and return (BGR ndarray, EXIF dict)."""
    path = Path(path)
    pil = Image.open(path)
    pil = _apply_exif_orientation(pil)
    rgb = np.array(pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    exif = _read_exif(pil)
    return bgr, exif


def decode_b64_image(b64: str) -> tuple[np.ndarray, dict]:
    """Decode a base64-encoded image (with or without data: prefix)."""
    if "," in b64 and b64.lstrip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(raw))
    pil = _apply_exif_orientation(pil)
    rgb = np.array(pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    exif = _read_exif(pil)
    return bgr, exif


def _apply_exif_orientation(pil: Image.Image) -> Image.Image:
    """Rotate by EXIF orientation tag — phones almost always need this."""
    try:
        exif = pil.getexif()
        orient_tag = _EXIF_TAGS.get("Orientation")
        if orient_tag is None:
            return pil
        orientation = exif.get(orient_tag)
        if orientation == 3:
            return pil.rotate(180, expand=True)
        if orientation == 6:
            return pil.rotate(270, expand=True)
        if orientation == 8:
            return pil.rotate(90, expand=True)
    except Exception:
        pass
    return pil


def _read_exif(pil: Image.Image) -> dict:
    out: dict = {}
    try:
        raw = pil.getexif()
        if not raw:
            return out
        for tag_id, value in raw.items():
            tag = ExifTags.TAGS.get(tag_id, str(tag_id))
            out[tag] = value
    except Exception as e:
        logger.debug(f"EXIF read failed: {e}")
    return out


def estimate_intrinsics_from_exif(
    width: int,
    height: int,
    exif: dict,
    sensor_diag_mm: float = 7.0,
    default_hfov_deg: float = 60.0,
) -> CameraIntrinsics:
    """Try to recover pixel-space focal length from EXIF.

    Order of preference:
    1. FocalLengthIn35mmFilm  (most reliable, normalized to 35mm full-frame)
    2. FocalLength + sensor_diag_mm assumption (decent for phones)
    3. Fall back to default HFoV
    """
    img_diag_px = float(np.sqrt(width**2 + height**2))

    f35 = exif.get("FocalLengthIn35mmFilm")
    if f35:
        try:
            f35_val = float(f35)
            # 35mm film diagonal is 43.27 mm
            f_px = f35_val / 43.27 * img_diag_px
            return CameraIntrinsics(
                fx=f_px, fy=f_px, cx=width / 2.0, cy=height / 2.0,
                width=width, height=height, is_estimated=False,
                source="exif_35mm_equiv",
            )
        except (TypeError, ValueError):
            pass

    fl = exif.get("FocalLength")
    if fl:
        try:
            fl_val = float(fl) if not hasattr(fl, "numerator") else fl.numerator / fl.denominator
            f_px = fl_val / sensor_diag_mm * img_diag_px
            return CameraIntrinsics(
                fx=f_px, fy=f_px, cx=width / 2.0, cy=height / 2.0,
                width=width, height=height, is_estimated=True,
                source=f"exif_focal_mm_assumed_sensor_{sensor_diag_mm}mm",
            )
        except (TypeError, ValueError, AttributeError):
            pass

    return CameraIntrinsics.from_fov(width, height, hfov_deg=default_hfov_deg)


def make_image_input(
    path: Path | str,
    view: ViewType = ViewType.FRONT,
    sensor_diag_mm: float = 7.0,
    default_hfov_deg: float = 60.0,
) -> ImageInput:
    """One-stop loader producing an ImageInput with camera populated."""
    bgr, exif = load_image(path)
    h, w = bgr.shape[:2]
    cam = estimate_intrinsics_from_exif(w, h, exif, sensor_diag_mm, default_hfov_deg)
    return ImageInput(image_bgr=bgr, view=view, camera=cam, exif=exif)
