
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.interfaces import IBodyRegressor
from ..core.types import CameraIntrinsics, Gender, PersonDetection, SMPLXParams
from ..utils.logging import get_logger

logger = get_logger(__name__)


class HeuristicRegressor(IBodyRegressor):
    """Zero-pose, zero-shape, depth-from-bbox-height initializer.

    Reasoning:
      - betas=0 puts us at the SMPL-X mean body. For optimization init that's
        usually fine; the optimizer will move shape from there.
      - body_pose=0 is T-pose. For front-facing photos this is closer to
        a typical standing pose than e.g. random init.
      - global_orient: zero rotation (model facing +Z).
      - transl.z is computed so that a person of height H_world
        projects to the bbox height H_pix under the given camera.

    This last bit is what makes the init usable instead of garbage:
    placing the body at roughly the right depth means the reprojection
    loss starts close to the right basin.
    """

    # Approximate height of the SMPL-X mean body in metres (with betas=0).
    # The exact value depends on the pose; for the mean body in T-pose this
    # is ~1.74m for the male model, ~1.62m for female, ~1.69m neutral.
    APPROX_MEAN_HEIGHTS = {
        Gender.NEUTRAL: 1.69,
        Gender.MALE: 1.74,
        Gender.FEMALE: 1.62,
    }

    def __init__(self, num_betas: int = 10):
        self.num_betas = num_betas

    def regress(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: Optional[np.ndarray],
        camera: CameraIntrinsics,
        gender: Gender,
    ) -> SMPLXParams:
        h_img, w_img = image_bgr.shape[:2]
        if bbox_xyxy is None:
            bbox_xyxy = np.array([w_img * 0.3, h_img * 0.05, w_img * 0.7, h_img * 0.95])
        bbox_h_px = float(bbox_xyxy[3] - bbox_xyxy[1])
        bbox_cx = float((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0)
        bbox_cy = float((bbox_xyxy[1] + bbox_xyxy[3]) / 2.0)

        # Depth: from similar triangles, person_height_world / depth = bbox_h_px / fy
        h_world = self.APPROX_MEAN_HEIGHTS[gender]
        z_cam = max(0.5, h_world * camera.fy / max(1.0, bbox_h_px))

        # XY: back-project the bbox center to 3D at depth z_cam
        x_cam = (bbox_cx - camera.cx) * z_cam / camera.fx
        y_cam = (bbox_cy - camera.cy) * z_cam / camera.fy

        # SMPL-X uses Y-up; image space is Y-down. Flip Y.
        # Also SMPL-X origin is at the pelvis, not the head; offset accordingly.
        transl = np.array([[x_cam, -y_cam, z_cam]], dtype=np.float32)

        return SMPLXParams(
            betas=np.zeros((1, self.num_betas), dtype=np.float32),
            body_pose=np.zeros((1, 63), dtype=np.float32),
            global_orient=np.zeros((1, 3), dtype=np.float32),
            transl=transl,
            gender=gender,
        )


class MultiHMRAdapter(IBodyRegressor):
    """Adapter for Multi-HMR (Baradel et al., 2024). Stubbed.

    To enable:
      1. Clone https://github.com/naver/multi-hmr
      2. Download the checkpoint (e.g. multiHMR_896_L.pt)
      3. Implement `_load` to instantiate the model
      4. Implement `regress` to crop, run inference, and translate the
         output to a SMPLXParams
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None

    def regress(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: Optional[np.ndarray],
        camera: CameraIntrinsics,
        gender: Gender,
    ) -> SMPLXParams:
        raise NotImplementedError(
            "MultiHMRAdapter is a stub. See module docstring for wiring instructions."
        )


class TokenHMRAdapter(IBodyRegressor):
    """TokenHMR (Dwivedi et al., 2024). Stubbed."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device

    def regress(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: Optional[np.ndarray],
        camera: CameraIntrinsics,
        gender: Gender,
    ) -> SMPLXParams:
        raise NotImplementedError("TokenHMRAdapter is a stub.")


class SapiensPoseAdapter(IBodyRegressor):
    """Sapiens-Pose (Khirodkar et al., Meta, 2024). Stubbed.

    Currently the strongest open HMR result on most benchmarks. Adapter
    wiring depends on whether you use the HuggingFace mirror or the
    official release.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device

    def regress(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: Optional[np.ndarray],
        camera: CameraIntrinsics,
        gender: Gender,
    ) -> SMPLXParams:
        raise NotImplementedError("SapiensPoseAdapter is a stub.")
