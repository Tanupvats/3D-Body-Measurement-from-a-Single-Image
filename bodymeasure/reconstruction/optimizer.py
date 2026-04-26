

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.config import OptimizerConfig
from ..core.interfaces import IBodyOptimizer
from ..core.types import (
    CameraIntrinsics,
    FittedBody,
    Gender,
    HumanParsing,
    ImageInput,
    Keypoints2D,
    SMPLXParams,
    ViewType,
)
from ..utils.logging import get_logger
from .body_model import SMPLXModel

logger = get_logger(__name__)


# Mapping from COCO-17 to SMPL-X joint indices.
# SMPL-X joint order is OpenPose-derived; the body subset has well-defined
# correspondences. These indices follow the SMPL-X joint regressor output.
# (full body subset: 0=pelvis, 1=L_hip, ..., we pick the joints that
#  correspond to COCO-17.)
COCO17_TO_SMPLX = {
    "nose":           24,  # SMPL-X "nose"
    "left_eye":       26,
    "right_eye":      25,
    "left_ear":       28,
    "right_ear":      27,
    "left_shoulder":  16,
    "right_shoulder": 17,
    "left_elbow":     18,
    "right_elbow":    19,
    "left_wrist":     20,
    "right_wrist":    21,
    "left_hip":       1,
    "right_hip":      2,
    "left_knee":      4,
    "right_knee":     5,
    "left_ankle":     7,
    "right_ankle":    8,
}
COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _project_perspective_torch(points_3d, K):
    """Project (B, N, 3) camera-space points through (3,3) intrinsics.

    Returns (B, N, 2) image-space coords. Z is assumed positive in front.
    """
    import torch
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = points_3d[..., 2].clamp(min=1e-3)
    u = points_3d[..., 0] * fx / z + cx
    v = points_3d[..., 1] * fy / z + cy
    return torch.stack([u, v], dim=-1)


@dataclass
class _OptimState:
    betas: "torch.Tensor"
    body_pose: "torch.Tensor"
    global_orients: list   # one per view
    transls: list


class SMPLifyXOptimizer(IBodyOptimizer):
    """Refines SMPL-X parameters against multi-view 2D evidence."""

    def __init__(
        self,
        body_model: SMPLXModel,
        cfg: OptimizerConfig | None = None,
        device: str = "cpu",
        enable_silhouette: bool = False,
    ):
        self.body_model = body_model
        self.cfg = cfg or OptimizerConfig()
        self.device = device
        self.enable_silhouette = enable_silhouette
        if enable_silhouette:
            try:
                import pytorch3d  # noqa
            except ImportError:
                logger.warning(
                    "Silhouette loss requested but pytorch3d not installed; disabling it."
                )
                self.enable_silhouette = False

    def fit(
        self,
        views: list[ImageInput],
        parsings: list[HumanParsing],
        keypoints: list[Keypoints2D],
        cameras: list[CameraIntrinsics],
        init_params: SMPLXParams,
        height_cm: Optional[float] = None,
        gender: Gender = Gender.NEUTRAL,
    ) -> FittedBody:
        import torch
        n_views = len(views)
        assert len(parsings) == n_views and len(keypoints) == n_views and len(cameras) == n_views, \
            "views/parsings/keypoints/cameras must have the same length"

        # Initialize torch params
        betas = torch.tensor(init_params.betas, dtype=torch.float32, device=self.device, requires_grad=True)
        body_pose = torch.tensor(init_params.body_pose, dtype=torch.float32, device=self.device, requires_grad=True)
        # One global_orient and transl per view (subject may have moved relative to camera between shots).
        global_orients = [
            torch.tensor(init_params.global_orient.copy(), dtype=torch.float32, device=self.device, requires_grad=True)
            for _ in range(n_views)
        ]
        transls = [
            torch.tensor(init_params.transl.copy(), dtype=torch.float32, device=self.device, requires_grad=True)
            for _ in range(n_views)
        ]
        # Apply view-prior rotation: side views are rotated 90deg about Y vs front.
        with torch.no_grad():
            for i, v in enumerate(views):
                if v.view == ViewType.SIDE:
                    global_orients[i][0, 1] += np.pi / 2.0  # axis-angle, around y
                elif v.view == ViewType.BACK:
                    global_orients[i][0, 1] += np.pi

        # Pre-compute per-view target keypoints + confidences as tensors
        target_kpts = []
        kpt_conf = []
        Ks = []
        for kp, cam in zip(keypoints, cameras):
            target_kpts.append(torch.tensor(kp.xy, dtype=torch.float32, device=self.device))
            target_kpts[-1] = target_kpts[-1].unsqueeze(0)  # (1, 17, 2)
            kpt_conf.append(torch.tensor(kp.confidence, dtype=torch.float32, device=self.device).unsqueeze(0))
            Ks.append(torch.tensor(cam.K, dtype=torch.float32, device=self.device))

        smplx_kpt_idx = torch.tensor(
            [COCO17_TO_SMPLX[n] for n in COCO17_NAMES],
            dtype=torch.long, device=self.device,
        )

        height_target_m = (height_cm / 100.0) if height_cm else None

        cfg = self.cfg

        def _stage(stage_idx: int, lr: float, n_iter: int,
                   w_kpt: float, w_silh: float, w_shape: float, w_pose: float,
                   freeze_shape: bool):
            params_to_opt = [body_pose, *global_orients, *transls]
            if not freeze_shape:
                params_to_opt = [betas] + params_to_opt
            opt = torch.optim.Adam(params_to_opt, lr=lr)
            for it in range(n_iter):
                opt.zero_grad()
                total = torch.zeros((), device=self.device)
                fit_metric = 0.0

                # Per-view forward passes (shape shared, pose shared, transl/orient per view).
                for i, (v, kp_xy, kp_c, K) in enumerate(zip(views, target_kpts, kpt_conf, Ks)):
                    out = self.body_model.forward_torch(
                        betas=betas,
                        body_pose=body_pose,
                        global_orient=global_orients[i],
                        transl=transls[i],
                    )
                    joints_3d = out.joints  # (1, J, 3)
                    sel_joints = joints_3d[:, smplx_kpt_idx, :]  # (1, 17, 3)
                    proj = _project_perspective_torch(sel_joints, K)  # (1, 17, 2)
                    diff = proj - kp_xy
                    # Robust: use per-keypoint confidence as weight, plus a soft Geman-McClure
                    # to keep outlier joints from dominating.
                    sq = (diff ** 2).sum(dim=-1)  # (1, 17)
                    rho = sq / (sq + 100.0)       # Geman-McClure-ish, 100 px^2 saturation
                    kpt_loss = (rho * kp_c).mean()
                    total = total + w_kpt * kpt_loss
                    fit_metric += float(kpt_loss.detach().cpu())

                    if self.enable_silhouette and w_silh > 0 and parsings[i].full_mask is not None:
                        sil_loss = self._silhouette_loss_torch(
                            out.vertices, self.body_model.faces, parsings[i].full_mask, cameras[i],
                        )
                        total = total + w_silh * sil_loss

                # Priors
                shape_reg = (betas ** 2).sum()
                pose_reg = (body_pose ** 2).sum()
                total = total + w_shape * shape_reg + w_pose * pose_reg

                # Height constraint — measured on the canonical (un-translated) mesh.
                if height_target_m is not None:
                    out0 = self.body_model.forward_torch(
                        betas=betas,
                        body_pose=torch.zeros_like(body_pose),
                        global_orient=torch.zeros_like(global_orients[0]),
                        transl=torch.zeros_like(transls[0]),
                    )
                    v = out0.vertices[0]  # (V, 3)
                    height_pred = v[:, 1].max() - v[:, 1].min()
                    h_diff = height_pred - height_target_m
                    h_diff_clamped = torch.where(
                        h_diff.abs() > cfg.height_tol_cm / 100.0,
                        h_diff,
                        torch.zeros_like(h_diff),
                    )
                    total = total + cfg.height_constraint_weight * (h_diff_clamped ** 2)

                total.backward()
                torch.nn.utils.clip_grad_norm_(params_to_opt, cfg.grad_clip_norm)
                opt.step()

                if it == 0 or it == n_iter - 1 or (it + 1) % 25 == 0:
                    logger.debug(
                        f"stage {stage_idx} iter {it+1}/{n_iter} "
                        f"loss={float(total.detach()):.4f} kpt_metric={fit_metric:.4f}"
                    )

            return float(total.detach().cpu())

        # Run schedule
        _stage(1, cfg.stage1_lr, cfg.stage1_iters,
               cfg.stage1_w_kpt, cfg.stage1_w_silh, cfg.stage1_w_shape_reg, cfg.stage1_w_pose_reg,
               freeze_shape=True)
        _stage(2, cfg.stage2_lr, cfg.stage2_iters,
               cfg.stage2_w_kpt, cfg.stage2_w_silh, cfg.stage2_w_shape_reg, cfg.stage2_w_pose_reg,
               freeze_shape=False)
        final_loss = _stage(
            3, cfg.stage3_lr, cfg.stage3_iters,
            cfg.stage3_w_kpt, cfg.stage3_w_silh, cfg.stage3_w_shape_reg, cfg.stage3_w_pose_reg,
            freeze_shape=False,
        )

        # Build canonical (un-translated, identity-orient) posed mesh for measurement.
        with torch.no_grad():
            out_canon = self.body_model.forward_torch(
                betas=betas,
                body_pose=torch.zeros_like(body_pose),  # T-pose for measurement
                global_orient=torch.zeros_like(global_orients[0]),
                transl=torch.zeros_like(transls[0]),
            )
            v_canon = out_canon.vertices[0].cpu().numpy()
            j_canon = out_canon.joints[0].cpu().numpy()

        params_out = SMPLXParams(
            betas=betas.detach().cpu().numpy(),
            body_pose=body_pose.detach().cpu().numpy(),
            global_orient=global_orients[0].detach().cpu().numpy(),
            transl=transls[0].detach().cpu().numpy(),
            gender=gender,
        )

        # Determine faces (numpy)
        faces = self.body_model.faces
        if hasattr(faces, "cpu"):
            faces = faces.cpu().numpy()
        faces = np.asarray(faces, dtype=np.int32)

        return FittedBody(
            params=params_out,
            vertices_m=v_canon,
            faces=faces,
            joints_m=j_canon,
            camera=cameras[0],
            fit_loss=final_loss,
        )

    def _silhouette_loss_torch(self, vertices, faces, mask, camera):
        """Render the mesh silhouette and compare to mask via 1 - IoU.

        Pytorch3D-based; only called if pytorch3d is installed.
        """
        import torch
        from pytorch3d.renderer import (
            FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer,
            MeshRenderer, SoftSilhouetteShader, BlendParams,
        )
        from pytorch3d.structures import Meshes

        if isinstance(faces, np.ndarray):
            faces_t = torch.tensor(faces.astype(np.int64), device=self.device)
        else:
            faces_t = faces.long()

        meshes = Meshes(verts=[vertices[0]], faces=[faces_t])
        # Approximate FoV from intrinsics (only need fov for SoftSilhouetteShader).
        fov = float(2.0 * np.degrees(np.arctan2(camera.height / 2.0, camera.fy)))
        cams = FoVPerspectiveCameras(device=self.device, fov=fov)
        raster = MeshRasterizer(
            cameras=cams,
            raster_settings=RasterizationSettings(
                image_size=self.cfg.render_image_size,
                blur_radius=np.log(1.0 / 1e-4 - 1.0) * 1e-5,
                faces_per_pixel=50,
            ),
        )
        renderer = MeshRenderer(rasterizer=raster, shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4)))
        rendered = renderer(meshes)[..., 3]  # (1, H, W) silhouette
        # Resize target mask to render size
        import cv2
        target = cv2.resize(mask, (self.cfg.render_image_size, self.cfg.render_image_size),
                            interpolation=cv2.INTER_NEAREST)
        target_t = torch.tensor((target > 127).astype(np.float32), device=self.device).unsqueeze(0)
        # Soft IoU
        inter = (rendered * target_t).sum()
        union = rendered.sum() + target_t.sum() - inter + 1e-6
        return 1.0 - inter / union
