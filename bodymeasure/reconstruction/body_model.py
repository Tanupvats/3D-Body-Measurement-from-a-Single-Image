

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.types import Gender, SMPLXParams
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BodyModelOutput:
    vertices: "torch.Tensor"  # (B, V, 3)
    joints: "torch.Tensor"    # (B, J, 3)
    faces: np.ndarray         # (F, 3)


class SMPLXModel:
    """Thin wrapper around the official `smplx` package.

    Loads gender-specific models lazily. The neutral model is used when
    gender is unknown — note that this is suboptimal for measurement
    accuracy because shape spaces are gendered.
    """

    NUM_BETAS = 10  # Use 10 by default; some research uses 16 or 300.
    NUM_BODY_JOINTS = 21
    BODY_POSE_DIM = NUM_BODY_JOINTS * 3  # 63

    def __init__(
        self,
        models_dir: Path,
        gender: Gender = Gender.NEUTRAL,
        num_betas: int = NUM_BETAS,
        use_pca_hands: bool = True,
        num_pca_comps: int = 6,
        device: str = "cpu",
    ):
        self.models_dir = Path(models_dir)
        self.gender = gender
        self.num_betas = num_betas
        self.use_pca_hands = use_pca_hands
        self.num_pca_comps = num_pca_comps
        self.device = device
        self._model = None

    @property
    def faces(self) -> np.ndarray:
        if self._model is None:
            self._load()
        return self._model.faces

    def _load(self):
        if self._model is not None:
            return
        try:
            import smplx  # noqa
            import torch  # noqa
        except ImportError as e:
            raise RuntimeError(
                "smplx and torch are required. `pip install smplx torch`"
            ) from e
        import smplx
        import torch
        # smplx expects a `models` directory containing `smplx/SMPLX_*.npz`
        self._model = smplx.create(
            model_path=str(self.models_dir),
            model_type="smplx",
            gender=self.gender.value,
            num_betas=self.num_betas,
            use_pca=self.use_pca_hands,
            num_pca_comps=self.num_pca_comps,
            create_global_orient=False,
            create_body_pose=False,
            create_betas=False,
            create_transl=False,
        ).to(self.device)
        logger.info(f"Loaded SMPL-X gender={self.gender.value} from {self.models_dir}")

    def forward(self, params: SMPLXParams) -> BodyModelOutput:
        self._load()
        import torch
        kwargs = {
            "betas": torch.as_tensor(params.betas, device=self.device, dtype=torch.float32),
            "body_pose": torch.as_tensor(params.body_pose, device=self.device, dtype=torch.float32),
            "global_orient": torch.as_tensor(params.global_orient, device=self.device, dtype=torch.float32),
            "transl": torch.as_tensor(params.transl, device=self.device, dtype=torch.float32),
        }
        if params.left_hand_pose is not None:
            kwargs["left_hand_pose"] = torch.as_tensor(params.left_hand_pose, device=self.device, dtype=torch.float32)
        if params.right_hand_pose is not None:
            kwargs["right_hand_pose"] = torch.as_tensor(params.right_hand_pose, device=self.device, dtype=torch.float32)
        if params.expression is not None:
            kwargs["expression"] = torch.as_tensor(params.expression, device=self.device, dtype=torch.float32)
        if params.jaw_pose is not None:
            kwargs["jaw_pose"] = torch.as_tensor(params.jaw_pose, device=self.device, dtype=torch.float32)
        out = self._model(**kwargs, return_verts=True)
        return BodyModelOutput(vertices=out.vertices, joints=out.joints, faces=self._model.faces)

    def forward_torch(self, betas, body_pose, global_orient, transl, **extra) -> BodyModelOutput:
        """Differentiable forward. All inputs must be torch tensors w/ grad as needed."""
        self._load()
        out = self._model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            return_verts=True,
            **extra,
        )
        return BodyModelOutput(vertices=out.vertices, joints=out.joints, faces=self._model.faces)


class SyntheticBodyModel:
    """A capsule-with-limbs mesh that varies smoothly with `betas`.

    DO NOT use for any user-facing measurement. Exists so the rest of the
    pipeline can be tested end-to-end without the proprietary SMPL-X
    weights. Mesh topology is fixed; vertices scale and shift with betas
    in a way that's geometrically plausible enough to exercise the
    measurement and optimization code paths.
    """

    NUM_BETAS = 10
    BODY_POSE_DIM = 63
    _CACHED: Optional["SyntheticBodyModel"] = None

    def __init__(self, n_height: int = 24, n_circ: int = 32):
        self.n_height = n_height
        self.n_circ = n_circ
        self._template_v, self._faces = self._build_template()
        # Precompute beta basis: 10 directions in vertex space.
        rng = np.random.default_rng(seed=42)
        self._beta_basis = rng.normal(0, 0.01, size=(self.NUM_BETAS, self._template_v.shape[0], 3)).astype(np.float32)
        # Make beta_0 control overall scale, beta_1 control width
        self._beta_basis[0] = self._template_v * 0.05
        radial = self._template_v.copy()
        radial[:, 1] = 0
        self._beta_basis[1] = radial * 0.05

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    def _build_template(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a layered cylindrical mesh roughly shaped like a person."""
        # Profile: radius along height, normalized height in [0,1] -> radius in metres.
        h_norm = np.linspace(0.0, 1.0, self.n_height)
        # Roughly: feet thin, calves, knees, thighs, hips wide, waist narrower,
        # chest wider, neck narrow, head wider.
        ctrl = np.array([
            (0.00, 0.05),  # ankle
            (0.10, 0.06),  # calf
            (0.25, 0.09),  # knee/lower thigh
            (0.40, 0.12),  # thigh
            (0.50, 0.16),  # hip
            (0.62, 0.13),  # waist
            (0.72, 0.18),  # chest
            (0.82, 0.10),  # shoulder/neck
            (0.90, 0.11),  # head bottom
            (1.00, 0.10),  # head top
        ])
        radius = np.interp(h_norm, ctrl[:, 0], ctrl[:, 1])
        # Y axis is up. Total template height ~1.7 m.
        y = h_norm * 1.7

        # Build vertices ring by ring.
        verts = []
        for hi in range(self.n_height):
            for ci in range(self.n_circ):
                theta = 2 * math.pi * ci / self.n_circ
                x = radius[hi] * math.cos(theta)
                z = radius[hi] * math.sin(theta)
                verts.append((x, y[hi], z))
        verts = np.array(verts, dtype=np.float32)

        # Build faces (two tris per quad).
        faces = []
        for hi in range(self.n_height - 1):
            for ci in range(self.n_circ):
                a = hi * self.n_circ + ci
                b = hi * self.n_circ + (ci + 1) % self.n_circ
                c = (hi + 1) * self.n_circ + ci
                d = (hi + 1) * self.n_circ + (ci + 1) % self.n_circ
                faces.append((a, b, c))
                faces.append((b, d, c))
        faces = np.array(faces, dtype=np.int32)
        return verts, faces

    def forward_np(self, betas: np.ndarray, transl: np.ndarray) -> np.ndarray:
        """betas: (B,10), transl: (B,3) -> verts (B, V, 3)"""
        deltas = (betas[..., None, None] * self._beta_basis[None, ...]).sum(axis=1)
        v = self._template_v[None, ...] + deltas + transl[:, None, :]
        return v.astype(np.float32)
