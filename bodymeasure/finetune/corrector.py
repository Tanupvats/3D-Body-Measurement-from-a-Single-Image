

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.types import Gender, SMPLXParams
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Feature vector layout — keep in lockstep with `features.py`.
FEATURE_DIM = 10 + 1 + 1 + 1 + 3   # betas(10) + height_norm + wh_ratio + sh_hip + gender_onehot(3)
NUM_BETAS = 10


class ShapeCorrector:
    """Pluggable shape-residual head. Identity by default."""

    def __init__(self, checkpoint_path: Optional[Path] = None, device: str = "cpu"):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device
        self._model = None
        self._loaded = False
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.load(self.checkpoint_path)
        else:
            logger.info("ShapeCorrector: no checkpoint, behaving as identity (no correction)")

    def load(self, path: Path):
        try:
            import torch  # noqa
        except ImportError:
            logger.warning("torch not installed; shape corrector disabled")
            return
        import torch
        ckpt = torch.load(path, map_location=self.device)
        self._model = _build_mlp().to(self.device)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()
        self._loaded = True
        logger.info(f"ShapeCorrector: loaded checkpoint {path}")

    def __call__(self, params: SMPLXParams, features: np.ndarray) -> SMPLXParams:
        if not self._loaded or self._model is None:
            return params
        import torch
        with torch.no_grad():
            x = torch.tensor(features[None, :], dtype=torch.float32, device=self.device)
            delta = self._model(x).cpu().numpy()[0]
        new_betas = params.betas + delta[None, :]
        return SMPLXParams(
            betas=new_betas.astype(np.float32),
            body_pose=params.body_pose,
            global_orient=params.global_orient,
            transl=params.transl,
            left_hand_pose=params.left_hand_pose,
            right_hand_pose=params.right_hand_pose,
            jaw_pose=params.jaw_pose,
            expression=params.expression,
            gender=params.gender,
        )


def _build_mlp():
    """Small MLP: feature_dim -> 64 -> 64 -> num_betas."""
    import torch
    from torch import nn
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, NUM_BETAS),
    )


def gender_onehot(gender: Gender) -> np.ndarray:
    """3-dim one-hot."""
    if gender == Gender.MALE:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if gender == Gender.FEMALE:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def build_feature_vector(
    betas: np.ndarray,        # (1, 10) or (10,)
    height_cm: float,
    bbox_w_h_ratio: float,
    shoulder_hip_ratio: float,
    gender: Gender,
) -> np.ndarray:
    betas = betas.flatten()[:NUM_BETAS]
    height_norm = (height_cm - 170.0) / 25.0  # rough normalization
    feat = np.concatenate([
        betas,
        np.array([height_norm], dtype=np.float32),
        np.array([bbox_w_h_ratio], dtype=np.float32),
        np.array([shoulder_hip_ratio], dtype=np.float32),
        gender_onehot(gender),
    ]).astype(np.float32)
    assert feat.shape[0] == FEATURE_DIM, f"feature dim mismatch {feat.shape[0]} vs {FEATURE_DIM}"
    return feat
