

from __future__ import annotations

from typing import Optional

from .config import Config
from .interfaces import (
    IBodyOptimizer,
    IBodyRegressor,
    ICameraEstimator,
    IHumanParser,
    IPersonDetector,
    IPoseEstimator,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


def make_camera_estimator(cfg: Config) -> ICameraEstimator:
    from ..camera.estimator import ExifCameraEstimator
    return ExifCameraEstimator(cfg.camera)


def make_detector(cfg: Config) -> IPersonDetector:
    name = cfg.detector_name
    if name == "yolo":
        from ..segmentation.detector import YoloPersonDetector
        return YoloPersonDetector(cfg=cfg.detection)
    if name == "mock":
        from ..segmentation.detector import MockPersonDetector
        return MockPersonDetector()
    raise ValueError(f"unknown detector: {name}")


def make_parser(cfg: Config) -> IHumanParser:
    name = cfg.parser_name
    if name == "rembg":
        from ..segmentation.parser import RembgParser
        return RembgParser(cfg=cfg.parsing)
    if name == "sapiens":
        from ..segmentation.parser import SapiensSegParser
        ckpt = cfg.model_path("sapiens", "seg", "sapiens_0.3b.pt")
        return SapiensSegParser(str(ckpt), cfg=cfg.parsing)
    if name == "mock":
        from ..segmentation.parser import MockParser
        return MockParser()
    raise ValueError(f"unknown parser: {name}")


def make_pose(cfg: Config) -> IPoseEstimator:
    name = cfg.pose_name
    if name == "ultralytics":
        from ..pose.estimator import UltralyticsPose
        return UltralyticsPose(cfg=cfg.pose)
    if name == "rtmpose":
        from ..pose.estimator import RTMPoseEstimator
        cfg_path = cfg.model_path("rtmpose", "rtmpose-l_8xb256-420e_coco-256x192.py")
        ckpt_path = cfg.model_path("rtmpose", "rtmpose-l_8xb256-420e_coco-256x192-eaeb96bc_20230126.pth")
        return RTMPoseEstimator(str(cfg_path), str(ckpt_path), cfg=cfg.pose)
    if name == "mock":
        from ..pose.estimator import MockPoseEstimator
        return MockPoseEstimator()
    raise ValueError(f"unknown pose model: {name}")


def make_regressor(cfg: Config) -> IBodyRegressor:
    name = cfg.regressor_name
    if name == "heuristic":
        from ..reconstruction.regressor import HeuristicRegressor
        return HeuristicRegressor()
    if name == "multihmr":
        from ..reconstruction.regressor import MultiHMRAdapter
        ckpt = cfg.model_path("multihmr", "multiHMR_896_L.pt")
        return MultiHMRAdapter(str(ckpt))
    if name == "tokenhmr":
        from ..reconstruction.regressor import TokenHMRAdapter
        ckpt = cfg.model_path("tokenhmr", "tokenhmr.pt")
        return TokenHMRAdapter(str(ckpt))
    if name == "sapiens":
        from ..reconstruction.regressor import SapiensPoseAdapter
        ckpt = cfg.model_path("sapiens", "pose", "sapiens_pose_0.3b.pt")
        return SapiensPoseAdapter(str(ckpt))
    raise ValueError(f"unknown regressor: {name}")


def make_body_model(cfg: Config, gender: str = "neutral", device: str = "cpu"):
    """Return a body-model object that supports both `forward_torch` (real)
    or `forward_np` (synthetic) depending on availability."""
    from ..reconstruction.body_model import SMPLXModel, SyntheticBodyModel
    from ..core.types import Gender
    if cfg.body_model in ("smplx", "smpl"):
        models_dir = cfg.model_path()
        try:
            return SMPLXModel(models_dir=models_dir, gender=Gender(gender), device=device)
        except Exception as e:
            logger.warning(f"SMPL-X load failed ({e}); falling back to synthetic body model")
            return SyntheticBodyModel()
    if cfg.body_model == "synthetic":
        return SyntheticBodyModel()
    raise ValueError(f"unknown body model: {cfg.body_model}")


def make_optimizer(cfg: Config, body_model, device: str = "cpu") -> IBodyOptimizer:
    from ..reconstruction.optimizer import SMPLifyXOptimizer
    return SMPLifyXOptimizer(
        body_model=body_model,
        cfg=cfg.optimizer,
        device=device,
        enable_silhouette=False,  # opt-in via config later
    )
