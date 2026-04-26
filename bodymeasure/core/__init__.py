"""Public surface.

Pipeline is imported lazily so that lightweight code paths (schema
validation, config loading, image IO) don't pull in trimesh/torch.
"""

from .config import Config, default_config
from .types import (
    CameraIntrinsics,
    FittedBody,
    Gender,
    HumanParsing,
    ImageInput,
    Keypoints2D,
    Measurement,
    PersonDetection,
    PredictionResult,
    QCReport,
    SMPLXParams,
    ViewType,
)


def __getattr__(name):
    if name in ("Pipeline", "PipelineDebugInfo"):
        from .pipeline import Pipeline, PipelineDebugInfo
        return {"Pipeline": Pipeline, "PipelineDebugInfo": PipelineDebugInfo}[name]
    raise AttributeError(f"module 'bodymeasure.core' has no attribute {name!r}")


__all__ = [
    "Config", "default_config", "Pipeline", "PipelineDebugInfo",
    "CameraIntrinsics", "FittedBody", "Gender", "HumanParsing",
    "ImageInput", "Keypoints2D", "Measurement", "PersonDetection",
    "PredictionResult", "QCReport", "SMPLXParams", "ViewType",
]
