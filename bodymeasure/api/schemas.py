"""API request/response schemas."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MeasurementOut(BaseModel):
    value_cm: Optional[float]
    uncertainty_cm: float
    method: str
    notes: str = ""


class QCOut(BaseModel):
    passed: bool
    warnings: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    measurements: dict[str, MeasurementOut]
    qc: QCOut
    metadata: dict


class PredictRequest(BaseModel):
    """Used when sending base64-encoded images instead of multipart upload."""
    front_image_b64: str
    side_image_b64: Optional[str] = None
    height_cm: float = Field(..., gt=50.0, lt=250.0)
    gender: str = Field("neutral", pattern="^(male|female|neutral)$")


class HealthResponse(BaseModel):
    status: str
    version: str
    pipeline_loaded: bool
    corrector_active: bool
