"""Production FastAPI server.

Loads the pipeline ONCE at startup. Subsequent requests reuse it.

Endpoints:
  GET  /health            : health + version + corrector status
  POST /predict           : multipart upload (front, optional side)
  POST /predict_b64       : JSON payload with base64-encoded images

Run with:
    uvicorn bodymeasure.api.main:app --host 0.0.0.0 --port 8000

Configuration via env vars:
    BODYMEASURE_CONFIG       path to YAML config (default: built-in defaults)
    BODYMEASURE_CORRECTOR    path to corrector.pt (optional)
    BODYMEASURE_DEVICE       cpu | cuda
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .. import __version__
from ..core.config import Config, default_config
from ..core.pipeline import Pipeline
from ..core.types import Gender
from ..finetune.corrector import ShapeCorrector
from ..io.image_io import decode_b64_image
from ..utils.logging import configure, get_logger
from .schemas import HealthResponse, MeasurementOut, PredictRequest, PredictResponse, QCOut

configure()
logger = get_logger(__name__)

app = FastAPI(
    title="bodymeasure API",
    version=__version__,
    description="Image-based body measurement service.",
)

_PIPELINE: Optional[Pipeline] = None
_CORRECTOR: Optional[ShapeCorrector] = None


def _load_pipeline() -> Pipeline:
    global _PIPELINE, _CORRECTOR
    if _PIPELINE is not None:
        return _PIPELINE
    config_path = os.environ.get("BODYMEASURE_CONFIG")
    cfg = Config.from_yaml(Path(config_path)) if config_path else default_config()
    corrector_path = os.environ.get("BODYMEASURE_CORRECTOR")
    device = os.environ.get("BODYMEASURE_DEVICE", "cpu")
    _CORRECTOR = ShapeCorrector(Path(corrector_path), device=device) if corrector_path else None
    _PIPELINE = Pipeline(cfg, device=device, corrector=_CORRECTOR)
    logger.info(f"pipeline loaded (corrector={'on' if _CORRECTOR else 'off'}, device={device})")
    return _PIPELINE


@app.on_event("startup")
def _startup():
    try:
        _load_pipeline()
    except Exception as e:
        logger.error(f"pipeline failed to load at startup: {e} — will retry on first request")


@app.get("/health", response_model=HealthResponse)
def health():
    loaded = _PIPELINE is not None
    corrector_active = bool(
        _CORRECTOR is not None and getattr(_CORRECTOR, "_loaded", False)
    )
    return HealthResponse(
        status="ok",
        version=__version__,
        pipeline_loaded=loaded,
        corrector_active=corrector_active,
    )


def _result_to_response(result) -> PredictResponse:
    measurements = {
        m.name: MeasurementOut(
            value_cm=None if (m.value_cm != m.value_cm) else round(float(m.value_cm), 2),
            uncertainty_cm=round(float(m.uncertainty_cm), 2),
            method=m.method,
            notes=m.notes,
        )
        for m in result.measurements
    }
    return PredictResponse(
        measurements=measurements,
        qc=QCOut(
            passed=result.qc.passed,
            warnings=result.qc.warnings,
            failures=result.qc.failures,
            metrics={k: float(v) for k, v in result.qc.metrics.items()},
        ),
        metadata=result.metadata,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    front: UploadFile = File(...),
    side: Optional[UploadFile] = File(None),
    height_cm: float = Form(...),
    gender: str = Form("neutral"),
):
    """Multipart upload endpoint."""
    if gender not in ("male", "female", "neutral"):
        raise HTTPException(status_code=400, detail="gender must be male|female|neutral")
    if not (50.0 < height_cm < 250.0):
        raise HTTPException(status_code=400, detail="height_cm out of plausible range")

    pipeline = _load_pipeline()

    # Save uploads to temp files so the loader can read EXIF properly
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        front_bytes = await front.read()
        front_path = td_path / "front"
        front_path.write_bytes(front_bytes)
        side_path: Optional[Path] = None
        if side is not None:
            side_bytes = await side.read()
            side_path = td_path / "side"
            side_path.write_bytes(side_bytes)
        try:
            result = pipeline.predict(
                front_image_path=front_path,
                side_image_path=side_path,
                height_cm=height_cm,
                gender=Gender(gender),
            )
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.exception("pipeline crash")
            raise HTTPException(status_code=500, detail=f"internal error: {e}")
    return _result_to_response(result)


@app.post("/predict_b64", response_model=PredictResponse)
async def predict_b64(req: PredictRequest):
    pipeline = _load_pipeline()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        front_bgr, _ = decode_b64_image(req.front_image_b64)
        # Save as PNG so EXIF parsing path is consistent (base64 has no EXIF anyway)
        import cv2
        front_path = td_path / "front.png"
        cv2.imwrite(str(front_path), front_bgr)
        side_path: Optional[Path] = None
        if req.side_image_b64:
            side_bgr, _ = decode_b64_image(req.side_image_b64)
            side_path = td_path / "side.png"
            cv2.imwrite(str(side_path), side_bgr)
        try:
            result = pipeline.predict(
                front_image_path=front_path,
                side_image_path=side_path,
                height_cm=req.height_cm,
                gender=Gender(req.gender),
            )
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
    return _result_to_response(result)
