import sys
import os
# --- FIX: Ensure the project root is in the python path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import base64
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Depends

from api.schemas import MeasurementRequest, MeasurementResponse
from core import (
    BodyMeasurementPipeline, 
    MockSegmentationModel, 
    MockPoseModel, 
    MockSMPLReconstructor
)

app = FastAPI(title="3D Body Measurement API", version="2.0.0")

def get_ml_pipeline() -> BodyMeasurementPipeline:
    return BodyMeasurementPipeline(
        seg_model=MockSegmentationModel(),
        pose_model=MockPoseModel(),
        smpl_model=MockSMPLReconstructor()
    )

@app.post("/predict-measurements", response_model=MeasurementResponse)
async def predict_measurements(req: MeasurementRequest, pipeline: BodyMeasurementPipeline = Depends(get_ml_pipeline)):
    try:
        results = pipeline.predict(
            image_b64=req.image_base64,
            height_cm=req.height_cm,
            gender=req.gender or "neutral"
        )
        return MeasurementResponse(
            **results,
            status="success",
            message="Measurements extracted successfully."
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)