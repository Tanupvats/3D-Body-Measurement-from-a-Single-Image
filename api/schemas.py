from pydantic import BaseModel, Field
from typing import Optional

class MeasurementRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded front-view RGB image")
    height_cm: float = Field(..., gt=50, lt=250, description="User's known height in cm")
    gender: Optional[str] = Field("neutral", description="Gender prior")
    focal_length: Optional[float] = Field(None, description="Optional focal length")

class MeasurementResponse(BaseModel):
    chest_circumference: float
    waist_circumference: float
    hip_circumference: float
    arm_length: float
    thigh_circumference: float
    inseam_length: float
    shoulder_width: float
    status: str
    message: str