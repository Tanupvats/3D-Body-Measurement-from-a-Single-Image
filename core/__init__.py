"""
Core module initialization.
Exposes the primary pipeline and geometry functions for easy importing.
"""

from .pipeline import (
    BodyMeasurementPipeline,
    ISegmentationModel,
    IPoseModel,
    ISMPlReconstructor,
    MockSegmentationModel,
    MockPoseModel,
    MockSMPLReconstructor
)
from .mesh_geometry import (
    scale_mesh, 
    get_circumference_at_y, 
    extract_measurements
)