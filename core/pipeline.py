import base64
import cv2
import numpy as np
import trimesh
from abc import ABC, abstractmethod
from typing import Dict, Tuple

# Relative imports within the core package
from .mesh_geometry import scale_mesh, extract_measurements

class ISegmentationModel(ABC):
    @abstractmethod
    def segment(self, img: np.ndarray) -> np.ndarray:
        pass

class IPoseModel(ABC):
    @abstractmethod
    def estimate_pose(self, img: np.ndarray) -> Dict[str, Tuple[float, float]]:
        pass

class ISMPlReconstructor(ABC):
    @abstractmethod
    def reconstruct(self, img: np.ndarray, mask: np.ndarray, keypoints: dict) -> trimesh.Trimesh:
        pass

class MockSegmentationModel(ISegmentationModel):
    def segment(self, img: np.ndarray) -> np.ndarray:
        return np.ones(img.shape[:2], dtype=np.uint8) * 255

class MockPoseModel(IPoseModel):
    def estimate_pose(self, img: np.ndarray) -> Dict[str, Tuple[float, float]]:
        return {"shoulder_L": (100, 150), "shoulder_R": (200, 150)}

class MockSMPLReconstructor(ISMPlReconstructor):
    def reconstruct(self, img: np.ndarray, mask: np.ndarray, keypoints: dict) -> trimesh.Trimesh:
        mesh = trimesh.creation.capsule(height=1.7, radius=0.15)
        mesh.apply_translation([0, 0.85, 0]) 
        return mesh

class BodyMeasurementPipeline:
    def __init__(self, seg_model: ISegmentationModel, pose_model: IPoseModel, smpl_model: ISMPlReconstructor):
        self.seg_model = seg_model
        self.pose_model = pose_model
        self.smpl_model = smpl_model

    def _decode_image(self, image_b64: str) -> np.ndarray:
        try:
            img_data = base64.b64decode(image_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image.")
            return img
        except Exception as e:
            raise ValueError(f"Invalid image payload: {e}")

    def predict(self, image_b64: str, height_cm: float, gender: str) -> dict:
        img = self._decode_image(image_b64)
        
        mask = self.seg_model.segment(img)
        keypoints = self.pose_model.estimate_pose(img)
        base_mesh = self.smpl_model.reconstruct(img, mask, keypoints)
        
        scaled_mesh = scale_mesh(base_mesh, height_cm)
        measurements = extract_measurements(scaled_mesh)
        
        return measurements