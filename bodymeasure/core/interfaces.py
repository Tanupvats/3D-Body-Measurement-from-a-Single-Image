

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .types import (
    CameraIntrinsics,
    FittedBody,
    Gender,
    HumanParsing,
    ImageInput,
    Keypoints2D,
    PersonDetection,
    SMPLXParams,
)


class IPersonDetector(ABC):
    """Detect people in an image. Returns the highest-confidence person."""

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> Optional[PersonDetection]:
        ...


class IHumanParser(ABC):
    """Pixel-accurate human segmentation, ideally with body-part labels."""

    @abstractmethod
    def parse(self, image_bgr: np.ndarray, bbox_xyxy: Optional[np.ndarray] = None) -> HumanParsing:
        ...


class IPoseEstimator(ABC):
    """2D keypoint detection."""

    @abstractmethod
    def estimate(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: Optional[np.ndarray] = None,
    ) -> Keypoints2D:
        ...


class IBodyRegressor(ABC):
    """Pretrained image -> SMPL-X parameter regressor.

    Provides the initialization for the optimization stage. Quality of
    this init dominates speed-of-convergence and basin-of-attraction; a
    bad init is the most common failure mode in HMR pipelines.
    """

    @abstractmethod
    def regress(
        self,
        image_bgr: np.ndarray,
        bbox_xyxy: Optional[np.ndarray],
        camera: CameraIntrinsics,
        gender: Gender,
    ) -> SMPLXParams:
        ...


class IBodyOptimizer(ABC):
    """SMPLify-X-style fitting: refine SMPL-X params against 2D evidence."""

    @abstractmethod
    def fit(
        self,
        views: list[ImageInput],
        parsings: list[HumanParsing],
        keypoints: list[Keypoints2D],
        cameras: list[CameraIntrinsics],
        init_params: SMPLXParams,
        height_cm: Optional[float] = None,
        gender: Gender = Gender.NEUTRAL,
    ) -> FittedBody:
        ...


class ICameraEstimator(ABC):
    """Recover camera intrinsics from EXIF / image / learned prior."""

    @abstractmethod
    def estimate(self, image_bgr: np.ndarray, exif: dict) -> CameraIntrinsics:
        ...
