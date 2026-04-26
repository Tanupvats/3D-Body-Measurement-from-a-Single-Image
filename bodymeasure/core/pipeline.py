

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.config import Config
from ..core.factory import (
    make_body_model,
    make_camera_estimator,
    make_detector,
    make_optimizer,
    make_parser,
    make_pose,
    make_regressor,
)
from ..core.types import (
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
from ..io.image_io import load_image
from ..measurement.extractor import extract_measurements, measurements_to_dict
from ..qc.gates import post_fit_qc, pre_pipeline_qc
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineDebugInfo:
    """Intermediate outputs needed by the fine-tune cache builder."""
    init_params: SMPLXParams
    fitted: FittedBody
    detection: PersonDetection
    keypoints: Keypoints2D
    parsing: HumanParsing
    measurements: list[Measurement]


class Pipeline:
    def __init__(self, config: Config, device: str = "cpu", corrector=None):
        self.config = config
        self.device = device
        self.corrector = corrector  # optional ShapeCorrector
        # Lazy-loaded
        self._cam_est = None
        self._detector = None
        self._parser = None
        self._pose = None
        self._regressor = None
        self._body_model = None
        self._optimizer = None

    # --- lazy loaders ---
    def _components_for_gender(self, gender: Gender):
        # body model + optimizer depend on gender (different shape spaces)
        if self._body_model is None or self._body_model.gender != gender:
            self._body_model = make_body_model(self.config, gender=gender.value, device=self.device)
            self._optimizer = make_optimizer(self.config, self._body_model, device=self.device) \
                if self.config.use_optimizer else None
        return self._body_model, self._optimizer

    def _ensure_loaded(self):
        if self._cam_est is None:
            self._cam_est = make_camera_estimator(self.config)
            self._detector = make_detector(self.config)
            self._parser = make_parser(self.config)
            self._pose = make_pose(self.config)
            self._regressor = make_regressor(self.config)

    # --- main API ---
    def predict(
        self,
        front_image_path: Path | str,
        side_image_path: Optional[Path | str] = None,
        height_cm: float = 170.0,
        gender: Gender = Gender.NEUTRAL,
    ) -> PredictionResult:
        debug = self.run_debug(
            front_image_path=front_image_path,
            side_image_path=side_image_path,
            height_cm=height_cm,
            gender=gender,
        )

        # Apply corrector if present
        fitted = debug.fitted
        if self.corrector is not None:
            from ..finetune.features import features_from_pipeline_state
            feat = features_from_pipeline_state(
                init_params=debug.init_params,
                height_cm=height_cm,
                detection=debug.detection,
                keypoints=debug.keypoints,
                gender=gender,
            )
            corrected_params = self.corrector(debug.fitted.params, feat)
            # Re-run body model with corrected betas (T-pose canonical for measurement)
            body_model, _ = self._components_for_gender(gender)
            from ..reconstruction.body_model import SMPLXModel, SyntheticBodyModel
            if isinstance(body_model, SMPLXModel):
                import torch
                with torch.no_grad():
                    out = body_model.forward(SMPLXParams(
                        betas=corrected_params.betas,
                        body_pose=np.zeros((1, 63), dtype=np.float32),
                        global_orient=np.zeros((1, 3), dtype=np.float32),
                        transl=np.zeros((1, 3), dtype=np.float32),
                        gender=gender,
                    ))
                    v = out.vertices[0].cpu().numpy()
                    j = out.joints[0].cpu().numpy()
                    faces_arr = out.faces if isinstance(out.faces, np.ndarray) else out.faces.cpu().numpy()
                fitted = FittedBody(
                    params=corrected_params,
                    vertices_m=v,
                    faces=np.asarray(faces_arr, dtype=np.int32),
                    joints_m=j,
                    camera=fitted.camera,
                    fit_loss=fitted.fit_loss,
                )
            # Re-extract measurements
            debug = PipelineDebugInfo(
                init_params=debug.init_params,
                fitted=fitted,
                detection=debug.detection,
                keypoints=debug.keypoints,
                parsing=debug.parsing,
                measurements=extract_measurements(fitted, self.config.measurement),
            )

        # Post-fit QC
        post_qc = post_fit_qc(
            fitted=debug.fitted,
            keypoints_per_view=[debug.keypoints],
            parsings_per_view=[debug.parsing],
            height_target_cm=height_cm,
            cfg=self.config.qc,
        )

        return PredictionResult(
            measurements=debug.measurements,
            fitted_body=debug.fitted,
            qc=post_qc,
            metadata={
                "regressor": self.config.regressor_name,
                "body_model": self.config.body_model,
                "corrector_active": self.corrector is not None and getattr(self.corrector, "_loaded", False),
                "gender": gender.value,
                "height_cm_input": float(height_cm),
            },
        )

    def run_debug(
        self,
        front_image_path: Path | str,
        side_image_path: Optional[Path | str] = None,
        height_cm: float = 170.0,
        gender: Gender = Gender.NEUTRAL,
    ) -> PipelineDebugInfo:
        self._ensure_loaded()

        # Load views
        front_bgr, front_exif = load_image(front_image_path)
        front_cam = self._cam_est.estimate(front_bgr, front_exif)
        front_view = ImageInput(image_bgr=front_bgr, view=ViewType.FRONT,
                                camera=front_cam, exif=front_exif)
        views = [front_view]
        if side_image_path is not None:
            side_bgr, side_exif = load_image(side_image_path)
            side_cam = self._cam_est.estimate(side_bgr, side_exif)
            views.append(ImageInput(image_bgr=side_bgr, view=ViewType.SIDE,
                                     camera=side_cam, exif=side_exif))

        # Per-view: detect, parse, pose
        detections = []
        parsings = []
        kpts_list = []
        for v in views:
            det = self._detector.detect(v.image_bgr)
            if det is None:
                raise RuntimeError("no person detected in image")
            detections.append(det)

            # Re-use detector mask if parser is the same source, else run parser
            try:
                parsing = self._parser.parse(v.image_bgr, det.bbox_xyxy)
            except Exception as e:
                logger.warning(f"parser failed ({e}); using detector mask")
                from ..core.types import HumanParsing
                fallback = det.mask if det.mask is not None else np.zeros(v.image_bgr.shape[:2], np.uint8)
                parsing = HumanParsing(full_mask=fallback)
            parsings.append(parsing)

            kp = self._pose.estimate(v.image_bgr, det.bbox_xyxy)
            kpts_list.append(kp)

        # Pre-fit QC on the front image only
        pre_qc = pre_pipeline_qc(views[0], detections[0], kpts_list[0], self.config.qc)
        if not pre_qc.passed:
            raise RuntimeError(f"pre-pipeline QC failed: {pre_qc.failures}")
        if pre_qc.warnings:
            for w in pre_qc.warnings:
                logger.warning(f"QC warning: {w}")

        # Regress initial SMPL-X params from the FRONT view
        init_params = self._regressor.regress(
            views[0].image_bgr, detections[0].bbox_xyxy, views[0].camera, gender,
        )

        # Optimization refinement
        body_model, optimizer = self._components_for_gender(gender)
        cameras = [v.camera for v in views]
        if optimizer is not None:
            try:
                fitted = optimizer.fit(
                    views=views,
                    parsings=parsings,
                    keypoints=kpts_list,
                    cameras=cameras,
                    init_params=init_params,
                    height_cm=height_cm,
                    gender=gender,
                )
            except Exception as e:
                logger.warning(f"optimizer failed ({e}); falling back to init params")
                fitted = self._fitted_from_init(init_params, body_model, cameras[0], gender)
        else:
            fitted = self._fitted_from_init(init_params, body_model, cameras[0], gender)

        measurements = extract_measurements(fitted, self.config.measurement)

        return PipelineDebugInfo(
            init_params=init_params,
            fitted=fitted,
            detection=detections[0],
            keypoints=kpts_list[0],
            parsing=parsings[0],
            measurements=measurements,
        )

    def _fitted_from_init(self, init_params, body_model, camera, gender) -> FittedBody:
        """Build a FittedBody by running the body model in T-pose with init betas."""
        from ..reconstruction.body_model import SMPLXModel, SyntheticBodyModel
        if isinstance(body_model, SMPLXModel):
            import torch
            with torch.no_grad():
                out = body_model.forward(SMPLXParams(
                    betas=init_params.betas,
                    body_pose=np.zeros((1, 63), dtype=np.float32),
                    global_orient=np.zeros((1, 3), dtype=np.float32),
                    transl=np.zeros((1, 3), dtype=np.float32),
                    gender=gender,
                ))
                v = out.vertices[0].cpu().numpy()
                j = out.joints[0].cpu().numpy()
                faces_arr = out.faces if isinstance(out.faces, np.ndarray) else out.faces.cpu().numpy()
        elif isinstance(body_model, SyntheticBodyModel):
            v = body_model.forward_np(init_params.betas, np.zeros((1, 3), dtype=np.float32))[0]
            faces_arr = body_model.faces
            # Synthetic model has no joint regressor; approximate joints as
            # ring centers at known fractional heights.
            j = self._approx_joints_synthetic(v)
        else:
            raise RuntimeError(f"unsupported body model type {type(body_model)}")
        return FittedBody(
            params=init_params,
            vertices_m=v,
            faces=np.asarray(faces_arr, dtype=np.int32),
            joints_m=j,
            camera=camera,
            fit_loss=float("nan"),
        )

    def _approx_joints_synthetic(self, vertices: np.ndarray) -> np.ndarray:
        """Cheap pelvis/hip/knee/etc. estimates for the synthetic mesh.

        Used only when running with `body_model=synthetic` for tests/CI.
        Returns a (22, 3) array roughly matching the SMPL-X joint layout
        used by the measurement extractor.
        """
        y_min = vertices[:, 1].min()
        y_max = vertices[:, 1].max()
        h = y_max - y_min
        # Approximate joint heights as fractions of total height.
        # These are rough ratios; the synthetic mesh is symmetric so x is +- offset.
        joints = np.zeros((22, 3), dtype=np.float32)
        joints[0]  = [0, y_min + 0.55 * h, 0]   # pelvis
        joints[1]  = [-0.10, y_min + 0.55 * h, 0]  # left_hip
        joints[2]  = [ 0.10, y_min + 0.55 * h, 0]  # right_hip
        joints[3]  = [0, y_min + 0.65 * h, 0]      # spine1
        joints[4]  = [-0.10, y_min + 0.30 * h, 0]  # left_knee
        joints[5]  = [ 0.10, y_min + 0.30 * h, 0]  # right_knee
        joints[6]  = [0, y_min + 0.70 * h, 0]      # spine2
        joints[7]  = [-0.10, y_min + 0.05 * h, 0]  # left_ankle
        joints[8]  = [ 0.10, y_min + 0.05 * h, 0]  # right_ankle
        joints[9]  = [0, y_min + 0.75 * h, 0]      # spine3
        joints[10] = [-0.10, y_min + 0.0  * h, 0.05]  # left_foot
        joints[11] = [ 0.10, y_min + 0.0  * h, 0.05]  # right_foot
        joints[12] = [0, y_min + 0.85 * h, 0]      # neck
        joints[13] = [-0.05, y_min + 0.78 * h, 0]  # left_collar
        joints[14] = [ 0.05, y_min + 0.78 * h, 0]  # right_collar
        joints[15] = [0, y_min + 0.93 * h, 0]      # head
        joints[16] = [-0.18, y_min + 0.78 * h, 0]  # left_shoulder
        joints[17] = [ 0.18, y_min + 0.78 * h, 0]  # right_shoulder
        joints[18] = [-0.20, y_min + 0.60 * h, 0]  # left_elbow
        joints[19] = [ 0.20, y_min + 0.60 * h, 0]  # right_elbow
        joints[20] = [-0.22, y_min + 0.45 * h, 0]  # left_wrist
        joints[21] = [ 0.22, y_min + 0.45 * h, 0]  # right_wrist
        return joints
