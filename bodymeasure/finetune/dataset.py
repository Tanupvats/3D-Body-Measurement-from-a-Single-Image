

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.config import Config
from ..core.types import Gender
from ..utils.logging import get_logger
from .dataset_schema import (
    GroundTruthSample,
    REQUIRED_COLUMNS,
    SUPPORTED_MEASUREMENTS,
    normalize_measurement_name,
)
from .features import features_from_pipeline_state

logger = get_logger(__name__)


def load_manifest(manifest_path: Path) -> list[GroundTruthSample]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    data_root = manifest_path.parent
    rows: list[dict] = []
    if manifest_path.suffix.lower() == ".json":
        with open(manifest_path) as fh:
            rows = json.load(fh)
        if isinstance(rows, dict) and "subjects" in rows:
            rows = rows["subjects"]
    else:
        with open(manifest_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

    if not rows:
        raise ValueError(f"manifest {manifest_path} contains no rows")

    columns = set(rows[0].keys())
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")

    samples: list[GroundTruthSample] = []
    for r in rows:
        try:
            measurements: dict = {}
            for col, val in r.items():
                canon = normalize_measurement_name(col)
                if canon is None:
                    continue
                if val is None or val == "":
                    continue
                try:
                    measurements[canon] = float(val)
                except (TypeError, ValueError):
                    pass

            front = data_root / str(r["front_image"])
            side = data_root / str(r["side_image"]) if r.get("side_image") else None

            samples.append(GroundTruthSample(
                subject_id=str(r["subject_id"]),
                height_cm=float(r["height_cm"]),
                gender=str(r["gender"]).lower(),
                front_image=front,
                side_image=side,
                measurements=measurements,
                weight_kg=float(r["weight_kg"]) if r.get("weight_kg") else None,
                age_years=int(r["age_years"]) if r.get("age_years") else None,
            ))
        except Exception as e:
            logger.warning(f"skipping malformed row {r.get('subject_id', '?')}: {e}")
    logger.info(f"loaded {len(samples)} subjects from {manifest_path}")
    return samples


def cache_key(image_path: Path, config: Config) -> str:
    """Hash image bytes + relevant config fields to invalidate cache on changes."""
    h = hashlib.sha256()
    h.update(image_path.read_bytes())
    h.update(config.regressor_name.encode())
    h.update(config.body_model.encode())
    h.update(str(config.use_optimizer).encode())
    return h.hexdigest()[:16]


class FineTuneDataset:
    """Computes (feature_vector, ground_truth_measurements) per subject.

    The expensive per-subject pipeline (regression + optimization +
    measurement extraction) is cached at `cache_dir/{key}.pkl`.

    Returns numpy arrays so the trainer is framework-agnostic.
    """

    def __init__(
        self,
        samples: list[GroundTruthSample],
        pipeline_factory,        # callable that returns a configured Pipeline
        config: Config,
        cache_dir: Path,
    ):
        self.samples = samples
        self.pipeline_factory = pipeline_factory
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = self.pipeline_factory()
        return self._pipeline

    def precompute(self, force: bool = False) -> None:
        """Run the pipeline on every subject and cache features + base predictions.

        This is the slow step. ~5-30s per subject depending on optimizer iters
        and whether GPU is available. Run once, train many times.
        """
        from ..core.pipeline import PipelineDebugInfo  # forward reference is fine
        for s in self.samples:
            key = cache_key(s.front_image, self.config)
            cache_path = self.cache_dir / f"{s.subject_id}_{key}.pkl"
            if cache_path.exists() and not force:
                continue
            logger.info(f"precomputing subject {s.subject_id}...")
            try:
                debug = self._get_pipeline().run_debug(
                    front_image_path=s.front_image,
                    side_image_path=s.side_image,
                    height_cm=s.height_cm,
                    gender=Gender(s.gender),
                )
                features = features_from_pipeline_state(
                    init_params=debug.init_params,
                    height_cm=s.height_cm,
                    detection=debug.detection,
                    keypoints=debug.keypoints,
                    gender=Gender(s.gender),
                )
                # Predicted measurements from this subject (BEFORE corrector).
                pred_dict = {m.name: m.value_cm for m in debug.measurements}
                with open(cache_path, "wb") as fh:
                    pickle.dump({
                        "features": features,
                        "init_betas": debug.init_params.betas.flatten(),
                        "predicted_measurements": pred_dict,
                        "ground_truth": s.measurements,
                    }, fh)
            except Exception as e:
                logger.error(f"  subject {s.subject_id} failed: {e}")

    def load_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Build training arrays from the cache.

        Returns:
            features:  (N, FEATURE_DIM)
            gt_meas:   (N, M) ground-truth measurements in CM (NaN where missing)
            pred_meas: (N, M) baseline predicted measurements in CM (NaN where missing)
            meas_names: list of measurement names for column index
        """
        # Discover which measurements have any data
        all_meas: set[str] = set()
        loaded = []
        for s in self.samples:
            key = cache_key(s.front_image, self.config)
            cache_path = self.cache_dir / f"{s.subject_id}_{key}.pkl"
            if not cache_path.exists():
                logger.warning(f"missing cache for {s.subject_id}, skipping")
                continue
            with open(cache_path, "rb") as fh:
                d = pickle.load(fh)
            loaded.append(d)
            all_meas.update(d["ground_truth"].keys())
        meas_names = sorted(all_meas)
        N = len(loaded)
        M = len(meas_names)
        if N == 0:
            raise RuntimeError("no cached samples — run precompute() first")
        features = np.stack([d["features"] for d in loaded], axis=0)
        gt = np.full((N, M), np.nan, dtype=np.float32)
        pr = np.full((N, M), np.nan, dtype=np.float32)
        for i, d in enumerate(loaded):
            for j, name in enumerate(meas_names):
                if name in d["ground_truth"]:
                    gt[i, j] = d["ground_truth"][name]
                if name in d["predicted_measurements"]:
                    pr[i, j] = d["predicted_measurements"][name]
        logger.info(f"built arrays: features {features.shape}, gt {gt.shape}, pred {pr.shape}")
        logger.info(f"measurement columns ({M}): {meas_names}")
        return features, gt, pr, meas_names
