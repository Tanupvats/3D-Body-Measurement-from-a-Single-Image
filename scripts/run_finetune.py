#!/usr/bin/env python
"""Fine-tune the body-measurement system on user-provided ground truth.

USAGE
-----

    python scripts/run_finetune.py \\
        --manifest data/my_subjects/manifest.csv \\
        --output-dir checkpoints/v1/ \\
        --config configs/default.yaml

The manifest format is documented in:
    bodymeasure/finetune/dataset_schema.py

WHAT IT DOES
------------
1. Loads + validates the manifest (subject_id, height_cm, gender,
   front_image, plus any measurement_cm columns you have).
2. For each subject, runs the full inference pipeline (regressor +
   optimizer + measurement extraction) and caches the result. This is
   the slow step (~5-30s per subject).
3. For each subject, computes a numerical Jacobian of measurements wrt
   betas (one cached per subject; ~10x extra forward passes).
4. Trains a small MLP (the "shape corrector") that maps features ->
   beta residual, supervised by ground-truth measurements.
5. Writes:
       <output-dir>/corrector.pt     (load via ShapeCorrector)
       <output-dir>/report.json      (per-measurement before/after MAE)
       <output-dir>/cache/           (precomputed pipeline outputs)

The pipeline picks up `corrector.pt` automatically at inference time
when you point the API/CLI at this output directory via:

    --corrector checkpoints/v1/corrector.pt

WHEN YOU HAVE NO GROUND TRUTH
-----------------------------
Don't run this. The base pipeline still works without it — measurements
will use the pretrained-model output as-is, with prior uncertainties.
The corrector ONLY helps once you have paired (image, tape-measure) data.

REPRODUCIBILITY
---------------
Cache key is sha256(image_bytes) + regressor_name + body_model + use_optimizer,
so changing any of these invalidates the cache automatically. Use --force-recache
to recompute even when keys match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bodymeasure.core.config import Config, default_config  # noqa: E402
from bodymeasure.core.pipeline import Pipeline  # noqa: E402
from bodymeasure.core.types import Gender, SMPLXParams  # noqa: E402
from bodymeasure.finetune.dataset import (  # noqa: E402
    FineTuneDataset,
    cache_key,
    load_manifest,
)
from bodymeasure.finetune.trainer import compute_jacobian, train_corrector  # noqa: E402
from bodymeasure.measurement.extractor import extract_measurements  # noqa: E402
from bodymeasure.utils.logging import configure, get_logger  # noqa: E402

import numpy as np
import pickle


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True,
                   help="Path to manifest.csv or manifest.json")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write corrector.pt + report.json + cache/")
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config overrides (optional)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--force-recache", action="store_true",
                   help="Recompute precomputed pipeline outputs even if cached")
    p.add_argument("--skip-jacobians", action="store_true",
                   help="If you've already computed and cached jacobians, skip recompute")
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    p.add_argument("--lr", type=float, default=None, help="Override config lr")
    p.add_argument("--batch-size", type=int, default=None, help="Override config batch_size")
    p.add_argument("--val-split", type=float, default=None, help="Override val_split")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    import logging as _logging
    configure(_logging.DEBUG if args.verbose else _logging.INFO)
    logger = get_logger("scripts.run_finetune")

    # Load config
    if args.config:
        cfg = Config.from_yaml(args.config)
        logger.info(f"loaded config from {args.config}")
    else:
        cfg = default_config()
        logger.info("using default config")

    # Override hyperparameters if provided
    if args.epochs is not None:
        cfg.finetune.epochs = args.epochs
    if args.lr is not None:
        cfg.finetune.lr = args.lr
    if args.batch_size is not None:
        cfg.finetune.batch_size = args.batch_size
    if args.val_split is not None:
        cfg.finetune.val_split = args.val_split

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"

    # ---- 1. Load manifest ----
    logger.info(f"loading manifest: {args.manifest}")
    samples = load_manifest(args.manifest)
    if not samples:
        logger.error("no usable subjects in manifest. Aborting.")
        sys.exit(1)
    logger.info(f"manifest contained {len(samples)} subjects")
    # Sanity check: how many subjects have at least one measurement?
    n_with_meas = sum(1 for s in samples if s.measurements)
    if n_with_meas == 0:
        logger.error("no subjects have ANY measurement columns — nothing to train against")
        sys.exit(1)
    logger.info(f"  {n_with_meas}/{len(samples)} subjects have at least one measurement")

    # ---- 2. Build pipeline + dataset ----
    def pipeline_factory():
        return Pipeline(cfg, device=args.device, corrector=None)

    dataset = FineTuneDataset(
        samples=samples,
        pipeline_factory=pipeline_factory,
        config=cfg,
        cache_dir=cache_dir,
    )
    logger.info("running pipeline on each subject (this is the slow step)...")
    dataset.precompute(force=args.force_recache)

    # ---- 3. Compute Jacobians per subject ----
    # We need d(measurement)/d(beta) at each subject's betas, so the linear
    # surrogate works during training. Cache them alongside the pipeline output.
    logger.info("computing per-subject measurement-vs-beta jacobians...")
    jacobian_path = lambda s: cache_dir / f"{s.subject_id}_jacobian.pkl"

    # For computing jacobians we need a body model loaded. We load gender-specific
    # models on demand.
    pipeline = pipeline_factory()
    pipeline._ensure_loaded()

    def measurement_fn_factory(gender: Gender):
        body_model, _ = pipeline._components_for_gender(gender)

        def fn(params: SMPLXParams) -> dict:
            fitted = pipeline._fitted_from_init(
                params, body_model, camera=None, gender=gender,
            )
            measurements = extract_measurements(fitted, cfg.measurement)
            return {m.name: m.value_cm for m in measurements
                    if not np.isnan(m.value_cm)}
        return fn

    valid_samples = []
    for s in samples:
        cache_path = cache_dir / f"{s.subject_id}_{cache_key(s.front_image, cfg)}.pkl"
        if not cache_path.exists():
            logger.warning(f"  skipping {s.subject_id}: no precomputed cache")
            continue
        with open(cache_path, "rb") as fh:
            d = pickle.load(fh)
        jac_path = jacobian_path(s)
        if jac_path.exists() and args.skip_jacobians:
            with open(jac_path, "rb") as fh:
                jac_data = pickle.load(fh)
            d["_jacobian"] = jac_data["J"]
            d["_baseline_for_jac"] = jac_data["baseline"]
        else:
            try:
                gender = Gender(s.gender)
            except ValueError:
                logger.warning(f"  unknown gender '{s.gender}' for {s.subject_id}, using neutral")
                gender = Gender.NEUTRAL
            mfn = measurement_fn_factory(gender)
            betas = d["init_betas"]
            try:
                J, baseline = compute_jacobian(
                    body_model=None,  # not used inside; mfn closes over it
                    betas=betas,
                    measurement_fn=mfn,
                    eps=0.01,
                )
                with open(jac_path, "wb") as fh:
                    pickle.dump({"J": J, "baseline": baseline}, fh)
                d["_jacobian"] = J
                d["_baseline_for_jac"] = baseline
            except Exception as e:
                logger.error(f"  jacobian failed for {s.subject_id}: {e}")
                continue
        valid_samples.append((s, d))

    if len(valid_samples) < 4:
        logger.error(f"only {len(valid_samples)} usable samples (need >= 4). Aborting.")
        sys.exit(1)

    # ---- 4. Build training arrays ----
    features = np.stack([d["features"] for _, d in valid_samples], axis=0)
    init_betas = np.stack([d["init_betas"] for _, d in valid_samples], axis=0)
    jacobians = [d["_jacobian"] for _, d in valid_samples]
    baselines = [d["_baseline_for_jac"] for _, d in valid_samples]
    ground_truth = [s.measurements for s, _ in valid_samples]

    # ---- 5. Train ----
    logger.info(f"training corrector on {len(valid_samples)} subjects...")
    report = train_corrector(
        features=features,
        init_betas=init_betas,
        jacobians=jacobians,
        baselines=baselines,
        ground_truth=ground_truth,
        cfg=cfg.finetune,
        output_dir=args.output_dir,
    )

    logger.info("=" * 60)
    logger.info("DONE.")
    logger.info(f"  Corrector checkpoint: {args.output_dir / 'corrector.pt'}")
    logger.info(f"  Report:               {args.output_dir / 'report.json'}")
    logger.info(f"  Best val L1:          {report['best_val_l1_cm']:.3f} cm")
    logger.info("")
    logger.info("To use the corrector at inference time:")
    logger.info(f"  python scripts/run_pipeline.py --image FRONT.jpg --height-cm 175 \\")
    logger.info(f"      --gender male --corrector {args.output_dir / 'corrector.pt'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
