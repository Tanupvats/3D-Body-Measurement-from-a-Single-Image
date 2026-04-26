#!/usr/bin/env python
"""Run the body-measurement pipeline on a single image (or front+side pair).

USAGE
-----
    python scripts/run_pipeline.py \\
        --front images/me_front.jpg \\
        --side  images/me_side.jpg  \\
        --height-cm 178.5 \\
        --gender male \\
        --output result.json

    # With a fine-tuned corrector
    python scripts/run_pipeline.py \\
        --front images/me_front.jpg \\
        --height-cm 178.5 --gender male \\
        --corrector checkpoints/v1/corrector.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bodymeasure.core.config import Config, default_config  # noqa: E402
from bodymeasure.core.pipeline import Pipeline  # noqa: E402
from bodymeasure.core.types import Gender  # noqa: E402
from bodymeasure.finetune.corrector import ShapeCorrector  # noqa: E402
from bodymeasure.measurement.extractor import measurements_to_dict  # noqa: E402
from bodymeasure.utils.logging import configure, get_logger  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--front", type=Path, required=True, help="Front-view image")
    p.add_argument("--side", type=Path, default=None, help="Side-view image (optional)")
    p.add_argument("--height-cm", type=float, required=True)
    p.add_argument("--gender", choices=["male", "female", "neutral"], default="neutral")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--corrector", type=Path, default=None,
                   help="Path to fine-tuned corrector.pt")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output", type=Path, default=None,
                   help="Write JSON output to this path (else stdout)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    import logging as _logging
    configure(_logging.DEBUG if args.verbose else _logging.INFO)
    logger = get_logger("scripts.run_pipeline")

    cfg = Config.from_yaml(args.config) if args.config else default_config()
    corrector = ShapeCorrector(args.corrector, device=args.device) if args.corrector else None
    pipeline = Pipeline(cfg, device=args.device, corrector=corrector)

    logger.info("running pipeline...")
    result = pipeline.predict(
        front_image_path=args.front,
        side_image_path=args.side,
        height_cm=args.height_cm,
        gender=Gender(args.gender),
    )

    out = {
        "measurements": result.measurements_dict(),
        "qc": {
            "passed": result.qc.passed,
            "warnings": result.qc.warnings,
            "failures": result.qc.failures,
            "metrics": result.qc.metrics,
        },
        "metadata": result.metadata,
    }
    text = json.dumps(out, indent=2)
    if args.output:
        args.output.write_text(text)
        logger.info(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
