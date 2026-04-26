"""bodymeasure — single/multi-image human body measurement.

Architecture overview:
    image(s) + height/metadata
        -> camera estimation
        -> person detection + tight crop
        -> human parsing (segmentation w/ part labels)
        -> 2D keypoint detection
        -> SMPL-X regression (pretrained init)
        -> SMPLify-X-style optimization refinement
        -> anatomical landmark + measurement extraction
        -> uncertainty + QC report
        -> measurement output

The package is deliberately structured so that every component is
swappable. Pretrained ML models are loaded via adapter classes that
all conform to the abstract interfaces in `bodymeasure.core.interfaces`.
"""

__version__ = "0.1.0"
