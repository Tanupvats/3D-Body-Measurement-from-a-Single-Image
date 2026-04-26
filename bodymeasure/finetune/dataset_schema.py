

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Canonical names used INSIDE the system. CSV column suffixes `_cm` are
# stripped when matching. Both `arm_length_cm` and `arm_length_mean_cm`
# resolve to the same internal key `arm_length_mean`.
SUPPORTED_MEASUREMENTS = [
    "chest_circumference",
    "waist_circumference",
    "hip_circumference",
    "neck_circumference",
    "shoulder_breadth",
    "arm_length_left",
    "arm_length_right",
    "arm_length_mean",
    "inseam_left",
    "inseam_right",
    "inseam_mean",
    "outseam_left",
    "outseam_right",
    "thigh_left_circumference",
    "thigh_right_circumference",
    "thigh_circumference_mean",
    "calf_left_circumference",
    "calf_right_circumference",
    "calf_circumference_mean",
    "bicep_left_circumference",
    "bicep_right_circumference",
    "bicep_circumference_mean",
    "stature",
]


REQUIRED_COLUMNS = {"subject_id", "height_cm", "gender", "front_image"}


@dataclass
class GroundTruthSample:
    subject_id: str
    height_cm: float
    gender: str
    front_image: Path
    side_image: Optional[Path]
    measurements: dict   # {canonical_name: value_cm}
    weight_kg: Optional[float] = None
    age_years: Optional[int] = None


def normalize_measurement_name(col: str) -> Optional[str]:
    """Map CSV column names to canonical internal names. Returns None if not measurement."""
    c = col.lower().strip()
    if not c.endswith("_cm"):
        return None
    base = c[:-3]
    # Common aliases
    aliases = {
        "chest": "chest_circumference",
        "waist": "waist_circumference",
        "hip":   "hip_circumference",
        "hips":  "hip_circumference",
        "neck":  "neck_circumference",
        "arm_length": "arm_length_mean",
        "inseam":     "inseam_mean",
        "thigh_circumference": "thigh_circumference_mean",
        "calf_circumference":  "calf_circumference_mean",
        "bicep_circumference": "bicep_circumference_mean",
        "shoulders": "shoulder_breadth",
        "shoulder_width": "shoulder_breadth",
    }
    if base in aliases:
        base = aliases[base]
    if base in SUPPORTED_MEASUREMENTS:
        return base
    return None
