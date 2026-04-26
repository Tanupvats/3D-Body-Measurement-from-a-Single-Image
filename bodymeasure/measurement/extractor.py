

from __future__ import annotations

from typing import Optional

import numpy as np
import trimesh

from ..core.config import MeasurementConfig
from ..core.types import FittedBody, Measurement
from ..utils.logging import get_logger
from . import landmarks as L
from .geometry import (
    convex_hull_perimeter,
    geodesic_distance,
    largest_polyline,
    planar_slice,
    polyline_length,
    slice_perpendicular_to,
    vertex_distance,
)

logger = get_logger(__name__)


# Population-statistic-derived prior uncertainties (cm).
# These are SHAPE-RECONSTRUCTION uncertainties when not running MC sampling
# — they reflect the typical disagreement between SMPL-X mean-body fits
# and ground-truth body scans. Replace with real numbers post-fine-tune.
PRIOR_UNCERTAINTY_CM = {
    "stature":           0.5,
    "chest":             3.0,
    "waist":             3.5,
    "hip":               3.0,
    "neck":              1.5,
    "shoulder_breadth":  1.5,
    "arm_length":        2.0,
    "inseam":            2.0,
    "outseam":           2.0,
    "thigh":             2.5,
    "calf":              1.5,
    "bicep":             1.5,
    "default":           3.0,
}


def _prior_uncertainty(name: str) -> float:
    for key, val in PRIOR_UNCERTAINTY_CM.items():
        if key in name:
            return val
    return PRIOR_UNCERTAINTY_CM["default"]


def _resolve_endpoint(spec, vertices: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Endpoint may be a vertex index (int) or a `joint:NAME` string."""
    if isinstance(spec, str) and spec.startswith("joint:"):
        joint_name = spec.split(":", 1)[1]
        return joints[L.SMPLX_JOINTS[joint_name]]
    return vertices[int(spec)]


def _girth_at_vertex_horizontal(
    mesh: trimesh.Trimesh, vertex_id: int, slack_cm: float = 0.0,
) -> float:
    """Convex-hull horizontal girth at the height of `vertex_id`. Returns cm."""
    y_val = mesh.vertices[vertex_id, 1]
    polylines = planar_slice(
        mesh, plane_origin=np.array([0, y_val, 0]), plane_normal=np.array([0, 1.0, 0]),
    )
    if not polylines:
        return float("nan")
    poly = largest_polyline(polylines)
    perim_m = convex_hull_perimeter(poly)
    return perim_m * 100.0 + slack_cm


def _girth_perpendicular_to_axis(
    mesh: trimesh.Trimesh,
    vertex_id: int,
    axis_point_a: np.ndarray,
    axis_point_b: np.ndarray,
    slack_cm: float = 0.0,
) -> float:
    """Slice perpendicular to (axis_a -> axis_b), through `vertex_id`."""
    direction = axis_point_b - axis_point_a
    point = mesh.vertices[vertex_id]
    polylines = slice_perpendicular_to(mesh, point, direction)
    if not polylines:
        return float("nan")
    poly = largest_polyline(polylines)
    perim_m = convex_hull_perimeter(poly)
    return perim_m * 100.0 + slack_cm


def extract_measurements(
    fitted: FittedBody,
    cfg: Optional[MeasurementConfig] = None,
) -> list[Measurement]:
    """Compute the full panel of measurements from a fitted body."""
    cfg = cfg or MeasurementConfig()
    verts = fitted.vertices_m
    faces = fitted.faces
    joints = fitted.joints_m

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    out: list[Measurement] = []

    # ---- Girths ----
    # Body-axis-aligned (chest/waist/hip/neck): horizontal slice
    for name in ("chest", "waist", "hip", "neck"):
        landmark_a, _, _ = L.GIRTH_LANDMARKS[name]
        try:
            cm = _girth_at_vertex_horizontal(mesh, landmark_a, slack_cm=cfg.convex_hull_slack_cm)
        except Exception as e:
            logger.warning(f"girth {name} failed: {e}")
            cm = float("nan")
        out.append(Measurement(
            name=f"{name}_circumference",
            value_cm=cm,
            uncertainty_cm=_prior_uncertainty(name),
            method="convex_hull_horizontal_slice",
            notes=f"slice at vertex {landmark_a}",
        ))

    # Limb girths: slice perpendicular to limb axis
    limb_axes = {
        "thigh_left":   ("left_hip",      "left_knee"),
        "thigh_right":  ("right_hip",     "right_knee"),
        "calf_left":    ("left_knee",     "left_ankle"),
        "calf_right":   ("right_knee",    "right_ankle"),
        "bicep_left":   ("left_shoulder", "left_elbow"),
        "bicep_right":  ("right_shoulder", "right_elbow"),
    }
    for name, (j_a, j_b) in limb_axes.items():
        landmark_a, _, _ = L.GIRTH_LANDMARKS[name]
        try:
            ja = joints[L.SMPLX_JOINTS[j_a]]
            jb = joints[L.SMPLX_JOINTS[j_b]]
            cm = _girth_perpendicular_to_axis(mesh, landmark_a, ja, jb,
                                              slack_cm=cfg.convex_hull_slack_cm)
        except Exception as e:
            logger.warning(f"girth {name} failed: {e}")
            cm = float("nan")
        out.append(Measurement(
            name=f"{name}_circumference",
            value_cm=cm,
            uncertainty_cm=_prior_uncertainty(name.split("_")[0]),
            method="convex_hull_perpendicular_slice",
            notes=f"slice at vertex {landmark_a}, axis {j_a}->{j_b}",
        ))

    # ---- Linear measurements ----
    for name, (start, end, use_geodesic) in L.LINEAR_LANDMARKS.items():
        try:
            p0 = _resolve_endpoint(start, verts, joints)
            p1 = _resolve_endpoint(end, verts, joints)
            if use_geodesic and cfg.enable_geodesic and isinstance(start, int) and isinstance(end, int):
                d_m = geodesic_distance(mesh, start, end)
            else:
                d_m = float(np.linalg.norm(p0 - p1))
            cm = d_m * 100.0
        except Exception as e:
            logger.warning(f"linear {name} failed: {e}")
            cm = float("nan")
        out.append(Measurement(
            name=name,
            value_cm=cm,
            uncertainty_cm=_prior_uncertainty(name),
            method="geodesic" if use_geodesic else "vertex_distance",
        ))

    # ---- Convenience: take the larger of left/right for arm/inseam where users
    #      typically expect a single value. We keep both sides but ALSO report
    #      a "max" variant for compatibility.
    for base in ("arm_length", "inseam", "thigh_circumference",
                 "calf_circumference", "bicep_circumference"):
        l_name = f"{base}_left" if base != "arm_length" else "arm_length_left"
        r_name = f"{base}_right" if base != "arm_length" else "arm_length_right"
        l_m = _find(out, l_name)
        r_m = _find(out, r_name)
        if l_m and r_m and not (np.isnan(l_m.value_cm) or np.isnan(r_m.value_cm)):
            mean_v = (l_m.value_cm + r_m.value_cm) / 2.0
            mean_u = (l_m.uncertainty_cm + r_m.uncertainty_cm) / 2.0
            out.append(Measurement(
                name=f"{base}_mean",
                value_cm=mean_v,
                uncertainty_cm=mean_u,
                method="mean_of_sides",
            ))

    return out


def _find(measurements: list[Measurement], name: str) -> Optional[Measurement]:
    for m in measurements:
        if m.name == name:
            return m
    return None


def measurements_to_dict(measurements: list[Measurement]) -> dict:
    return {
        m.name: {
            "value_cm": round(m.value_cm, 2) if not np.isnan(m.value_cm) else None,
            "uncertainty_cm": round(m.uncertainty_cm, 2),
            "method": m.method,
            "notes": m.notes,
        }
        for m in measurements
    }
