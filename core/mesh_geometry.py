import trimesh
import numpy as np
from typing import Dict

def scale_mesh(mesh: trimesh.Trimesh, target_height_cm: float) -> trimesh.Trimesh:
    """Scales a 3D mesh uniformly to match a target physical height."""
    bounds = mesh.bounds
    mesh_height = bounds[1][1] - bounds[0][1] 
    
    scale_factor = target_height_cm / mesh_height
    matrix = trimesh.transformations.scale_matrix(scale_factor, [0, 0, 0])
    
    scaled_mesh = mesh.copy()
    scaled_mesh.apply_transform(matrix)
    return scaled_mesh

def get_circumference_at_y(mesh: trimesh.Trimesh, y_val: float) -> float:
    """Slices the mesh at a specific Y-axis height and computes the perimeter."""
    try:
        plane_origin = [0, y_val, 0]
        plane_normal = [0, 1, 0]

        slice_3d = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        if slice_3d is None:
            return 0.0 
            
        slice_2d, _ = slice_3d.to_planar()
        return float(slice_2d.length)
        
    except Exception as e:
        print(f"Geometry Error at Y={y_val}: {e}")
        return 0.0

def extract_measurements(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """Computes all standard physical measurements from a correctly scaled 3D mesh."""
    bounds = mesh.bounds
    min_y, max_y = bounds[0][1], bounds[1][1]
    total_height = max_y - min_y

    y_chest = min_y + (total_height * 0.75)
    y_waist = min_y + (total_height * 0.60)
    y_hip = min_y + (total_height * 0.50)
    y_thigh = min_y + (total_height * 0.40)

    measurements = {
        "chest_circumference": get_circumference_at_y(mesh, y_chest),
        "waist_circumference": get_circumference_at_y(mesh, y_waist),
        "hip_circumference": get_circumference_at_y(mesh, y_hip),
        "thigh_circumference": get_circumference_at_y(mesh, y_thigh),
        "arm_length": total_height * 0.35,
        "inseam_length": total_height * 0.45,
        "shoulder_width": (bounds[1][0] - bounds[0][0]) * 0.8
    }

    return {k: round(v, 1) for k, v in measurements.items()}