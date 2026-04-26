

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import trimesh


def planar_slice(mesh: "trimesh.Trimesh", plane_origin: np.ndarray, plane_normal: np.ndarray):
    """Slice a mesh with a plane. Returns a list of (N_i, 2) polylines in plane coords.

    Returns empty list if no intersection. Multiple polylines occur when
    the slice cuts disconnected components (e.g. a horizontal slice
    through both arms and the torso at chest height — three loops).
    """
    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        return []
    planar, _ = section.to_planar()
    if planar is None:
        return []
    polylines = []
    for ent in planar.entities:
        pts = planar.vertices[ent.points]
        polylines.append(pts.astype(np.float64))
    return polylines


def convex_hull_perimeter(points: np.ndarray) -> float:
    """Convex-hull perimeter of a 2D point set in metres.

    A tape measure does not follow body concavities (e.g. the dip behind
    the navel), it stretches over them. Convex-hull perimeter matches
    this. Raw mesh-slice perimeter undercounts by ~3-8% at the waist.
    """
    if points.shape[0] < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
    except ImportError as e:
        raise RuntimeError("scipy required for convex_hull_perimeter") from e
    try:
        hull = ConvexHull(points)
    except Exception:
        return 0.0
    hull_pts = points[hull.vertices]
    closed = np.vstack([hull_pts, hull_pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    return float(seg.sum())


def polyline_length(points: np.ndarray, closed: bool = True) -> float:
    if points.shape[0] < 2:
        return 0.0
    if closed:
        pts = np.vstack([points, points[:1]])
    else:
        pts = points
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(seg.sum())


def largest_polyline(polylines: list[np.ndarray]) -> np.ndarray:
    """From a list of slice polylines, return the longest (typically torso)."""
    if not polylines:
        return np.zeros((0, 2))
    return max(polylines, key=lambda p: polyline_length(p, closed=True))


def vertex_distance(verts: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(verts[a] - verts[b]))


def geodesic_distance(mesh: "trimesh.Trimesh", a: int, b: int) -> float:
    """Surface shortest-path distance via Dijkstra on the vertex graph.

    Approximate but cheap. For higher accuracy use Heat method (libigl)
    or Crane et al.'s heat-geodesics, both of which are heavier deps.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise RuntimeError("networkx required for geodesic_distance") from e
    g = mesh.vertex_adjacency_graph
    # weight edges by Euclidean distance
    for u, v, data in g.edges(data=True):
        data["weight"] = float(np.linalg.norm(mesh.vertices[u] - mesh.vertices[v]))
    try:
        return float(nx.shortest_path_length(g, a, b, weight="weight"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("nan")


def slice_at_vertex_height(
    mesh: "trimesh.Trimesh",
    vertex_id: int,
    axis: str = "y",
) -> list[np.ndarray]:
    """Convenience: slice horizontally at the height of a given vertex."""
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    plane_origin = np.zeros(3)
    plane_origin[axis_idx] = mesh.vertices[vertex_id, axis_idx]
    plane_normal = np.zeros(3)
    plane_normal[axis_idx] = 1.0
    return planar_slice(mesh, plane_origin, plane_normal)


def slice_perpendicular_to(
    mesh: "trimesh.Trimesh",
    point: np.ndarray,
    direction: np.ndarray,
) -> list[np.ndarray]:
    """Slice with a plane through `point` whose normal is `direction`."""
    n = direction / (np.linalg.norm(direction) + 1e-9)
    return planar_slice(mesh, point, n)
