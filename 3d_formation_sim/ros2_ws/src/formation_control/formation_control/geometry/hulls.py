"""3D convex hull computation utilities.

Wraps scipy.spatial.ConvexHull with careful handling of degenerate cases
(too few points, collinear points, coplanar points).
"""

import numpy as np
from scipy.spatial import ConvexHull


def compute_convex_hull_3d(points: np.ndarray):
    """Compute the 3D convex hull of a set of points.

    Handles degenerate configurations gracefully:
      - 0 or 1 points: returns None (no hull possible)
      - 2 points: returns None (line segment, not a 3D hull)
      - 3 points: returns None (triangle, not a 3D hull)
      - Collinear or coplanar points: returns None after catching errors
      - Otherwise: returns scipy.spatial.ConvexHull

    Uses the 'QJ' option (joggle input) to handle near-degenerate inputs
    that would otherwise cause Qhull precision errors.

    Args:
        points: (N, 3) array of 3D point coordinates.

    Returns:
        scipy.spatial.ConvexHull object if a valid 3D hull exists, None otherwise.
    """
    if points is None or len(points) == 0:
        return None

    points = np.atleast_2d(points)
    if points.shape[0] < 4 or points.shape[1] != 3:
        return None

    # Remove duplicate points before attempting hull
    unique_pts = np.unique(np.round(points, decimals=10), axis=0)
    if unique_pts.shape[0] < 4:
        return None

    try:
        return ConvexHull(unique_pts, qhull_options='QJ')
    except Exception:
        # Qhull can fail for nearly degenerate configurations even with QJ
        return None


def extract_hull_vertices_3d(points: np.ndarray) -> np.ndarray:
    """Extract the vertices of the 3D convex hull of the given points.

    For degenerate cases where a full 3D hull cannot be formed (e.g., fewer
    than 4 unique points, collinear or coplanar configurations), returns the
    unique points themselves as a fallback.

    Args:
        points: (N, 3) array of 3D point coordinates.

    Returns:
        (K, 3) array of convex hull vertex positions. K <= N.
        For degenerate cases, returns the unique input points.
    """
    if points is None or len(points) == 0:
        return np.empty((0, 3))

    points = np.atleast_2d(points)
    if points.shape[1] != 3:
        raise ValueError(f"Expected 3D points with shape (N, 3), got {points.shape}.")

    # Get unique points as fallback
    unique_pts = np.unique(np.round(points, decimals=10), axis=0)

    hull = compute_convex_hull_3d(points)
    if hull is None:
        # Degenerate case: return unique points
        return unique_pts

    return hull.points[hull.vertices]
