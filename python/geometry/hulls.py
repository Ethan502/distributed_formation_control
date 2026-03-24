"""Convex hull utilities shared across multiple phases.

These helpers wrap scipy.spatial.ConvexHull in a form that is convenient for
the consensus algorithm and for plotting.

Used by:
    - consensus/convex_hull.py  — the distributed hull consensus (Phase 1).
    - plotting/draw_hulls.py    — drawing local hull estimates.
    - formation/optimizer.py    — checking that formation fits inside the hull.
"""

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def compute_convex_hull(points: np.ndarray) -> np.ndarray:
    """Compute the convex hull of a set of 2D points and return vertex indices.

    Args:
        points: Array of shape (N, 2) with 2D point coordinates.
                N must be at least 1.

    Returns:
        hull_indices: 1-D integer array of indices into `points` that lie on
                      the convex hull, ordered counter-clockwise.

    Notes:
        - Uses scipy.spatial.ConvexHull for N >= 3 non-collinear points.
        - For N == 1: returns np.array([0]).
        - For N == 2: returns np.array([0, 1]).
        - For N >= 3 but all collinear: return the two extreme point indices.
        - scipy.spatial.ConvexHull raises QhullError for degenerate inputs;
          catch it and fall back to the collinear case.
    """

    N = points.shape[0]

    if N == 1:
        return np.array([0])
    if N ==2:
        return np.array([0,1])
    try:
        hull = ConvexHull(points)
        return hull.vertices
    except QhullError:
        diffs = points[:,np.newaxis,:] - points[np.newaxis,:,:]
        dists = np.linalg.norm(diffs,axis=2)
        i,j = np.unravel_index(np.argmax(dists),dists.shape)
        return np.array([i,j])

def extract_hull_vertices(points: np.ndarray) -> np.ndarray:
    """Return the convex hull vertex coordinates for a set of 2D points.

    Args:
        points: Array of shape (N, 2).

    Returns:
        hull_vertices: Array of shape (K, 2) where K <= N is the number of
                       hull vertices, ordered counter-clockwise.

    Notes:
        - This is a convenience wrapper: calls compute_convex_hull and indexes
          into points.
        - The returned array is a new array (a copy), not a view.
    """
    indices = compute_convex_hull(points)
    return points[indices].copy()
