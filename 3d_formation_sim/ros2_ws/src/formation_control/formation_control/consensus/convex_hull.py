"""Single-robot convex hull consensus update.

Implements the per-robot update step of the distributed convex hull
consensus algorithm (Algorithm 1 from the paper). The ROS2 node handles
communication rounds; this module provides the merge logic for one step.
"""

import numpy as np
from ..geometry.hulls import extract_hull_vertices_3d


def merge_hull_estimates(my_hull: np.ndarray, neighbor_hulls: list) -> np.ndarray:
    """Merge this robot's hull estimate with received neighbor hull estimates.

    This implements one round of Algorithm 1 from the paper for a single robot:
      C_i(k+1) = ConvexHull(C_i(k) U U_{j in N_i} C_j(k))

    Each robot maintains a local estimate of the team's convex hull. In each
    consensus round, it merges its own hull vertices with those received from
    neighbors, then recomputes the hull to keep only the vertices of the
    combined set.

    Args:
        my_hull: (M, 3) array of this robot's current hull vertices.
        neighbor_hulls: List of (M_j, 3) arrays from each neighbor.
            Entries may be None or empty, which are skipped.

    Returns:
        Updated hull vertices as (K, 3) array. K is the number of vertices
        of the convex hull of the merged point set.
    """
    all_points = [my_hull]
    for nh in neighbor_hulls:
        if nh is not None and len(nh) > 0:
            all_points.append(np.atleast_2d(nh))
    merged = np.vstack(all_points)
    return extract_hull_vertices_3d(merged)
