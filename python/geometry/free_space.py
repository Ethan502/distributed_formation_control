"""Local obstacle-free convex region construction (Phase 3) and
distributed consensus intersection (Phase 4).

Section 3.3 of the paper: each robot computes a large convex region in
obstacle-free position-time space P_i ⊂ R^3 × [0, τ], then robots run
distributed consensus (Algorithm 3) to agree on the intersection P = ∩ P_i.

In our 2D simplification we will work in position space R^2 × [0, τ].
The region is represented as a half-space system {x | A x ≤ b}, so that
intersection is simply stacking the (A, b) rows together.

Paper reference: Section 3.3, Algorithm 3, Proposition 2 (paper p. 1086–1087).
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import HalfspaceIntersection, ConvexHull
from geometry.rectangles import Rectangle


def compute_separating_halfspace(seed_point, obstacle):
    gap_left = obstacle.x_min - seed_point[0]
    gap_right = seed_point[0] - obstacle.x_max
    gap_bottom = obstacle.y_min - seed_point[1]
    gap_top = seed_point[1] - obstacle.y_max

    gaps = [gap_left, gap_right, gap_bottom, gap_top]  # fix: was gap_bottom_gap_top
    best = np.argmax(gaps)

    if gaps[best] <= 0:
        return None

    faces = [
        (np.array([ 1.,  0.]),  obstacle.x_min),
        (np.array([-1.,  0.]), -obstacle.x_max),   # fix: was +obstacle.x_max
        (np.array([ 0.,  1.]),  obstacle.y_min),
        (np.array([ 0., -1.]), -obstacle.y_max),   # fix: was +obstacle.y_max
    ]
    return faces[best]
   
def compute_local_free_region(seed_point,obstacles,world_bounds=(-2.,12.,-2.,12.)):
    x_min,x_max,y_min,y_max = world_bounds
    A = np.array([
        [1.,0.],
        [-1.,0.],
        [0.,1.],
        [0.,-1.]
    ])
    b = np.array([x_max,-x_min,y_max,-y_min])

    for obs in obstacles:
        result = compute_separating_halfspace(seed_point, obs)
        if result is None:
            print(f"Warning: seed point is inside obstacle {obs}, skipping")
            continue
        a,b_val = result
        A = np.vstack([A,a])
        b = np.append(b, b_val)

    return A, b


def halfspace_to_vertices(A, b, feasible_point):
    """Convert {x | Ax ≤ b} to a sorted vertex array for plotting.

    scipy expects constraints as A_sci @ x + b_sci ≤ 0, which maps to
    A_sci = A, b_sci = -b, stacked as the single array [A | -b].
    Returns vertices sorted by angle (ready to draw as a polygon), or None.
    """
    try:
        hs_array = np.column_stack([A, -b])
        hi = HalfspaceIntersection(hs_array, feasible_point)
        hull = ConvexHull(hi.intersections)
        verts = hi.intersections[hull.vertices]
        centroid = verts.mean(axis=0)
        angles = np.arctan2(verts[:, 1] - centroid[1], verts[:, 0] - centroid[0])
        return verts[np.argsort(angles)]
    except Exception:
        return None


def is_point_in_region(point, A, b, tol=1e-6):
    """Return True if A @ point ≤ b (within tolerance) for all constraints."""
    return bool(np.all(A @ point <= b + tol))
