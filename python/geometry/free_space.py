"""Local obstacle-free convex region construction (Phase 3) and
distributed consensus intersection (Phase 4).

Section 3.3 of the paper: each robot computes a large convex region in
obstacle-free position-time space P_i ⊂ R^3 × [0, τ], then robots run
distributed consensus (Algorithm 3) to agree on the intersection P = ∩ P_i.

In our 2D simplification we will work in position space R^2 × [0, τ].
The region is represented as a half-space system {x | A x ≤ b}, so that
intersection is simply stacking the (A, b) rows together.

Paper reference: Section 3.3, Algorithm 3, Proposition 2 (paper p. 1086–1087).

TODO: implement in Phase 3.
"""

from __future__ import annotations
import numpy as np
from geometry.rectangles import Rectangle


def visible_obstacles(
    position: np.ndarray,
    obstacles: list[Rectangle],
    sensing_range: float,
) -> list[Rectangle]:
    """Return only the obstacles within sensing_range of position.

    Uses the nearest-point distance to the rectangle, so an obstacle that
    partially overlaps the sensing circle is included even if its center
    is further away.

    Args:
        position:      Robot position, shape (2,).
        obstacles:     Full list of Rectangle obstacles in the environment.
        sensing_range: Radius of the robot's sensing disc.

    Returns:
        Subset of obstacles whose nearest point is within sensing_range.
    """
    return [obs for obs in obstacles if obs.distance_to_point(position) <= sensing_range]


def obstacle_halfspaces(
    query_point: np.ndarray,
    obstacle: Rectangle,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a half-space constraint that separates query_point from one obstacle.

    Finds the face of the obstacle with the largest gap to query_point and
    returns the constraint that keeps the region on query_point's side of it.

    Args:
        query_point: Shape (2,). The reference point — typically χ = c + θ*τ,
                     the centroid projected forward in the preferred direction.
                     The returned constraint keeps the region on this point's side.
        obstacle:    A Rectangle obstacle.

    Returns:
        A_rows: Array of shape (1, 2) — the half-space normal.
        b_vals: Array of shape (1,)  — the half-space bound.
                Returns shape (0, 2) and (0,) if query_point is inside obstacle.

    Notes:
        - Always returns exactly ONE constraint per obstacle — the face with
          the largest gap between query_point and the obstacle.
        - Using χ instead of the centroid means the face selection is driven by
          where the team is heading, not where it currently is. For example, if
          θ* points east through a corridor, χ lands east of the obstacles and
          the y-faces (corridor walls) are selected instead of the x-faces.
    """
    # Compute the gap between query_point and each face.
    # A positive gap means query_point is outside the obstacle on that side.
    gaps = {
        'left':  obstacle.x_min - query_point[0],
        'right': query_point[0] - obstacle.x_max,
        'below': obstacle.y_min - query_point[1],
        'above': query_point[1] - obstacle.y_max,
    }

    # Keep only faces where the centroid is actually outside
    outside = {face: gap for face, gap in gaps.items() if gap > 0}

    if not outside:
        # Centroid is inside the obstacle — should not happen in a valid scenario
        return np.zeros((0, 2)), np.zeros((0,))

    # Use only the face with the largest gap — the most separating face.
    # This avoids adding corner constraints that cut off hull vertices on the other axis.
    best = max(outside, key=outside.get)

    if best == 'left':
        return np.array([[1, 0]]),  np.array([obstacle.x_min])
    elif best == 'right':
        return np.array([[-1, 0]]), np.array([-obstacle.x_max])
    elif best == 'below':
        return np.array([[0, 1]]),  np.array([obstacle.y_min])
    else:  # above
        return np.array([[0, -1]]), np.array([-obstacle.y_max])


def obstacle_halfspaces_hull(
    hull_vertices: np.ndarray,
    obstacle: Rectangle,
    chi: np.ndarray,
    eps: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a half-space that separates ALL hull vertices from an obstacle (P^C).

    Two-stage face selection:
        1. A face is *candidate* if all hull vertices have gap ≥ -eps on that side
           (i.e., the entire hull is approximately on the safe side).
        2. Among candidate faces, prefer the one that chi is furthest outside of
           (same direction preference as P^c).  If chi doesn't prefer any candidate
           face, fall back to the candidate with the largest minimum hull gap.

    Using chi to break ties ensures P^C and P^c agree on which obstacle face to
    use.  In particular, once the team is in the corridor and chi points east
    through it, both P^C and P^c pick the corridor's y-faces (y ≥ y_min,
    y ≤ y_max) rather than the x-face — so the agreed region is no longer
    clipped at x = obstacle.x_min and the formation can advance eastward.

    Args:
        hull_vertices: Array of shape (K, 2) — the agreed convex hull vertices.
        obstacle:      A Rectangle obstacle.
        chi:           Shape (2,). Look-ahead query point χ = c + τ·[cos θ*, sin θ*].
        eps:           Tolerance for hull-vertex gap (default 1e-6).

    Returns:
        A_rows: Array of shape (1, 2) — the half-space normal.
        b_vals: Array of shape (1,)  — the half-space bound.
                Returns shape (0, 2) and (0,) if no candidate face exists.
    """
    # Step 1: compute per-face minimum gap across all hull vertices
    gaps_per_face = {
        'left':  [obstacle.x_min - v[0] for v in hull_vertices],
        'right': [v[0] - obstacle.x_max  for v in hull_vertices],
        'below': [obstacle.y_min - v[1] for v in hull_vertices],
        'above': [v[1] - obstacle.y_max  for v in hull_vertices],
    }
    # Candidate faces: hull is approximately on the safe side (gap ≥ -eps)
    candidates = {face: min(gs) for face, gs in gaps_per_face.items()
                  if min(gs) >= -eps}

    if not candidates:
        return np.zeros((0, 2)), np.zeros((0,))

    # Step 2: among candidates, prefer the face that chi is furthest outside of
    chi_gaps = {
        'left':  obstacle.x_min - chi[0],
        'right': chi[0] - obstacle.x_max,
        'below': obstacle.y_min - chi[1],
        'above': chi[1] - obstacle.y_max,
    }
    # chi-preferred candidates: chi must also be on the safe side of that face
    chi_preferred = {f: chi_gaps[f] for f in candidates if chi_gaps[f] > 0}

    if chi_preferred:
        best = max(chi_preferred, key=chi_preferred.get)
    else:
        # chi is not outside any candidate face (obstacle is ahead of chi)
        # fall back to the candidate with the largest minimum hull gap
        best = max(candidates, key=candidates.get)

    if best == 'left':
        return np.array([[1, 0]]),  np.array([obstacle.x_min])
    elif best == 'right':
        return np.array([[-1, 0]]), np.array([-obstacle.x_max])
    elif best == 'below':
        return np.array([[0, 1]]),  np.array([obstacle.y_min])
    else:  # above
        return np.array([[0, -1]]), np.array([-obstacle.y_max])


def compute_hull_free_region(
    hull_vertices: np.ndarray,
    obstacles: list[Rectangle],
    chi: np.ndarray,
    workspace_bounds: tuple[float, float, float, float] = (-1.0, -2.0, 10.0, 8.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Build the P^C half-space region: must contain the entire hull C.

    Uses obstacle_halfspaces_hull so each constraint separates the FULL hull
    from an obstacle, not just the centroid.  chi is passed through so that
    face selection is directed toward the team's look-ahead point.

    Args:
        hull_vertices:    Shape (K, 2). Agreed hull vertices (from Phase 1).
        obstacles:        List of Rectangle obstacles.
        chi:              Shape (2,). Look-ahead point χ = c + τ·[cos θ*, sin θ*].
        workspace_bounds: (x_min, y_min, x_max, y_max).

    Returns:
        A: Array of shape (M, 2).
        b: Array of shape (M,).
    """
    x_min, y_min, x_max, y_max = workspace_bounds

    A_ws = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=float)
    b_ws = np.array([-x_min, x_max, -y_min, y_max], dtype=float)

    A_list = [A_ws]
    b_list = [b_ws]

    for obs in obstacles:
        A_rows, b_vals = obstacle_halfspaces_hull(hull_vertices, obs, chi)
        if len(A_rows) > 0:
            A_list.append(A_rows)
            b_list.append(b_vals)

    return np.vstack(A_list), np.concatenate(b_list)


def compute_local_free_region(
    centroid: np.ndarray,
    hull_vertices: np.ndarray,
    obstacles: list[Rectangle],
    workspace_bounds: tuple[float, float, float, float] = (-1.0, -2.0, 10.0, 8.0),
    theta_star: float = 0.0,
    tau: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full (A, b) half-space representation of the local safe region.

    Computes P_i = P^C_i ∩ P^c_i as described in Section 3.3 of the paper:

        P^C_i: contains the entire convex hull C (all robots already inside).
               Uses obstacle_halfspaces_hull — face valid only if ALL hull
               vertices are on the safe side.

        P^c_i: contains the centroid c, pointed toward χ = c + τ·[cos θ*, sin θ*].
               Uses obstacle_halfspaces — face chosen by where the team is heading.

    The intersection P^C ∩ P^c guarantees:
        1. All robots are already inside P (straight-line paths to targets safe).
        2. The region is biased toward the corridor the team wants to pass through.

    Args:
        centroid:         Shape (2,). Hull centroid.
        hull_vertices:    Shape (K, 2). Agreed convex hull vertices (from Phase 1).
        obstacles:        List of Rectangle obstacles seen by this robot.
        workspace_bounds: (x_min, y_min, x_max, y_max) of the allowed workspace.
        theta_star:       Preferred direction angle in radians (from Phase 2).
        tau:              Look-ahead distance for χ.

    Returns:
        A: Array of shape (M, 2) — all half-space normals stacked.
        b: Array of shape (M,)  — all half-space bounds stacked.
    """
    # --- P^c: centroid-based region (existing logic) ---
    x_min, y_min, x_max, y_max = workspace_bounds

    # Project centroid forward in the preferred direction to get the query point χ.
    chi = centroid + tau * np.array([np.cos(theta_star), np.sin(theta_star)])

    A_ws = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=float)
    b_ws = np.array([-x_min, x_max, -y_min, y_max], dtype=float)

    A_c_list = [A_ws]
    b_c_list = [b_ws]
    for obs in obstacles:
        A_rows, b_vals = obstacle_halfspaces(chi, obs)
        if len(A_rows) > 0:
            A_c_list.append(A_rows)
            b_c_list.append(b_vals)
    A_c = np.vstack(A_c_list)
    b_c = np.concatenate(b_c_list)

    # --- P^C: hull-based region (new) ---
    # Workspace bounds already included in P^c; only add obstacle constraints here.
    # Pass chi so face selection is directed toward the look-ahead point.
    A_C_list: list[np.ndarray] = []
    b_C_list: list[np.ndarray] = []
    for obs in obstacles:
        A_rows, b_vals = obstacle_halfspaces_hull(hull_vertices, obs, chi)
        if len(A_rows) > 0:
            A_C_list.append(A_rows)
            b_C_list.append(b_vals)

    if A_C_list:
        A_C = np.vstack(A_C_list)
        b_C = np.concatenate(b_C_list)
        A = np.vstack([A_c, A_C])
        b = np.concatenate([b_c, b_C])
    else:
        A, b = A_c, b_c

    return A, b
