"""Polygon-aware geometry for the me595 alternate runner.

The base ``python/`` pipeline assumes axis-aligned rectangle obstacles
in three places:

    * ``python/consensus/preferred_direction.ray_clearance``
    * ``python/geometry/free_space.obstacle_halfspaces`` and
      ``obstacle_halfspaces_hull`` (face selection in 4-direction enum)
    * ``python/robots/dynamics.project_velocity_obstacles``

The me595 scenario adds a triangle obstacle, so this module supplies
generalized versions that work on any **convex** polygon (Triangle,
Rectangle, or any future convex shape).

Convention used throughout this module:

    A convex polygon is described by its CCW vertices  v_0, ..., v_{K-1}.
    The face between v_i and v_{i+1} has outward normal

        n_i = ( (v_{i+1}.y - v_i.y),  -(v_{i+1}.x - v_i.x) )  / ||edge||

    A point p is **outside** the polygon on face i  iff   n_i · p > b_i,
    where  b_i = n_i · v_i  is the face offset.  The polygon's interior
    is the set { x | n_i · x <= b_i  for all i }.

Anything we add to the team's free-space half-space system has the
opposite sign — we want the team to stay on the *outside* of a face,
which becomes  -n_i · x  <=  -b_i.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np

from me595.rectangle import Rectangle
from me595.triangle import Triangle, _point_to_segment_distance


# A "convex obstacle" is anything we can hand to ``polygon_vertices`` and
# ``polygon_face_halfspaces`` below.  Currently those are the two simple
# shapes used by the me595 scenario.
ConvexObstacle = Union[Triangle, Rectangle]


# ---------------------------------------------------------------------------
# Polygon -> CCW vertex / face representation
# ---------------------------------------------------------------------------

def polygon_vertices(obstacle: ConvexObstacle) -> np.ndarray:
    """Return the obstacle's vertices as an (K, 2) array in CCW order.

    Triangles already store CCW vertices.  Rectangles are unwrapped into
    their four corners (bottom-left, bottom-right, top-right, top-left).
    """
    if isinstance(obstacle, Triangle):
        return obstacle.vertices
    if isinstance(obstacle, Rectangle):
        return obstacle.corners
    raise TypeError(f"Unknown obstacle type: {type(obstacle).__name__}")


def polygon_face_halfspaces(
    obstacle: ConvexObstacle,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-face outward normals and offsets.

    Each face i is defined by  n_i · x = b_i  with  n_i  the unit
    outward normal.  The polygon's interior is  { x | n_i · x <= b_i  ∀ i }.

    Returns:
        normals: Array (K, 2) of outward unit normals, one per face.
        offsets: Array (K,)  with  b_i = n_i · v_i.
    """
    verts = polygon_vertices(obstacle)
    k = len(verts)
    normals = np.zeros((k, 2))
    offsets = np.zeros(k)
    for i in range(k):
        v_i = verts[i]
        v_next = verts[(i + 1) % k]
        edge = v_next - v_i
        # Right-hand normal of the CCW edge points OUTWARD.
        n = np.array([edge[1], -edge[0]], dtype=float)
        norm = float(np.linalg.norm(n))
        if norm < 1e-12:
            raise ValueError("Degenerate polygon: zero-length edge.")
        n /= norm
        normals[i] = n
        offsets[i] = float(n @ v_i)
    return normals, offsets


def polygon_distance_to_point(
    obstacle: ConvexObstacle, point: np.ndarray
) -> float:
    """Minimum distance from ``point`` to the obstacle (0 if inside)."""
    if isinstance(obstacle, Rectangle):
        return obstacle.distance_to_point(np.asarray(point, dtype=float))
    if isinstance(obstacle, Triangle):
        return obstacle.distance_to_point(np.asarray(point, dtype=float))
    # Generic convex polygon fallback (kept for future shapes).
    p = np.asarray(point, dtype=float)
    verts = polygon_vertices(obstacle)
    if polygon_contains_point(obstacle, p):
        return 0.0
    k = len(verts)
    return min(
        _point_to_segment_distance(p, verts[i], verts[(i + 1) % k])
        for i in range(k)
    )


def polygon_contains_point(
    obstacle: ConvexObstacle, point: np.ndarray
) -> bool:
    """Strict containment test for any convex CCW polygon."""
    p = np.asarray(point, dtype=float)
    normals, offsets = polygon_face_halfspaces(obstacle)
    return bool(np.all(normals @ p < offsets))


# ---------------------------------------------------------------------------
# Ray-vs-convex-polygon clearance (replaces ray-vs-AABB)
# ---------------------------------------------------------------------------

def ray_polygon_clearance(
    position: np.ndarray,
    theta: float,
    obstacle: ConvexObstacle,
    max_range: float,
) -> float:
    """Distance along a ray before it first enters this convex obstacle.

    Uses the half-plane method: a convex polygon is the intersection of
    its outward half-planes  { n_i · x <= b_i }.  For a parametric ray
    p + t·d, slab i constrains  n_i·(p + t·d) <= b_i:

        if  d·n_i > 0:  the ray EXITS slab i at  t = (b_i - n_i·p)/(d·n_i)
        if  d·n_i < 0:  the ray ENTERS slab i at  t = (b_i - n_i·p)/(d·n_i)

    The ray is inside the polygon for t in [t_enter, t_exit] where
    t_enter = max over entering faces and t_exit = min over exiting
    faces.  The polygon is hit iff t_enter < t_exit and t_exit > 0.

    Args:
        position:  Robot / query origin, shape (2,).
        theta:     Direction angle in radians.
        obstacle:  A convex obstacle.
        max_range: Sensing cap returned when the ray misses.

    Returns:
        Distance to the obstacle along the ray, or ``max_range`` if missed.
        Returns 0 if ``position`` lies inside the obstacle.
    """
    d = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    p = np.asarray(position, dtype=float)
    normals, offsets = polygon_face_halfspaces(obstacle)

    t_enter = -np.inf
    t_exit = np.inf
    for n, b in zip(normals, offsets):
        denom = float(d @ n)
        rhs = float(b - n @ p)
        if abs(denom) < 1e-12:
            # Ray parallel to this face: must already be on the safe side.
            if rhs < 0:
                return max_range  # ray will never satisfy this slab → miss
            continue
        t = rhs / denom
        if denom > 0:
            # Ray exits this slab at t.
            if t < t_exit:
                t_exit = t
        else:
            # Ray enters this slab at t.
            if t > t_enter:
                t_enter = t

    if t_enter < t_exit and t_exit > 0.0:
        return float(max(t_enter, 0.0))
    return max_range


def ray_clearance(
    position: np.ndarray,
    theta: float,
    obstacles: Sequence[ConvexObstacle],
    max_range: float = 20.0,
) -> float:
    """Distance to the nearest obstacle along direction theta, capped at max_range.

    Same shape as ``python/consensus/preferred_direction.ray_clearance``
    but works on a list of arbitrary convex polygons instead of axis-
    aligned rectangles.
    """
    closest = max_range
    for obs in obstacles:
        closest = min(closest, ray_polygon_clearance(position, theta, obs, max_range))
    return closest


# ---------------------------------------------------------------------------
# Free-space construction (polygon-aware versions of python/geometry/free_space)
# ---------------------------------------------------------------------------

def visible_obstacles(
    position: np.ndarray,
    obstacles: Iterable[ConvexObstacle],
    sensing_range: float,
) -> list[ConvexObstacle]:
    """Filter obstacles to those whose nearest point lies within sensing_range."""
    p = np.asarray(position, dtype=float)
    return [
        obs
        for obs in obstacles
        if polygon_distance_to_point(obs, p) <= sensing_range
    ]


def obstacle_halfspaces(
    query_point: np.ndarray,
    obstacle: ConvexObstacle,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the most-separating face of a convex obstacle vs ``query_point``.

    Generalizes the rectangle version (which enumerated 4 faces).  For an
    arbitrary convex polygon we evaluate the gap

        gap_i = n_i · query - b_i

    for every face.  A positive gap means ``query_point`` lies on the
    outside (safe) side of face i.  We pick the face with the largest
    gap and return the half-space constraint  -n_i · x <= -b_i  that
    keeps the team on ``query_point``'s side of that face.

    If ``query_point`` is inside the obstacle (no positive gap), no
    constraint is returned (rows of shape (0, 2)).
    """
    q = np.asarray(query_point, dtype=float)
    normals, offsets = polygon_face_halfspaces(obstacle)
    gaps = normals @ q - offsets

    best_i = int(np.argmax(gaps))
    if gaps[best_i] <= 0.0:
        return np.zeros((0, 2)), np.zeros((0,))

    n = normals[best_i]
    b = offsets[best_i]
    A_row = -n.reshape(1, 2)
    b_row = np.array([-b])
    return A_row, b_row


def obstacle_halfspaces_hull(
    hull_vertices: np.ndarray,
    obstacle: ConvexObstacle,
    chi: np.ndarray,
    eps: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a face that separates the *entire hull* from a convex obstacle.

    Two-stage face selection mirroring
    ``python/geometry/free_space.obstacle_halfspaces_hull``:

        1. A face i is *candidate* iff every hull vertex has  gap_i ≥ -eps.
           (the whole hull is approximately on the safe side).
        2. Among candidates, prefer the one whose look-ahead point ``chi``
           is furthest outside.  Fall back to the candidate with the
           largest minimum hull gap if chi doesn't strictly prefer any.

    Args:
        hull_vertices: Shape (H, 2). Agreed convex-hull vertices.
        obstacle:      A convex obstacle.
        chi:           Look-ahead point  c + tau·[cos θ*, sin θ*].
        eps:           Tolerance for hull-vertex gap.
    """
    H = np.asarray(hull_vertices, dtype=float)
    chi = np.asarray(chi, dtype=float)

    normals, offsets = polygon_face_halfspaces(obstacle)
    K = len(normals)

    # Per-face gap of every hull vertex (shape (K, H)).
    hull_gaps = (normals @ H.T) - offsets[:, None]
    min_hull_gap = hull_gaps.min(axis=1)

    candidate_mask = min_hull_gap >= -eps
    if not np.any(candidate_mask):
        return np.zeros((0, 2)), np.zeros((0,))

    chi_gaps = normals @ chi - offsets
    chi_preferred = candidate_mask & (chi_gaps > 0)

    if np.any(chi_preferred):
        idx_pool = np.where(chi_preferred)[0]
        best_i = int(idx_pool[np.argmax(chi_gaps[idx_pool])])
    else:
        idx_pool = np.where(candidate_mask)[0]
        best_i = int(idx_pool[np.argmax(min_hull_gap[idx_pool])])

    n = normals[best_i]
    b = offsets[best_i]
    return -n.reshape(1, 2), np.array([-b])


def compute_local_free_region(
    centroid: np.ndarray,
    hull_vertices: np.ndarray,
    obstacles: Sequence[ConvexObstacle],
    workspace_bounds: tuple[float, float, float, float],
    theta_star: float = 0.0,
    tau: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Polygon-aware version of ``compute_local_free_region``.

    Builds  P_i = P^C_i ∩ P^c_i  exactly like ``python/geometry/free_space``
    but using the polygon-aware ``obstacle_halfspaces`` /
    ``obstacle_halfspaces_hull`` so triangle obstacles are handled.
    """
    x_min, y_min, x_max, y_max = workspace_bounds

    chi = centroid + tau * np.array([np.cos(theta_star), np.sin(theta_star)])

    # Workspace box.
    A_ws = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=float)
    b_ws = np.array([-x_min, x_max, -y_min, y_max], dtype=float)

    # P^c — centroid-driven (uses chi as query point).
    A_c_list = [A_ws]
    b_c_list = [b_ws]
    for obs in obstacles:
        A_rows, b_vals = obstacle_halfspaces(chi, obs)
        if len(A_rows) > 0:
            A_c_list.append(A_rows)
            b_c_list.append(b_vals)
    A_c = np.vstack(A_c_list)
    b_c = np.concatenate(b_c_list)

    # P^C — hull-driven.
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
        return np.vstack([A_c, A_C]), np.concatenate([b_c, b_C])
    return A_c, b_c


# ---------------------------------------------------------------------------
# Polygon-aware preferred-direction utilities
# ---------------------------------------------------------------------------

def compute_local_utilities(
    position: np.ndarray,
    obstacles: Sequence[ConvexObstacle],
    candidate_angles: np.ndarray,
    max_range: float = 20.0,
    goal_angle: float | None = None,
) -> np.ndarray:
    """Per-direction clearance utility, optionally weighted toward the goal.

    Same shape and semantics as
    ``python/consensus/preferred_direction.compute_local_utilities``,
    but ``obstacles`` may contain triangles in addition to rectangles.
    """
    utilities = np.array([
        ray_clearance(position, theta, obstacles, max_range)
        for theta in candidate_angles
    ])

    if goal_angle is not None:
        angular_diff = np.abs(np.arctan2(
            np.sin(candidate_angles - goal_angle),
            np.cos(candidate_angles - goal_angle),
        ))
        goal_weight = 0.2 + 0.8 * (1.0 - angular_diff / np.pi)
        utilities *= goal_weight

    return utilities


def run_direction_consensus(
    positions: np.ndarray,
    adjacency: np.ndarray,
    obstacles: Sequence[ConvexObstacle],
    candidate_angles: np.ndarray,
    num_rounds: int,
    max_range: float = 20.0,
    centroid: np.ndarray | None = None,
    goal: np.ndarray | None = None,
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    """Polygon-aware copy of ``run_direction_consensus``.

    Identical update rule (component-wise min over neighbors), but the
    initial utilities use ``compute_local_utilities`` from this module so
    triangles are handled correctly.  Like the original we evaluate ray
    clearance from the hull centroid (if supplied) rather than each
    individual robot position.
    """
    N = positions.shape[0]
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape (N, 2), got {positions.shape}")
    if adjacency.shape != (N, N):
        raise ValueError(f"adjacency must have shape ({N}, {N}), got {adjacency.shape}")

    goal_angle = None
    if centroid is not None and goal is not None:
        diff = goal - centroid
        goal_angle = float(np.arctan2(diff[1], diff[0]))

    query_points = [
        centroid if centroid is not None else positions[i] for i in range(N)
    ]
    utilities = [
        compute_local_utilities(
            query_points[i], obstacles, candidate_angles, max_range, goal_angle
        )
        for i in range(N)
    ]
    history = [[u.copy() for u in utilities]]

    for _ in range(num_rounds):
        broadcast = [u.copy() for u in utilities]
        for i in range(N):
            neighbors = np.where(adjacency[i] == 1)[0]
            merged = broadcast[i].copy()
            for j in neighbors:
                merged = np.minimum(merged, broadcast[j])
            utilities[i] = merged
        history.append([u.copy() for u in utilities])

    return history, np.array(utilities)
