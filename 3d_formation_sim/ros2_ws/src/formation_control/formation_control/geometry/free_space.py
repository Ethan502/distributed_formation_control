"""3D local free-region construction.

Computes a convex safe region around the robot team as an intersection of
half-spaces. Each obstacle contributes a separating half-space that keeps
the robots on the obstacle-free side. Workspace boundaries contribute
additional bounding half-spaces.

This is the 3D port of the key paper algorithm for local free region
construction (used before formation optimization).
"""

import numpy as np
from .boxes import Box


def obstacle_halfspace(query_point: np.ndarray, box: Box) -> tuple:
    """Compute a separating half-space between a query point and a box obstacle.

    The half-space {x : a . x <= b} is chosen so that query_point satisfies
    it (is on the safe side) and the box interior is excluded. Among the 6
    candidate face-aligned half-spaces, the one with the largest margin
    (gap between query_point and the box face) is selected.

    The 6 candidates correspond to the 6 faces of the box:
      +x face: gap = q_x - x_max, a = [-1,0,0], b = -x_max
      -x face: gap = x_min - q_x, a = [1,0,0],  b = x_min
      +y face: gap = q_y - y_max, a = [0,-1,0], b = -y_max
      -y face: gap = y_min - q_y, a = [0,1,0],  b = y_min
      +z face: gap = q_z - z_max, a = [0,0,-1], b = -z_max
      -z face: gap = z_min - q_z, a = [0,0,1],  b = z_min

    If all gaps are negative (point is inside the box), the face with the
    least negative gap is chosen (closest escape direction).

    Args:
        query_point: (3,) position of the robot or centroid.
        box: Box obstacle.

    Returns:
        (a, b): a is (3,) unit normal, b is scalar bound,
                defining the half-space a . x <= b.
    """
    q = query_point

    # Each candidate: (gap, normal_vector, bound)
    candidates = [
        (q[0] - box.x_max, np.array([-1.0, 0.0, 0.0]), -box.x_max),  # +x face
        (box.x_min - q[0], np.array([1.0, 0.0, 0.0]),   box.x_min),   # -x face
        (q[1] - box.y_max, np.array([0.0, -1.0, 0.0]), -box.y_max),   # +y face
        (box.y_min - q[1], np.array([0.0, 1.0, 0.0]),   box.y_min),   # -y face
        (q[2] - box.z_max, np.array([0.0, 0.0, -1.0]), -box.z_max),   # +z face
        (box.z_min - q[2], np.array([0.0, 0.0, 1.0]),   box.z_min),   # -z face
    ]

    # Pick the face with the largest gap
    best_idx = 0
    best_gap = candidates[0][0]
    for i in range(1, 6):
        if candidates[i][0] > best_gap:
            best_gap = candidates[i][0]
            best_idx = i

    _, a, b = candidates[best_idx]
    return a, b


def obstacle_halfspace_hull(hull_vertices: np.ndarray, box: Box,
                            chi: np.ndarray, eps: float = 0.1) -> tuple:
    """Compute a separating half-space that keeps ALL hull vertices safe.

    Like obstacle_halfspace, but ensures every hull vertex satisfies the
    constraint a . v <= b. This is needed because the formation must fit
    entirely within the safe region.

    Two-stage face selection:
      1. Find candidate faces where ALL hull vertices satisfy a . v <= b
         (with eps margin for numerical safety).
      2. Among valid candidates, pick the face whose normal is most aligned
         with the direction toward chi (the look-ahead point), measured by
         a . (chi - box.center).

    If no face satisfies all hull vertices, fall back to the face with the
    smallest maximum violation across hull vertices.

    Args:
        hull_vertices: (M, 3) array of convex hull vertex positions.
        box: Box obstacle.
        chi: (3,) look-ahead point for direction-aware face selection.
        eps: Margin tolerance for hull vertex satisfaction.

    Returns:
        (a, b): a is (3,) unit normal, b is scalar bound.
    """
    # Define the 6 candidate faces: (normal, bound)
    faces = [
        (np.array([-1.0, 0.0, 0.0]), -box.x_max),  # +x face
        (np.array([1.0, 0.0, 0.0]),   box.x_min),   # -x face
        (np.array([0.0, -1.0, 0.0]), -box.y_max),   # +y face
        (np.array([0.0, 1.0, 0.0]),   box.y_min),   # -y face
        (np.array([0.0, 0.0, -1.0]), -box.z_max),   # +z face
        (np.array([0.0, 0.0, 1.0]),   box.z_min),   # -z face
    ]

    box_center = box.center
    chi_dir = chi - box_center  # direction from box center toward look-ahead

    # Evaluate each face
    valid_faces = []       # (index, alignment_score)
    fallback_faces = []    # (index, max_violation)

    for i, (a, b) in enumerate(faces):
        # Check if all hull vertices satisfy a . v <= b + eps
        violations = hull_vertices @ a - b  # positive means violation
        max_violation = np.max(violations)

        if max_violation <= eps:
            # All vertices are on the safe side (within tolerance)
            alignment = np.dot(a, chi_dir)
            valid_faces.append((i, alignment))
        else:
            fallback_faces.append((i, max_violation))

    if valid_faces:
        # Pick the face most aligned with chi direction
        best = max(valid_faces, key=lambda x: x[1])
        a, b = faces[best[0]]
    else:
        # No face satisfies all vertices; pick least-violating
        best = min(fallback_faces, key=lambda x: x[1])
        a, b = faces[best[0]]

    return a, b


def compute_local_free_region(centroid: np.ndarray, hull_vertices: np.ndarray,
                              obstacles: list, workspace_bounds: list,
                              theta_star: np.ndarray,
                              tau: float = 3.0,
                              sensing_range: float = 6.0) -> tuple:
    """Compute the local free region as an intersection of half-spaces.

    Constructs the region {x : A x <= b} that is:
      - free of nearby obstacles (via separating half-spaces),
      - within the workspace boundaries.

    The look-ahead point chi = centroid + tau * theta_star is used for
    direction-aware face selection (preferring half-spaces that leave room
    in the preferred travel direction).

    For each nearby obstacle (within sensing_range of centroid):
      - A centroid-based half-space via obstacle_halfspace
      - A hull-based half-space via obstacle_halfspace_hull

    Plus 6 workspace boundary constraints.

    Args:
        centroid: (3,) centroid of the robot team (or current hull).
        hull_vertices: (M, 3) current convex hull vertices.
        obstacles: List of Box obstacles.
        workspace_bounds: [x_min, y_min, z_min, x_max, y_max, z_max] workspace limits.
        theta_star: (3,) preferred direction unit vector.
        tau: Look-ahead distance multiplier for chi computation.
        sensing_range: Maximum distance from centroid to consider an obstacle.

    Returns:
        (A, b) tuple:
            A: (K, 3) constraint normals.
            b: (K,) constraint bounds.
            Defining the region {x : A x <= b}.
    """
    A_rows = []
    b_vals = []

    # Look-ahead point for direction-aware face selection
    chi = centroid + tau * theta_star

    # Process each obstacle within sensing range
    for obs in obstacles:
        dist = obs.distance_to_point(centroid)
        if dist > sensing_range:
            continue

        # Centroid-based separating half-space
        a_c, b_c = obstacle_halfspace(centroid, obs)
        A_rows.append(a_c)
        b_vals.append(b_c)

        # Hull-based separating half-space (ensures all hull vertices are safe)
        if hull_vertices is not None and len(hull_vertices) > 0:
            a_h, b_h = obstacle_halfspace_hull(hull_vertices, obs, chi)
            A_rows.append(a_h)
            b_vals.append(b_h)

    # Workspace boundary constraints (convention: a . x <= b)
    ws = workspace_bounds  # [x_min, y_min, z_min, x_max, y_max, z_max]

    # x <= x_max
    A_rows.append(np.array([1.0, 0.0, 0.0]))
    b_vals.append(ws[3])

    # x >= x_min  =>  -x <= -x_min
    A_rows.append(np.array([-1.0, 0.0, 0.0]))
    b_vals.append(-ws[0])

    # y <= y_max
    A_rows.append(np.array([0.0, 1.0, 0.0]))
    b_vals.append(ws[4])

    # y >= y_min  =>  -y <= -y_min
    A_rows.append(np.array([0.0, -1.0, 0.0]))
    b_vals.append(-ws[1])

    # z <= z_max
    A_rows.append(np.array([0.0, 0.0, 1.0]))
    b_vals.append(ws[5])

    # z >= z_min  =>  -z <= -z_min
    A_rows.append(np.array([0.0, 0.0, -1.0]))
    b_vals.append(-ws[2])

    A = np.array(A_rows)
    b = np.array(b_vals)
    return A, b
