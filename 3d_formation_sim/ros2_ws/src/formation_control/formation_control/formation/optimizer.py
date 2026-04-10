"""Formation optimization via Linear Programming.

Finds the largest formation (scale) and best placement (center, yaw angle)
that fits entirely within the agreed-upon safe region (intersection of
half-spaces). Optionally biases the center toward the preferred direction.

This is the 3D port of the formation optimizer from the paper.
"""

import numpy as np
import cvxpy as cp
from .templates import rotation_matrix_z


def solve_fixed_rotation(A: np.ndarray, b: np.ndarray,
                         template: np.ndarray, angle: float,
                         direction: np.ndarray = None,
                         centroid: np.ndarray = None,
                         tau: float = 3.0,
                         epsilon: float = 0.01) -> tuple:
    """Solve the formation placement LP for a fixed yaw angle.

    Maximizes formation scale while keeping all formation slots inside
    the safe region {x : A x <= b}. Optionally adds a small incentive
    to push the formation center in the preferred direction.

    Optimization problem:
        maximize  s + epsilon * (direction . c)
        subject to:
            A @ (c + s * R_z(angle) @ r_i) <= b,  for all slots i
            s >= 0
        and optionally:
            direction . c <= direction . centroid + tau

    Args:
        A: (M, 3) constraint normals defining the safe region.
        b: (M,) constraint bounds.
        template: (N, 3) unit-scale formation template vertices.
        angle: Yaw rotation angle in radians.
        direction: (3,) preferred direction unit vector. If None, no
                   directional bias is applied.
        centroid: (3,) current team centroid. Used with direction to
                  bound forward progress.
        tau: Maximum forward progress beyond current centroid.
        epsilon: Weight for the directional incentive term.

    Returns:
        (center, scale) tuple if feasible:
            center: (3,) optimal formation center.
            scale: Non-negative optimal scale.
        None if the problem is infeasible.
    """
    N = template.shape[0]
    M = A.shape[0]

    # Decision variables
    c = cp.Variable(3)       # formation center
    s = cp.Variable()        # formation scale (non-negative)

    # Rotate template vertices by the fixed angle
    R = rotation_matrix_z(angle)
    rotated = template @ R.T  # (N, 3)

    # Constraints: each slot must lie within the safe region
    constraints = [s >= 0]
    for i in range(N):
        # slot_i = c + s * rotated[i]
        # A @ slot_i <= b  =>  A @ c + s * A @ rotated[i] <= b
        constraints.append(A @ c + s * (A @ rotated[i]) <= b)

    # Optional: bound forward progress
    if direction is not None and centroid is not None:
        constraints.append(direction @ c <= direction @ centroid + tau)

    # Objective: maximize scale with small directional incentive
    if direction is not None:
        objective = cp.Maximize(s + epsilon * (direction @ c))
    else:
        objective = cp.Maximize(s)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        try:
            prob.solve(solver=cp.ECOS)
        except cp.SolverError:
            return None

    if prob.status in ('infeasible', 'infeasible_inaccurate', 'unbounded'):
        return None

    if c.value is None or s.value is None:
        return None

    return np.array(c.value).flatten(), float(s.value)


def optimize_formation(A: np.ndarray, b: np.ndarray,
                       template: np.ndarray,
                       direction: np.ndarray,
                       n_rotations: int = 16,
                       centroid: np.ndarray = None,
                       tau: float = 3.0) -> tuple:
    """Find the best formation placement by sweeping over yaw angles.

    Tries n_rotations evenly spaced yaw angles from 0 to 2*pi, solves
    the placement LP for each, and returns the configuration with the
    best combined score (scale + epsilon * directional_progress).

    Args:
        A: (M, 3) constraint normals defining the safe region.
        b: (M,) constraint bounds.
        template: (N, 3) unit-scale formation template vertices.
        direction: (3,) preferred direction unit vector.
        n_rotations: Number of yaw angles to sweep.
        centroid: (3,) current team centroid.
        tau: Maximum forward progress beyond current centroid.

    Returns:
        (center, scale, angle, slots) tuple if any angle is feasible:
            center: (3,) optimal formation center.
            scale: Non-negative optimal scale.
            angle: Best yaw angle in radians.
            slots: (N, 3) array of target slot positions.
        None if all angles are infeasible.
    """
    epsilon = 0.01
    best_result = None
    best_score = -np.inf

    angles = np.linspace(0, 2 * np.pi, n_rotations, endpoint=False)

    for angle in angles:
        result = solve_fixed_rotation(
            A, b, template, angle,
            direction=direction,
            centroid=centroid,
            tau=tau,
            epsilon=epsilon,
        )
        if result is None:
            continue

        center, scale = result

        # Score: scale plus directional progress incentive
        if direction is not None:
            score = scale + epsilon * np.dot(direction, center)
        else:
            score = scale

        if score > best_score:
            best_score = score
            R = rotation_matrix_z(angle)
            slots = center + scale * (template @ R.T)
            best_result = (center, scale, angle, slots)

    return best_result
