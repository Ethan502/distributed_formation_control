"""Formation optimization inside the agreed obstacle-free region (Phase 5).

Given the agreed region P = {x | Ax ≤ b} and a formation template, find
the best placement (center c, scale s, rotation φ) so that every slot
    p_i* = c + s * R(φ) * r_i
lies inside P, while maximizing s (the largest formation that fits).

Because rotation φ makes the problem non-convex, we handle it by
discretizing: try N candidate rotations, solve an LP for each with φ
fixed, and keep the rotation that gives the largest scale.

For a fixed φ the problem is a linear program:

    maximize    s
    subject to  A @ c + s * (A @ R(φ) @ r_i) <= b   for all i
                s >= 0

Paper reference: Section 3.4, Eq. (17)–(18).
"""

from __future__ import annotations
import numpy as np
import cvxpy as cp

from formation.templates import apply_template, rotation_matrix


def solve_fixed_rotation(
    A: np.ndarray,
    b: np.ndarray,
    template: np.ndarray,
    angle: float,
    direction: np.ndarray | None = None,
    centroid: np.ndarray | None = None,
    tau: float = 3.0,
    epsilon: float = 0.01,
) -> tuple[np.ndarray, float] | None:
    """Find the largest scaled formation at a fixed rotation that fits inside P.

    Solves the LP:
        maximize    s + ε · (c · direction)
        subject to  A @ c + s * (A @ R(angle) @ r_i) <= b   for all i
                    direction · c  <=  direction · centroid + tau
                    s >= 0

    The ε · (c · direction) term breaks ties among equally-scaled formations
    by preferring the one whose center is furthest in the preferred direction.

    The travel-distance constraint (direction · c ≤ direction · centroid + tau)
    caps how far the formation center can advance per planning iteration.
    Without it, any positive ε causes the LP to push the center all the way to
    the workspace boundary when the region is open (no obstacles ahead),
    overshooting the goal.

    Args:
        A:         Shape (M, 2) — half-space normals of the agreed region.
        b:         Shape (M,)  — half-space bounds.
        template:  Shape (N, 2) — unit-scale template offsets r_i.
        angle:     Fixed rotation angle φ in radians.
        direction: Shape (2,). Unit vector for preferred direction θ*.
                   Defaults to [1, 0] (east) if None.
        centroid:  Shape (2,). Current hull centroid — used as the reference
                   point for the travel-distance constraint.
                   Defaults to [0, 0] if None.
        tau:       Maximum advance of the formation center in the preferred
                   direction per planning iteration (default 3.0).
        epsilon:   Weight on the directional bias term (default 0.01).

    Returns:
        (center, scale) if the LP is feasible and bounded, else None.
    """
    if direction is None:
        direction = np.array([1.0, 0.0])
    if centroid is None:
        centroid = np.zeros(2)

    R = rotation_matrix(angle)
    c = cp.Variable(2)
    s = cp.Variable(nonneg=True)

    constraints = []
    for r_i in template:
        r_rotated = R @ r_i
        constraints.append(A @ c + s * (A @ r_rotated) <= b)

    # Limit how far the center can advance per iteration
    constraints.append(direction @ c <= direction @ centroid + tau)

    objective = cp.Maximize(s + epsilon * (direction @ c))
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return None
    return (c.value, float(s.value))

def optimize_formation(
    A: np.ndarray,
    b: np.ndarray,
    template: np.ndarray,
    theta_star: float,
    n_rotations: int = 16,
    centroid: np.ndarray | None = None,
    tau: float = 3.0,
) -> tuple[np.ndarray, float, float, np.ndarray] | None:
    """Find the best formation placement by trying several rotation angles.

    Tries n_rotations candidate angles evenly distributed over [0, 2π),
    solves an LP for each via solve_fixed_rotation, and returns the result
    with the largest scale.

    Args:
        A:           Shape (M, 2) — agreed region half-space normals.
        b:           Shape (M,)  — agreed region bounds.
        template:    Shape (N, 2) — unit-scale formation template.
        theta_star:  Preferred direction angle (radians) — used as the
                     starting point for the rotation sweep.
        n_rotations: Number of candidate rotations to try.

    Returns:
        (center, scale, angle, slots) for the best feasible rotation, or
        None if no rotation produced a feasible LP.
        center: Shape (2,) — optimal formation center.
        scale:  Float — optimal scale (largest feasible).
        angle:  Float — rotation angle that gave the best scale.
        slots:  Shape (N, 2) — actual target positions for each robot,
                computed with apply_template(template, center, scale, angle).

    Notes:
        - Candidate angles: np.linspace(0, 2*np.pi, n_rotations, endpoint=False).
          (theta_star is informational — the sweep covers the full circle.)
        - For each angle, call solve_fixed_rotation; skip if it returns None.
        - Track the best (center, scale, angle) seen so far.
        - After the sweep, call apply_template to get the final slot positions.
    """
    candidate_angles = np.linspace(0, 2*np.pi, n_rotations, endpoint=False)
    direction = np.array([np.cos(theta_star), np.sin(theta_star)])

    best_center = None
    best_scale = -np.inf
    best_angle = None

    for angle in candidate_angles:
        result = solve_fixed_rotation(A, b, template, angle, direction=direction,
                                      centroid=centroid, tau=tau)
        if result is None:
            continue
        center, scale = result
        if scale > best_scale:
            best_scale = scale
            best_center = center
            best_angle = angle
    if best_center is None:
        return None

    slots = apply_template(template,best_center,best_scale,best_angle)
    return best_center,best_scale,best_angle,slots
