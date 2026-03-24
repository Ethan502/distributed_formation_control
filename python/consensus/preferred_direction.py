"""Distributed preferred direction of motion consensus — Algorithm 2 (Phase 2).

Section 3.2 of the paper: robots agree on the best direction θ* for the team
to move by running a min-consensus on per-robot utility vectors.

Key idea:
    - A discrete set Θ = {θ_1, ..., θ_κ} of candidate angles is shared by all.
    - Each robot i assigns a utility u_i(θ) ≥ 0 to each angle θ based on its
      local perception (e.g., clearance from obstacles in direction θ).
    - The global utility u(θ) = min_{i} u_i(θ).
    - The team selects θ* = argmax_θ u(θ)  (best worst-case direction).
    - Distributed update (Algorithm 2):
        u_i(k+1) = min_{j ∈ N_i} ( u_i(k), u_j(k) )  component-wise.
    - After d rounds, every robot knows the global utility vector.

Paper reference: Algorithm 2, Section 3.2 (paper p. 1085–1086).
"""
from __future__ import annotations
import numpy as np
from geometry.rectangles import Rectangle


def ray_clearance(
    position: np.ndarray,
    theta: float,
    obstacles: list[Rectangle],
    max_range: float = 20.0,
) -> float:
    """Return the distance from position in direction theta before hitting an obstacle.

    Uses the slab method for ray-AABB (axis-aligned bounding box) intersection.
    For each obstacle, find the range of t values where the ray p + t*d is inside
    the box on each axis, then check if those ranges overlap.

    Args:
        position:  Robot position, shape (2,).
        theta:     Direction angle in radians.
        obstacles: List of Rectangle obstacles.
        max_range: Maximum sensing distance. Returned if nothing is hit.

    Returns:
        Distance to the nearest obstacle in direction theta, capped at max_range.
    """
    d = np.array([np.cos(theta), np.sin(theta)])
    closest = max_range

    for obs in obstacles:
        # --- x-axis slab ---
        if abs(d[0]) < 1e-9:
            # Ray is parallel to the x faces (moving purely vertically).
            # Can only hit this box if robot is already between the x faces.
            if position[0] < obs.x_min or position[0] > obs.x_max:
                continue
            tx_min, tx_max = -np.inf, np.inf
        else:
            tx_min = (obs.x_min - position[0]) / d[0]
            tx_max = (obs.x_max - position[0]) / d[0]
            if tx_min > tx_max:
                tx_min, tx_max = tx_max, tx_min

        # --- y-axis slab ---
        if abs(d[1]) < 1e-9:
            # Ray is parallel to the y faces (moving purely horizontally).
            if position[1] < obs.y_min or position[1] > obs.y_max:
                continue
            ty_min, ty_max = -np.inf, np.inf
        else:
            ty_min = (obs.y_min - position[1]) / d[1]
            ty_max = (obs.y_max - position[1]) / d[1]
            if ty_min > ty_max:
                ty_min, ty_max = ty_max, ty_min

        # The ray is inside BOTH slabs between t_enter and t_exit.
        t_enter = max(tx_min, ty_min)
        t_exit  = min(tx_max, ty_max)

        # Valid hit: slabs overlap (t_enter < t_exit) and box is ahead (t_exit > 0).
        if t_enter < t_exit and t_exit > 0:
            # If t_enter < 0 the robot starts inside the box — count distance as 0.
            t_hit = max(t_enter, 0.0)
            closest = min(closest, t_hit)

    return closest

def compute_local_utilities(position: np.ndarray, obstacles: list[Rectangle],candidate_angles: np.ndarray, max_range: float=20.0) -> np.ndarray:
    utilities = np.zeros(len(candidate_angles),)
    for k,theta in enumerate(candidate_angles):
        utilities[k] = ray_clearance(position,theta,obstacles,max_range)
    return utilities

def run_direction_consensus(
    positions: np.ndarray,
    adjacency: np.ndarray,
    obstacles: list[Rectangle],
    candidate_angles: np.ndarray,
    num_rounds: int,
    max_range: float=20.0) -> tuple[list[list[np.ndarray]], np.ndarray]:

    """Run Algorithm 2: distributed min-consensus on utility vectors.

      Args:
          positions:        Robot positions, shape (N, 2).
          adjacency:        Binary adjacency matrix, shape (N, N).
          obstacles:        List of Rectangle obstacles.
          candidate_angles: Array of κ candidate angles, shape (κ,).
          num_rounds:       Number of consensus rounds (should be >= graph diameter).
          max_range:        Sensing range passed to compute_local_utilities.

      Returns:
          history: list of length (num_rounds + 1), where history[k] is a list
                   of N arrays of shape (κ,) — each robot's utility vector at round k.
          final_utilities: Array of shape (N, κ) — each robot's utility vector
                           after the final round.

      Notes:
          - Initialize utilities[i] = compute_local_utilities(positions[i], ...)
          - Each round: broadcast own utilities, receive neighbors', take
            component-wise min with np.minimum.
          - All robots should end with the same utility vector after d rounds.
      """
    N = positions.shape[0]
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape (N, 2), got {positions.shape}")
    if adjacency.shape != (N, N):
        raise ValueError(f"adjacency must have shape ({N}, {N}), got {adjacency.shape}")
    
    utilities = [compute_local_utilities(positions[i],obstacles,candidate_angles,max_range) for i in range(N)]
    history = [[u.copy() for u in utilities]]

    for k in range(num_rounds):
        broadcast = [utilities[i].copy() for i in range(N)]
        for i in range(N):
            neighbors = np.where(adjacency[i] == 1)[0]
            merged = broadcast[i].copy()
            for j in neighbors:
                merged = np.minimum(merged, broadcast[j])
            utilities[i] = merged
        history.append([u.copy() for u in utilities])

    return history, np.array(utilities)

def select_preferred_direction(
        utility_vector: np.ndarray,
        candidate_angles: np.ndarray
) -> float:
    index = np.argmax(utility_vector)
    return candidate_angles[index]
