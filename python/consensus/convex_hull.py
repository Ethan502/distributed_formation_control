"""Distributed convex hull consensus algorithm — Algorithm 1 from the paper.

Section 3.1 describes how all robots reach agreement on the convex hull C
of their positions using only local communication over the graph G.

Key idea:
    - Each robot i maintains a local estimate C_i of the global hull.
    - Initialized with only its own position: C_i(0) = {p_i}.
    - At each round k, robot i:
        (a) sends its NEW hull points  C̃_i(k) = C_i(k) \\ C_i(k-1)  to neighbors,
        (b) receives new points from all neighbors j ∈ N_i,
        (c) updates:  C_i(k+1) = convhull( C_i(k) ∪ received points ).
    - After d rounds (graph diameter), all robots converge:
        C_i(d) = C,  for all i  (Proposition 1).

Paper reference: Algorithm 1, Proposition 1 (paper p. 1084–1085).
"""

from __future__ import annotations
import numpy as np

from geometry.hulls import extract_hull_vertices


def run_convex_hull_consensus(
    points: np.ndarray,
    adjacency: np.ndarray,
    num_rounds: int,
) -> list[list[np.ndarray]]:
    """Run the distributed convex hull consensus algorithm for a robot team.

    Args:
        points:     Array of shape (N, 2) with initial robot positions.
        adjacency:  Binary adjacency matrix of shape (N, N).
        num_rounds: Number of consensus rounds to execute.  Should be at
                    least the graph diameter for guaranteed convergence.

    Returns:
        history: A list of length (num_rounds + 1).
                 history[k] is a list of N arrays, where history[k][i]
                 is the set of hull vertices that robot i holds after round k,
                 stored as an array of shape (M_i, 2).
                 history[0] contains initialization: each robot holds only
                 its own position as a (1, 2) array.

    Notes:
        - To save communication bandwidth, robots only broadcast NEWLY
          discovered hull vertices at each round (points not in their
          previous estimate).  This mirrors the paper's efficiency idea.
          For a simpler first version you may broadcast the full hull each
          round — this is correct but less efficient.
        - Represent each robot's candidate set as a list of 2D point arrays.
          Recompute the hull vertex set after each merge step.
        - Edge case: if two robots are at identical positions, the convex
          hull degenerates to a line or point.  extract_hull_vertices
          should handle this gracefully.
    """
    # Step 1: Validate that points has shape (N, 2) and adjacency is (N, N).
    N = points.shape[0]
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N,2), got {points.shape}")
    if adjacency.shape != (N,N):
        raise ValueError(f"adjacency must have shape ({N},{N}), got {adjacency.shape}")
    # Step 2: Initialize each robot's candidate point set with its own position.
    candidate_points = [np.array([points[i]]) for i in range(N)]
    prev_hull = [np.array([points[i]]) for i in range(N)]
    history = [[cp.copy() for cp in candidate_points]]

    for k in range(num_rounds):
        broadcast = [candidate_points[i].copy() for i in range(N)] #TODO: Still need to make this into the efficient version
        new_candidate_points = []
        for i in range(N):
            neighbors = np.where(adjacency[i] == 1)[0]
            received = [broadcast[j] for j in neighbors]
            merged = np.vstack([candidate_points[i]]+received) # stack own + all received
            new_candidate_points.append(merged)

        round_history = []
        for i in range(N):
            prev_hull[i] = candidate_points[i].copy()
            candidate_points[i] = extract_hull_vertices(new_candidate_points[i])
            round_history.append(candidate_points[i].copy())
        history.append(round_history)
    return history


def sort_vertices_by_angle(verts: np.ndarray) -> np.ndarray:
# verts: shape (K,2)
    centroid = verts.mean(axis=0)
    angles = np.arctan2(verts[:,1] - centroid[1], verts[:,0] - centroid[0])
    return verts[np.argsort(angles)]

def did_hulls_converge(
    history: list[list[np.ndarray]],
) -> tuple[bool, int]:
    """Check whether the local hull estimates have converged to a common hull.

    Two hull estimates are considered equal if they contain the same set of
    vertices (up to ordering and floating-point tolerance).

    Args:
        history: The list returned by run_convex_hull_consensus.

    Returns:
        converged:     True if, in the final round, all robots share the
                       same hull vertex set.
        convergence_k: The index of the first round at which all estimates
                       were equal, or len(history)-1 if convergence was never
                       detected.

    Notes:
        - To compare two vertex sets independent of ordering, sort vertices
          by angle from their centroid before comparing with np.allclose.
        - It is also acceptable to compare the areas of the convex hulls
          as a quick proxy for equality.
        - Returns (True, 0) if num_rounds=0 and there is only one robot.
    """
    for k in range(len(history)):
        round_estimates = history[k] # List of N arrays
        sorted_0 = sort_vertices_by_angle(round_estimates[0])
        all_match = True
        for i in range(1,len(round_estimates)):
            sorted_i = sort_vertices_by_angle(round_estimates[i])
            if sorted_i.shape != sorted_0.shape:
                all_match = False
                break
            if not np.allclose(sorted_i, sorted_0):
                all_match = False
                break
        if all_match:
            return(True, k)
    return(False, len(history)-1)
