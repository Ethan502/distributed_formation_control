"""Robot-to-formation-slot assignment (Phase 6).

Given current robot positions p_i and target slot positions s_j, solve the
assignment that minimizes total travel distance:

    min_σ  Σ_i  ||p_i - s_{σ(i)}||

where σ is a permutation — each robot gets exactly one slot and each slot
gets exactly one robot.

This is the linear assignment problem, solved in polynomial time by the
Hungarian algorithm via scipy.optimize.linear_sum_assignment.

Steps:
    1. Build the N×N cost matrix C where C[i, j] = ||p_i - s_j||.
    2. Call linear_sum_assignment(C) to get the optimal (row_ind, col_ind).
    3. col_ind[i] is the slot index assigned to robot i.

Paper reference: Section 3.5, Eq. (19).
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def build_cost_matrix(
    positions: np.ndarray,
    slots: np.ndarray,
) -> np.ndarray:
    """Build the N×N cost matrix for the assignment problem.

    C[i, j] = Euclidean distance from robot i's current position to slot j.

    Args:
        positions: Shape (N, 2) — current robot positions.
        slots:     Shape (N, 2) — target slot positions.

    Returns:
        C: Shape (N, N) cost matrix.

    Notes:
        - Use broadcasting: positions[:, np.newaxis, :] has shape (N, 1, 2),
          slots[np.newaxis, :, :] has shape (1, N, 2). Their difference has
          shape (N, N, 2) — one displacement vector per (robot, slot) pair.
        - Apply np.linalg.norm(..., axis=2) to get the (N, N) distance matrix.
    """
    diff = positions[:,np.newaxis,:] - slots[np.newaxis,:,:]
    return np.linalg.norm(diff,axis=2)

def solve_assignment(
    positions: np.ndarray,
    slots: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve the linear assignment problem to match robots to slots.

    Args:
        positions: Shape (N, 2) — current robot positions.
        slots:     Shape (N, 2) — target slot positions from Phase 5.

    Returns:
        assignment: Shape (N,) integer array. assignment[i] = j means
                    robot i is assigned to slot j.
        total_cost: Total distance of the optimal assignment.

    Notes:
        - Call build_cost_matrix to get C.
        - Call linear_sum_assignment(C) — returns (row_ind, col_ind).
          row_ind is always [0, 1, ..., N-1]; col_ind[i] is the slot for robot i.
        - Total cost = sum of C[row_ind, col_ind].
        - Return col_ind as the assignment array.
    """
    C = build_cost_matrix(positions,slots)
    row_ind, col_ind = linear_sum_assignment(C)
    total_cost = float(C[row_ind,col_ind].sum())
    return col_ind, total_cost
