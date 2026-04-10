"""Robot-to-slot assignment using the Hungarian algorithm.

Solves the optimal assignment problem: given N robot positions and N
target formation slots, find the assignment that minimizes total
Euclidean travel distance.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def build_cost_matrix(positions: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Build the Euclidean distance cost matrix between robots and slots.

    Args:
        positions: (N, 3) current robot positions.
        slots: (N, 3) target slot positions.

    Returns:
        (N, N) cost matrix where C[i, j] = ||positions[i] - slots[j]||.
    """
    # Broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = positions[:, np.newaxis, :] - slots[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def solve_assignment(positions: np.ndarray, slots: np.ndarray) -> tuple:
    """Solve optimal robot-to-slot assignment using the Hungarian algorithm.

    Finds the permutation that minimizes total Euclidean distance from
    each robot to its assigned slot.

    Args:
        positions: (N, 3) current robot positions.
        slots: (N, 3) target slot positions.

    Returns:
        (assignment, cost) tuple:
            assignment: (N,) array where assignment[i] = slot index for robot i.
            cost: Total assignment cost (sum of distances).
    """
    C = build_cost_matrix(positions, slots)
    row_ind, col_ind = linear_sum_assignment(C)
    assignment = col_ind
    total_cost = C[row_ind, col_ind].sum()
    return assignment, total_cost
