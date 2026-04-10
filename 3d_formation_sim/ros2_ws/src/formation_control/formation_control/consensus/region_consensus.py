"""Region consensus -- merge half-space constraints.

Each robot maintains a local safe region represented as an intersection of
half-spaces: {x : A x <= b}. During consensus, robots exchange their
constraint sets and intersect them (by stacking rows). Deduplication
prevents unbounded growth of the constraint system.

This implements Algorithm 3 from the paper for a single robot's update step.
"""

import numpy as np


def merge_regions(my_A: np.ndarray, my_b: np.ndarray,
                  neighbor_As: list, neighbor_bs: list,
                  tol: float = 1e-6) -> tuple:
    """Merge this robot's safe region with neighbor regions.

    Implements one round of region consensus (Algorithm 3):
      P_i(k+1) = P_i(k) intersection Union_{j in N_i} P_j(k)

    The intersection of half-space systems is computed by stacking all
    constraints together. To prevent the constraint set from growing
    without bound over multiple rounds, near-duplicate constraints are
    removed.

    Deduplication procedure:
      1. Form the augmented row [a_i, b_i] for each constraint.
      2. Normalize each row to unit length.
      3. Remove rows that are within tolerance of an already-kept row.

    Args:
        my_A: (M, 3) constraint normals for this robot's current region.
        my_b: (M,) constraint bounds for this robot's current region.
        neighbor_As: List of (M_j, 3) constraint normal arrays from neighbors.
        neighbor_bs: List of (M_j,) constraint bound arrays from neighbors.
        tol: Tolerance for deduplication. Two normalized constraint rows
             closer than this (in Euclidean distance) are considered identical.

    Returns:
        (A, b) tuple:
            A: (K, 3) deduplicated constraint normals.
            b: (K,) deduplicated constraint bounds.
    """
    # Stack all constraints from self and neighbors
    all_A = [my_A]
    all_b = [my_b]
    for nA, nb in zip(neighbor_As, neighbor_bs):
        if nA is not None and len(nA) > 0:
            all_A.append(np.atleast_2d(nA))
            all_b.append(np.atleast_1d(nb))

    A_stacked = np.vstack(all_A)
    b_stacked = np.concatenate(all_b)

    # Deduplicate: normalize each [a_i | b_i] row, then remove near-duplicates
    M = A_stacked.shape[0]
    if M == 0:
        return A_stacked, b_stacked

    # Form augmented rows [a_i, b_i] and normalize
    augmented = np.column_stack([A_stacked, b_stacked])  # (M, 4)
    norms = np.linalg.norm(augmented, axis=1, keepdims=True)
    # Avoid division by zero for zero rows
    norms = np.where(norms < 1e-15, 1.0, norms)
    normalized = augmented / norms

    # Greedy deduplication: keep a row if it is not close to any already-kept row.
    # To handle sign ambiguity (a*x <= b is the same as -a*x <= -b after
    # normalization may flip sign), we compare both the row and its negation.
    keep_mask = np.ones(M, dtype=bool)
    for i in range(1, M):
        if not keep_mask[i]:
            continue
        for j in range(i):
            if not keep_mask[j]:
                continue
            diff_pos = np.linalg.norm(normalized[i] - normalized[j])
            diff_neg = np.linalg.norm(normalized[i] + normalized[j])
            if diff_pos < tol or diff_neg < tol:
                keep_mask[i] = False
                break

    return A_stacked[keep_mask], b_stacked[keep_mask]
