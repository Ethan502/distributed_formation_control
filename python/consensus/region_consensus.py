"""Distributed safe-region consensus — Algorithm 3 (Phase 4).

Section 3.3 of the paper: each robot i holds a local safe region
P_i = {x | A_i x ≤ b_i} computed in Phase 3.  The team runs a
distributed accumulation protocol so that every robot ends up with
the full intersection:

    P = P_1 ∩ P_2 ∩ ... ∩ P_N  =  {x | A x ≤ b}

where A and b are formed by stacking all robots' rows together.

Key insight: intersection of half-space polytopes is trivial in this
representation — just concatenate the (A, b) rows.  The distributed
part is making sure every robot learns every other robot's rows
using only neighbor-to-neighbor communication.

Protocol (one round):
    1. Each robot broadcasts its full current (A, b) to neighbors.
    2. Each robot stacks in every neighbor's (A, b) on top of its own.
    3. Duplicate rows are removed so the matrix doesn't grow unboundedly.

After d rounds (graph diameter), information from every robot has
propagated to every other robot through the graph.

Paper reference: Section 3.3, Algorithm 3, Proposition 2.
"""

from __future__ import annotations
import numpy as np


def deduplicate_halfspaces(
    A: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove near-duplicate rows from a half-space system (A, b).

    Two rows (a1, b1) and (a2, b2) are considered duplicates if
    ||a1 - a2|| < tol and |b1 - b2| < tol.

    Args:
        A:   Shape (M, 2) constraint normals.
        b:   Shape (M,)  constraint bounds.
        tol: Distance threshold for treating two rows as equal.

    Returns:
        A_unique: Deduplicated normals.
        b_unique: Deduplicated bounds.

    Notes:
        - Iterate through rows, keep a list of indices to keep.
        - For each new row, check if it matches any already-kept row.
        - If no match found, add its index to the keep list.
        - Return A[keep] and b[keep].
    """
    if len(A) == 0:
        return A,b
    
    keep = [0]
    for i in range(1,len(A)):
        is_dup = False
        for k in keep:
            if np.linalg.norm(A[i] - A[k]) < tol and abs(b[i] - b[k]) < tol:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)
    return A[keep], b[keep]


def run_region_consensus(
    local_regions: list[tuple[np.ndarray, np.ndarray]],
    adjacency: np.ndarray,
    num_rounds: int,
) -> tuple[list, tuple[np.ndarray, np.ndarray]]:
    """Run distributed consensus to intersect all robots' local safe regions.

    Each robot starts with only its own (A_i, b_i).  Every round, robots
    broadcast their full accumulated constraint set to neighbors and stack
    in what they receive.  After num_rounds, each robot holds the
    intersection of every region it has learned about.

    If num_rounds >= graph diameter, all robots agree on the full
    intersection P = ∩ P_i.

    Args:
        local_regions: List of N (A_i, b_i) tuples — one per robot,
                       produced by compute_local_free_region in Phase 3.
        adjacency:     Binary adjacency matrix, shape (N, N).
        num_rounds:    Number of consensus rounds to run.
                       Should be >= graph diameter for full convergence.

    Returns:
        history: list of length (num_rounds + 1).  history[k] is a list
                 of N (A, b) tuples — each robot's accumulated constraint
                 set at round k.
        final_region: (A, b) tuple for the agreed intersection.
                      After d rounds all robots hold the same region,
                      so return robot 0's final state.

    Notes:
        - Step 1: validate adjacency shape against N.
        - Step 2: initialize each robot's region as a copy of its own (A_i, b_i).
        - Step 3: save the round-0 state to history.
        - Step 4: for each round:
            a. Snapshot current regions into a broadcast list (so all robots
               send from the same round-start state, not mid-round updates).
            b. For each robot i, find its neighbors with np.where(adjacency[i] == 1).
            c. Stack in each neighbor j's (A, b) using np.vstack / np.concatenate.
            d. Deduplicate with deduplicate_halfspaces.
            e. Save the new state to history.
        - Step 5: return history and robot 0's final (A, b).
    """
    N = len(local_regions)
    if adjacency.shape != (N,N):
        raise ValueError(f"adjacency must have shape ({N},{N}), got {adjacency.shape}")
    regions = [(A.copy(), b.copy()) for A,b in local_regions]
    history = [list(regions)]
    for _ in range(num_rounds):
        broadcast = [(A.copy(), b.copy()) for A,b in regions]
        
        for i in range(N):
            neighbors = np.where(adjacency[i] == 1)[0]
            A_acc, b_acc = broadcast[i]

            for j in neighbors:
                A_j, b_j = broadcast[j]
                A_acc = np.vstack([A_acc,A_j])
                b_acc = np.concatenate([b_acc, b_j])

            A_acc, b_acc = deduplicate_halfspaces(A_acc,b_acc)
            regions[i] = (A_acc,b_acc)

        history.append(list(regions))

    return history, regions[0]
