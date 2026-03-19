"""Utilities for the communication graph used by the robot team.

The graph G = (I, E) is represented as a binary N×N adjacency matrix A where
A[i, j] = 1 means robots i and j can communicate directly (paper Sect. 2.2).

Assumptions (matching the paper):
    - undirected graph  →  A is symmetric
    - no self-loops     →  diagonal of A is zero
    - connected graph   →  required for consensus convergence guarantees
                           (Propositions 1–2 in the paper)
"""

import numpy as np
import networkx as nx


def validate_adjacency_matrix(adjacency: np.ndarray) -> None:
    """Check that an adjacency matrix is valid for an undirected connected graph.

    Checks performed (in order):
        1. adjacency is a numpy array.
        2. adjacency is 2-dimensional.
        3. adjacency is square (N × N).
        4. all entries are 0 or 1 (binary matrix).
        5. the diagonal is all zeros (no self-loops).
        6. the matrix is symmetric (undirected edges).
        7. the graph is connected (required for consensus).

    Args:
        adjacency: Candidate adjacency matrix.

    Raises:
        ValueError: With a descriptive message if any check fails.

    Notes:
        - Check 6 uses np.allclose to tolerate minor floating-point noise.
        - Check 7 tests connectivity via the algebraic connectivity of the
          graph Laplacian: L = D - A, where D is the diagonal degree matrix.
          The graph is connected iff the second-smallest eigenvalue of L is > 0
          (the Fiedler value).
    """
    if not isinstance(adjacency,np.ndarray):
        raise ValueError("adjacency matrix must be a numpy array")
    if not adjacency.ndim == 2:
        raise ValueError("adjacency matrix must be 2 dimensional")
    if not adjacency.shape[0] == adjacency.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    if not np.all((adjacency == 0) | (adjacency == 1)):
        raise ValueError("Adjacency matrix must be binary")
    if not np.all(np.diag(adjacency) == 0):
        raise ValueError("Diagonal of adjacency matrix must be all 0s. (No self loops)")
    if not np.allclose(adjacency,adjacency.T):
        raise ValueError("Not sure what the issue is. But its with the adjacency matrix")
    L = np.diag(adjacency.sum(axis=1)) - adjacency
    eigenvalues = np.linalg.eigvalsh(L)
    if eigenvalues[1] <= 0:
        raise ValueError("Graph is not connected. Consensus requires a connected graph")


def get_neighbors(adjacency: np.ndarray, robot_id: int) -> list[int]:
    """Return the indices of all direct neighbors of a given robot.

    Args:
        adjacency: Binary adjacency matrix, shape (N, N).
        robot_id:  Index of the robot whose neighbors we want (0-based).

    Returns:
        neighbors: List of robot indices j such that adjacency[robot_id, j] == 1.
                   Does not include robot_id itself.

    Notes:
        - Equivalent to the neighbor set N_i in the paper (Sect. 2.2).
        - Robot j is a neighbor of i iff there is a direct edge (i, j) ∈ E.
    """
    neighbors = np.where(adjacency[robot_id] == 1)[0].tolist()
    return neighbors

def graph_diameter(adjacency: np.ndarray) -> int:
    """Compute the diameter of the communication graph.

    The diameter d is the maximum over all pairs (i, j) of the shortest
    path length between i and j.  It determines the minimum number of
    consensus rounds needed for convergence (paper Proposition 1).

    Args:
        adjacency: Binary adjacency matrix, shape (N, N).

    Returns:
        diameter: Integer diameter of the graph.

    Notes:
        - Assumes the graph is connected; call validate_adjacency_matrix first.
        - One approach: repeated matrix multiplication.
          Let R = A + I.  Then R^k[i,j] > 0 iff there is a path of length
          <= k between i and j.  The diameter is the smallest k such that
          all entries of R^k are positive.
        - Another approach: BFS from every node, then take the max
          eccentricity (max of all BFS depths).
    """
    validate_adjacency_matrix(adjacency)
    G = nx.from_numpy_array(adjacency)
    diameter = nx.diameter(G)
    return diameter

def is_connected(adjacency: np.ndarray) -> bool:
    """Check whether the communication graph is connected.

    A graph is connected if every pair of nodes is linked by at least one path.

    Args:
        adjacency: Binary adjacency matrix, shape (N, N).

    Returns:
        True if the graph is connected, False otherwise.

    Notes:
        - Use the Fiedler value (second-smallest eigenvalue of the Laplacian).
          A graph is connected iff Fiedler value > 0.
        - Alternatively, run BFS from node 0 and check that all nodes are visited.
        - This function does not raise; it simply returns a bool.
          validate_adjacency_matrix calls this and raises if False.
    """
    G = nx.from_numpy_array(adjacency)
    return nx.is_connected(G)
