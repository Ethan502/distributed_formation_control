"""Graph utility functions for distributed communication topology.

These functions operate on adjacency matrices representing the communication
graph among robots. They are dimension-agnostic (work for both 2D and 3D).
"""

import numpy as np
from scipy.sparse.csgraph import shortest_path
from collections import deque


def validate_adjacency_matrix(adj: np.ndarray) -> None:
    """Validate that an adjacency matrix is well-formed for an undirected graph.

    Checks:
        - Square shape
        - Binary entries (0 or 1)
        - Symmetric (undirected graph assumption)
        - Zero diagonal (no self-loops)
        - Connected graph

    Args:
        adj: (N, N) numpy array representing the adjacency matrix.

    Raises:
        ValueError: If any check fails, with a descriptive message.
    """
    if adj.ndim != 2:
        raise ValueError(f"Adjacency matrix must be 2D, got shape {adj.shape}.")
    n, m = adj.shape
    if n != m:
        raise ValueError(f"Adjacency matrix must be square, got shape ({n}, {m}).")
    if n == 0:
        raise ValueError("Adjacency matrix must have at least one node.")

    # Check binary entries
    unique_vals = np.unique(adj)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(
            f"Adjacency matrix must contain only 0s and 1s, "
            f"found values: {unique_vals}."
        )

    # Check zero diagonal
    if np.any(np.diag(adj) != 0):
        raise ValueError("Adjacency matrix diagonal must be all zeros (no self-loops).")

    # Check symmetry
    if not np.allclose(adj, adj.T):
        raise ValueError("Adjacency matrix must be symmetric for an undirected graph.")

    # Check connectivity
    if not is_connected(adj):
        raise ValueError("Communication graph is not connected.")


def get_neighbors(adj: np.ndarray, robot_id: int) -> list:
    """Return the list of neighbor indices for a given robot.

    Args:
        adj: (N, N) binary adjacency matrix.
        robot_id: Index of the robot whose neighbors to find.

    Returns:
        List of integer indices of neighboring robots.

    Raises:
        IndexError: If robot_id is out of range.
    """
    n = adj.shape[0]
    if robot_id < 0 or robot_id >= n:
        raise IndexError(
            f"robot_id {robot_id} out of range for graph with {n} nodes."
        )
    return list(np.where(adj[robot_id] > 0)[0])


def graph_diameter(adj: np.ndarray) -> int:
    """Compute the diameter of the graph (longest shortest path between any two nodes).

    Uses scipy's shortest_path with BFS (unweighted). The diameter determines
    the number of consensus rounds needed for information to propagate across
    the entire graph.

    Args:
        adj: (N, N) binary adjacency matrix of a connected graph.

    Returns:
        Integer diameter of the graph.

    Raises:
        ValueError: If the graph is disconnected (infinite diameter).
    """
    # shortest_path on unweighted graph: treat adjacency as sparse graph
    dist_matrix = shortest_path(adj, method='D', unweighted=True)

    max_dist = dist_matrix.max()
    if np.isinf(max_dist):
        raise ValueError("Graph is disconnected; diameter is infinite.")

    return int(max_dist)


def is_connected(adj: np.ndarray) -> bool:
    """Check whether the graph is connected using BFS from node 0.

    Args:
        adj: (N, N) binary adjacency matrix.

    Returns:
        True if the graph is connected, False otherwise.
    """
    n = adj.shape[0]
    if n <= 1:
        return True

    visited = set()
    queue = deque([0])
    visited.add(0)

    while queue:
        node = queue.popleft()
        for neighbor in range(n):
            if adj[node, neighbor] > 0 and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == n
