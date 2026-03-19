"""Defines the Team class: a collection of robots sharing a communication graph.

The Team is the main data structure passed between phases.  It holds:
    - a list of Agent objects (one per robot),
    - the binary adjacency matrix encoding who can communicate with whom.

Paper reference: Section 2.2 (communication graph G = (I, E)).
"""

from __future__ import annotations

import numpy as np

from robots.agent import Agent
from consensus.graph_utils import validate_adjacency_matrix


class Team:
    """A collection of robots sharing a communication graph.

    Attributes:
        agents:    List of Agent objects, length N.
        adjacency: Binary adjacency matrix, shape (N, N).
                   Entry (i, j) = 1 means robots i and j communicate directly.
                   Must be symmetric with zero diagonal (undirected, no self-loops).
    """

    def __init__(self, agents: list[Agent], adjacency: np.ndarray) -> None:
        """Create a team.

        Args:
            agents:    List of N Agent objects.
            adjacency: Binary adjacency matrix of shape (N, N).

        Raises:
            ValueError: If len(agents) does not match adjacency.shape[0], or
                        if the adjacency matrix is invalid (see validate_adjacency_matrix).
        """
        if len(agents) != adjacency.shape[0]:
            raise ValueError(
                f"Number of agents ({len(agents)}) must match "
                f"adjacency matrix size ({adjacency.shape[0]})."
            )
        validate_adjacency_matrix(adjacency)
        self.agents = agents
        self.adjacency = adjacency

    @property
    def positions(self) -> np.ndarray:
        """Return all robot positions as an array of shape (N, 2)."""
        return np.stack([a.position for a in self.agents])

    @property
    def velocities(self) -> np.ndarray:
        """Return all robot velocities as an array of shape (N, 2)."""
        return np.stack([a.velocity for a in self.agents])

    def __len__(self) -> int:
        return len(self.agents)

    def __repr__(self) -> str:
        return f"Team(n={len(self.agents)} robots)"
