"""Reusable scenario presets for the simulation.

A scenario bundles together everything needed to start a demo:
    - a list of Agent objects (initial positions and velocities),
    - a binary adjacency matrix for the communication graph,
    - a list of Rectangle obstacles,
    - a goal position for the formation centroid.

This is the single place to change initial conditions so that all
main.py modes start from the same reproducible setup.
"""

from __future__ import annotations
import numpy as np

from robots.agent import Agent
from geometry.rectangles import Rectangle


def create_default_scenario() -> dict:
    """Create the default 5-robot scenario used throughout the project.

    Layout (approximate, in world coordinates):
        - 5 robots start in a loose cluster on the left side of the world,
          spread across roughly x ∈ [0, 2], y ∈ [0, 4].
        - Two rectangular obstacles sit in the middle of the world,
          creating a corridor the formation must navigate through.
        - The goal is to the right of the obstacles.

    Communication graph:
        Use a ring topology (each robot connected to its two ring-neighbors)
        so the graph is connected with diameter 2.  Feel free to change this
        once the scaffold is running.

        Example ring for 5 robots (0-indexed):
            0 — 1 — 2 — 3 — 4 — 0

    Returns:
        scenario: dict with keys:
            'agents'    : list of 5 Agent objects with zero initial velocity.
            'adjacency' : np.ndarray of shape (5, 5), binary.
            'obstacles' : list of Rectangle objects.
            'goal'      : np.ndarray of shape (2,), goal for formation centroid.

    Notes:
        - Initial velocities are zero so the first visualizations are clean.
        - The adjacency matrix must satisfy validate_adjacency_matrix (symmetric,
          binary, zero diagonal, connected).
        - Obstacle rectangles must not overlap with initial robot positions.
    """
    positions = np.array([
        [0.0,2.0],
        [0.5,0.5],
        [1.0,1.0],
        [2.0,3.0],
        [1.0,4.0]
    ])

    agents = [Agent(id=i,position=positions[i],velocity=np.zeros(2)) for i in range(5)]

    adjacency = np.zeros((5,5),dtype=float)
    for i in range(5):
        adjacency[i,(i+1)%5] = adjacency[(i+1)%5,i] = 1

    # Two obstacles creating a corridor centered around y=2.
    # Robots start at x ∈ [0, 2]; obstacles sit at x ∈ [3, 5]; goal is at x=7.
    obstacles = [
        Rectangle(x_min=3.0, y_min=3.0, x_max=5.0, y_max=6.0),  # upper block
        Rectangle(x_min=3.0, y_min=-1.0, x_max=5.0, y_max=1.0), # lower block
    ]

    goal = np.array([7.0, 2.0])

    return {'agents': agents, 'adjacency': adjacency,
            'obstacles': obstacles, 'goal': goal}
