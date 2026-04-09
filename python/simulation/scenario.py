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
        [0.5,1.2],
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


def create_circular_choke_scenario() -> dict:
    """Create a 5-robot scenario where the team loops around a circular path.

    Layout:
        - 5 robots start in a loose cluster near the south waypoint (5, 1.5).
        - 6 waypoints evenly spaced counterclockwise on a circle centered at
          (5, 5) with radius 3.5.
        - Two rectangular obstacles form a choke point on the east side
          (centered at y=5), leaving a narrow gap of ~1.6 units.  The
          pentagon template's unit-scale width is ~2.0, so the optimizer
          must shrink the formation to roughly s ≤ 0.8 to squeeze through.

    Waypoints (counterclockwise, starting south):
        0: (5.0, 1.5)  — south  (start)
        1: (8.5, 3.0)  — southeast
        2: (8.5, 7.0)  — northeast  ← choke here
        3: (5.0, 8.5)  — north
        4: (1.5, 7.0)  — northwest
        5: (1.5, 3.0)  — southwest

    Communication graph:
        Ring topology — same as the default scenario.

    Returns:
        scenario: dict with keys:
            'agents'    : list of 5 Agent objects (zero initial velocity).
            'adjacency' : np.ndarray of shape (5, 5), binary ring graph.
            'obstacles' : list of 2 Rectangle objects (choke blocks).
            'waypoints' : list of 6 np.ndarray of shape (2,).

    Notes:
        - There is no 'goal' key; the caller cycles through 'waypoints'.
        - Workspace bounds for free-space computation should be (-2, -2, 12, 12).
    """
    # Robots start in a loose cluster near waypoint 0 (south side of circle)
    positions = np.array([
        [4.0, 1.8],
        [4.5, 1.2],
        [5.0, 1.5],
        [5.5, 1.2],
        [6.0, 1.8],
    ])

    agents = [Agent(id=i, position=positions[i], velocity=np.zeros(2))
              for i in range(5)]

    # Ring topology: 0–1–2–3–4–0
    adjacency = np.zeros((5, 5), dtype=float)
    for i in range(5):
        adjacency[i, (i + 1) % 5] = adjacency[(i + 1) % 5, i] = 1

    # Choke obstacles on the east side, leaving a ~1.6-unit gap at y ∈ [4.2, 5.8]
    obstacles = [
        Rectangle(x_min=7.5, y_min=5.8, x_max=11.0, y_max=9.5),   # upper block
        Rectangle(x_min=7.5, y_min=0.5, x_max=11.0, y_max=4.2),   # lower block
    ]

    # 6 waypoints counterclockwise on a circle (center=(5,5), radius=3.5)
    waypoints = [
        np.array([5.0, 1.5]),   # 0 south
        np.array([8.5, 3.0]),   # 1 southeast
        np.array([8.5, 7.0]),   # 2 northeast  (choke)
        np.array([5.0, 8.5]),   # 3 north
        np.array([1.5, 7.0]),   # 4 northwest
        np.array([1.5, 3.0]),   # 5 southwest
    ]

    return {'agents': agents, 'adjacency': adjacency,
            'obstacles': obstacles, 'waypoints': waypoints}
