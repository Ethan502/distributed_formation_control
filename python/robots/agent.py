"""Defines the Agent class representing a single robot in the team.

Each robot is a 2D double-integrator:

    dp/dt = v
    dv/dt = u

where p ∈ R² is position, v ∈ R² is velocity, and u ∈ R² is the
acceleration control input.

The Agent also carries local consensus state so it can participate in
the distributed algorithms (Phases 1–4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Agent:
    """A single 2D robot with double-integrator dynamics.

    Attributes:
        id:       Integer identifier for this robot (0-based).
        position: Current position [px, py], shape (2,).
        velocity: Current velocity [vx, vy], shape (2,).
        target:   Assigned formation slot position [px*, py*], shape (2,).
                  None until assignment is solved (Phase 6).

    Local consensus storage (one variable per distributed algorithm phase):
        hull_estimate:     List of 2D point arrays that this robot currently
                           believes are the convex hull vertices of all robot
                           positions.  Initialized to [own position].
                           Updated by run_convex_hull_consensus (Phase 1).

        direction_utility: Array of shape (K,) holding this robot's current
                           utility estimate for each of K candidate directions.
                           Initialized to the robot's own local utility.
                           Updated by run_direction_consensus (Phase 2).

        region:            This robot's local obstacle-free convex region in
                           2D position space (Phase 3) or position-time space
                           (Phase 4).  Represented as a half-space system
                           {x | A x <= b} once Phase 3 is implemented.
    """

    id: int
    position: np.ndarray            # shape (2,)
    velocity: np.ndarray            # shape (2,)
    sensing_range: float = 3.0      # radius within which obstacles are visible
    target: np.ndarray | None = None  # shape (2,), set during Phase 6

    # Local consensus variables — initialized lazily in each phase
    hull_estimate: list = field(default_factory=list)
    direction_utility: np.ndarray | None = None
    region: object = None           # becomes a polytope in Phase 3/4

    def __post_init__(self) -> None:
        """Validate array shapes and seed the hull estimate with own position."""
        if self.position.shape != (2,):
            raise ValueError(
                f"Agent {self.id}: position must have shape (2,), "
                f"got {self.position.shape}"
            )
        if self.velocity.shape != (2,):
            raise ValueError(
                f"Agent {self.id}: velocity must have shape (2,), "
                f"got {self.velocity.shape}"
            )

        # Algorithm 1 (paper p. 1085) initializes C_i(0) = {p_i}.
        # We set the hull estimate here so it is never empty.
        if not self.hull_estimate:
            self.hull_estimate = [self.position.copy()]

    @property
    def state(self) -> np.ndarray:
        """Return the full state vector [px, py, vx, vy], shape (4,)."""
        return np.concatenate([self.position, self.velocity])
