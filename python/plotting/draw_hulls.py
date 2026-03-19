"""Visualization of convex hull consensus progress (Phase 1).

Draws each robot's local hull estimate as a transparent filled polygon,
showing how estimates grow and converge across consensus rounds.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

ROBOT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def draw_local_hull_estimates(
    ax: Axes,
    hull_history: list[list[np.ndarray]],
    robot_positions: np.ndarray,
    round_idx: int,
    highlight_robot: int | None = None,
) -> None:
    """Draw every robot's local hull estimate at a given consensus round.

    For each robot, its current hull estimate is drawn as a semi-transparent
    filled polygon, colored by robot id.

    Args:
        ax:              Matplotlib Axes on which to draw.
        hull_history:    Output of run_convex_hull_consensus.
                         hull_history[k][i] is robot i's hull after round k.
        robot_positions: Array of shape (N, 2) with robot positions (for dots).
        round_idx:       Which round's estimates to display (0 = initialization).
        highlight_robot: If given, draw this robot's hull with a thicker
                         border so it is easy to inspect.

    Notes:
        - If a robot's hull estimate has only 1 point, draw a dot.
        - If it has exactly 2 points, draw a line segment.
        - If it has ≥ 3 points, draw a filled polygon (plt.Polygon or ax.fill).
        - Draw robot positions as dots on top of the filled polygons.
        - Set a title like "Round k: hull estimates".
    """
    # TODO: retrieve estimates = hull_history[round_idx]
    # TODO: for i, vertices in enumerate(estimates):
    #   color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
    #   lw = 2.5 if i == highlight_robot else 1.0
    #   if len(vertices) == 1: ax.scatter(...)
    #   elif len(vertices) == 2: ax.plot(...)
    #   else: draw filled polygon with alpha=0.25
    # TODO: draw robot_positions as dots on top (zorder high)
    # TODO: ax.set_aspect('equal')
    # TODO: ax.set_title(f"Round {round_idx}: local hull estimates")
    raise NotImplementedError


def draw_hull_convergence_summary(
    hull_history: list[list[np.ndarray]],
    robot_positions: np.ndarray,
) -> plt.Figure:
    """Create a multi-panel figure showing hull estimates across all rounds.

    Each subplot corresponds to one consensus round and shows all robots'
    local hull estimates overlaid, making convergence visually obvious.

    Args:
        hull_history:    Output of run_convex_hull_consensus.
        robot_positions: Array of shape (N, 2) with robot positions.

    Returns:
        fig: The matplotlib Figure object (caller can call plt.show() or
             fig.savefig(...)).

    Notes:
        - Number of subplots = len(hull_history)  (rounds 0 through d).
        - Arrange subplots in a single row for up to 5 rounds; use two rows
          for more.
        - Add an overall figure title: "Convex hull consensus convergence".
        - Each subplot title should indicate the round number and whether
          all robots share the same hull at that round.
    """
    # TODO: num_rounds = len(hull_history)
    # TODO: choose subplot grid (1 row or 2 rows depending on num_rounds)
    # TODO: fig, axes = plt.subplots(nrows, ncols, ...)
    # TODO: for k, ax in enumerate(axes.flat):
    #         draw_local_hull_estimates(ax, hull_history, robot_positions, k)
    # TODO: fig.suptitle("Convex hull consensus convergence")
    # TODO: return fig
    raise NotImplementedError
