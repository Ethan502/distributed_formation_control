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
    adjacency: np.ndarray | None = None,
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
    estimates = hull_history[round_idx]

    for i, vertices in enumerate(estimates):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        lw = 2.5 if i == highlight_robot else 1.0

        if len(vertices) == 1:
            ax.scatter(vertices[0, 0], vertices[0, 1], color=color, s=40, zorder=3)
        elif len(vertices) == 2:
            ax.plot(vertices[:, 0], vertices[:, 1], color=color, linewidth=lw, zorder=3)
        else:
            poly = plt.Polygon(vertices, closed=True,
                               facecolor=color, edgecolor=color,
                               alpha=0.25, linewidth=lw, zorder=2)
            ax.add_patch(poly)

    # Draw communication graph as directed arrows if adjacency is provided
    if adjacency is not None:
        N = len(robot_positions)
        for i in range(N):
            for j in range(N):
                if adjacency[i, j]:
                    ax.annotate(
                        "",
                        xy=robot_positions[j],
                        xytext=robot_positions[i],
                        arrowprops=dict(
                            arrowstyle='->', color='dimgrey',
                            lw=1.2, shrinkA=6, shrinkB=6,
                        ),
                        zorder=1,
                    )

    # Robot positions drawn on top of all hull polygons
    for i, pos in enumerate(robot_positions):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ax.scatter(pos[0], pos[1], color=color, s=60, zorder=4, edgecolors='black', linewidths=0.5)

    ax.set_aspect('equal')
    ax.set_title(f"Round {round_idx}: local hull estimates")


def draw_hull_convergence_summary(
    hull_history: list[list[np.ndarray]],
    robot_positions: np.ndarray,
    adjacency: np.ndarray | None = None,
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
    num_rounds = len(hull_history)

    # Use one row for <= 5 panels, two rows otherwise
    if num_rounds <= 5:
        nrows, ncols = 1, num_rounds
    else:
        ncols = (num_rounds + 1) // 2
        nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    # axes may be a single Axes, a 1-D array, or a 2-D array — flatten to iterate
    axes_flat = np.array(axes).flatten()

    for k, ax in enumerate(axes_flat[:num_rounds]):
        draw_local_hull_estimates(ax, hull_history, robot_positions, k,
                                  adjacency=adjacency if k == 0 else None)

    # Hide any unused subplot panels
    for ax in axes_flat[num_rounds:]:
        ax.set_visible(False)

    fig.suptitle("Convex hull consensus convergence", fontsize=13)
    fig.tight_layout()
    return fig
