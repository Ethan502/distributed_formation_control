"""Visualization of the robot team state (Phase 0).

Draws robots, velocity arrows, communication graph edges, rectangular
obstacles, and the goal position on a matplotlib Axes.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes

from robots.agent import Agent
from geometry.rectangles import Rectangle

# One color per robot id (supports up to 10 robots)
ROBOT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def draw_team_state(
    ax: Axes,
    agents: list[Agent],
    adjacency: np.ndarray,
    goal: np.ndarray | None = None,
) -> None:
    """Draw the current team state on a matplotlib Axes.

    Draws (in order so elements stack cleanly):
        1. Communication graph edges as thin grey lines between neighbors.
        2. Velocity arrow for each robot (scaled for visibility).
        3. Robot position as a filled circle, colored by robot id.
        4. Robot id label next to each robot.
        5. Goal position as a star marker (if provided).

    Args:
        ax:        Matplotlib Axes on which to draw.
        agents:    List of N Agent objects.
        adjacency: Binary adjacency matrix, shape (N, N).
        goal:      Optional 2D goal position for the formation centroid,
                   shape (2,).  Drawn as a gold star if provided.

    Notes:
        - Draw each edge only once: iterate over pairs (i, j) with i < j
          and check adjacency[i, j].
        - Scale velocity arrows by a constant factor (e.g., 0.3) so they
          are visible at typical position scales without dominating the plot.
        - Use ax.set_aspect('equal') so distances look correct.
        - Add a brief title indicating the number of robots and graph edges.
    """
    n = len(agents)

    # 1. Communication graph edges — draw each pair (i, j) once (i < j).
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                ax.plot(
                    [agents[i].position[0], agents[j].position[0]],
                    [agents[i].position[1], agents[j].position[1]],
                    color='lightgrey', linewidth=1.2, zorder=1,
                )

    # 2. Velocity arrows — scale so they are visible but not overwhelming.
    arrow_scale = 0.3
    for agent in agents:
        if np.linalg.norm(agent.velocity) > 1e-6:
            ax.annotate(
                "",
                xy=agent.position + arrow_scale * agent.velocity,
                xytext=agent.position,
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.5),
                zorder=3,
            )

    # 3. Robot positions and id labels.
    legend_handles = []
    for agent in agents:
        color = ROBOT_COLORS[agent.id % len(ROBOT_COLORS)]
        sc = ax.scatter(
            agent.position[0], agent.position[1],
            color=color, s=80, zorder=4, label=f"Robot {agent.id}",
        )
        legend_handles.append(sc)
        ax.text(
            agent.position[0] + 0.1, agent.position[1] + 0.1,
            str(agent.id), fontsize=9, zorder=5,
        )

    # 4. Goal marker.
    if goal is not None:
        sc_goal = ax.scatter(
            goal[0], goal[1],
            marker='*', s=250, color='gold', edgecolors='darkorange',
            linewidths=0.8, zorder=4, label='Goal',
        )
        legend_handles.append(sc_goal)

    # 5. Legend entries for edges and obstacles.
    legend_handles.append(
        plt.Line2D([0], [0], color='lightgrey', linewidth=1.2, label='Comm. edge')
    )
    legend_handles.append(
        mpatches.Patch(facecolor='dimgrey', edgecolor='black', label='Obstacle')
    )

    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)
    n_edges = int(adjacency.sum()) // 2
    ax.set_aspect('equal')
    ax.set_title(f"{n} robots, {n_edges} edges")


def draw_direction_result(
    ax: Axes,
    agents: list[Agent],
    obstacles: list[Rectangle],
    centroid: np.ndarray,
    candidate_angles: np.ndarray,
    final_utility: np.ndarray,
    theta_star: float,
    max_range: float = 20.0,
    goal: np.ndarray | None = None,
) -> None:
    """Draw the outcome of the direction consensus (Phase 2).

    Shows:
        1. Obstacles and robot positions.
        2. All candidate directions as arrows from the hull centroid,
           colored by global utility (red = blocked, green = clear).
        3. The chosen direction theta_star as a bold black arrow.
        4. The hull centroid as a large dot.

    Args:
        ax:               Matplotlib Axes.
        agents:           List of Agent objects (for robot positions).
        obstacles:        List of Rectangle obstacles.
        centroid:         Hull centroid, shape (2,). Arrow origin.
        candidate_angles: Array of κ angles in radians, shape (κ,).
        final_utility:    Global utility vector after consensus, shape (κ,).
                          All robots share this same vector after d rounds.
        theta_star:       The selected direction angle in radians.
        max_range:        Used to normalise utility colours (0 → max_range).
    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    draw_rect_obstacles(ax, obstacles)

    # Robot positions
    for agent in agents:
        color = ROBOT_COLORS[agent.id % len(ROBOT_COLORS)]
        ax.scatter(agent.position[0], agent.position[1],
                   color=color, s=80, zorder=4, edgecolors='black', linewidths=0.5)
        ax.text(agent.position[0] + 0.1, agent.position[1] + 0.1,
                str(agent.id), fontsize=9, zorder=5)

    # Candidate direction arrows — coloured by utility
    norm = Normalize(vmin=0, vmax=max_range)
    cmap = cm.RdYlGn
    arrow_len = 1.2
    for theta, utility in zip(candidate_angles, final_utility):
        tip = centroid + arrow_len * np.array([np.cos(theta), np.sin(theta)])
        color = cmap(norm(utility))
        ax.annotate("", xy=tip, xytext=centroid,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=3)

    # Winning direction — bold black arrow, slightly longer
    tip_star = centroid + 1.8 * np.array([np.cos(theta_star), np.sin(theta_star)])
    ax.annotate("", xy=tip_star, xytext=centroid,
                arrowprops=dict(arrowstyle='->', color='black', lw=3.0), zorder=5)

    # Hull centroid marker
    ax.scatter(centroid[0], centroid[1], color='black', s=60,
               zorder=6, marker='D', label='Centroid')

    # Colourbar to show utility scale
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Ray clearance (m)', fraction=0.03, pad=0.04)

    # Goal marker
    if goal is not None:
        ax.scatter(goal[0], goal[1], marker='*', s=250, color='gold',
                   edgecolors='darkorange', linewidths=0.8, zorder=6, label='Goal')

    # Set axis limits to show the full scenario (robots + obstacles + goal) with margin
    all_x = [a.position[0] for a in agents]
    all_y = [a.position[1] for a in agents]
    for obs in obstacles:
        all_x += [obs.x_min, obs.x_max]
        all_y += [obs.y_min, obs.y_max]
    if goal is not None:
        all_x.append(goal[0])
        all_y.append(goal[1])
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect('equal')
    ax.set_title(f"Direction consensus — θ* = {np.degrees(theta_star):.1f}°")
    ax.legend(loc='upper right', fontsize=8)


def draw_rect_obstacles(ax: Axes, obstacles: list[Rectangle]) -> None:
    """Draw axis-aligned rectangular obstacles on the given axes.

    Each obstacle is drawn as a filled grey rectangle with a dark border.

    Args:
        ax:        Matplotlib Axes on which to draw.
        obstacles: List of Rectangle objects.

    Notes:
        - Use matplotlib.patches.Rectangle(xy, width, height, ...).
        - xy is the bottom-left corner: (rect.x_min, rect.y_min).
        - Width  = rect.x_max - rect.x_min.
        - Height = rect.y_max - rect.y_min.
        - Use a consistent style: facecolor='dimgrey', edgecolor='black',
          linewidth=1.5, zorder=2 (so robots draw on top).
    """
    for obstacle in obstacles:
        patch = mpatches.Rectangle(
            (obstacle.x_min, obstacle.y_min),
            obstacle.width, obstacle.height,
            facecolor='dimgrey', edgecolor='black', linewidth=1.5, zorder=2,
        )
        ax.add_patch(patch)
