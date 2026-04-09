"""Draw the me595 alternate scenario.

Run from the repo root:

    python -m me595.draw_map

Shows the 5-robot swarm, the ring communication graph, the 30-60-90
triangular obstacle, and the goal point.  Visual style mirrors
``python/plotting/draw_team.py`` (grey edges, colored robot dots, gold
star goal, dimgrey filled obstacle with a black border).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from me595.scenario import create_triangle_scenario
from me595.triangle import Triangle
from me595.rectangle import Rectangle
from me595.polygon import Polygon


# One color per robot id (matches the matplotlib default cycle, like the
# main project's plotting code).
ROBOT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def draw_triangle_obstacle(ax, triangle: Triangle) -> None:
    """Draw a single Triangle as a filled grey polygon with a dark border."""
    patch = mpatches.Polygon(
        triangle.vertices,
        closed=True,
        facecolor="dimgrey",
        edgecolor="black",
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(patch)


def draw_rectangle_obstacle(ax, rect: Rectangle) -> None:
    """Draw a single Rectangle in the same dimgrey style as the triangle."""
    patch = mpatches.Rectangle(
        (rect.x_min, rect.y_min),
        rect.width, rect.height,
        facecolor="dimgrey",
        edgecolor="black",
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(patch)


def draw_polygon_obstacle(ax, poly: Polygon) -> None:
    """Draw a generic Polygon obstacle as a single filled patch."""
    patch = mpatches.Polygon(
        poly.vertices,
        closed=True,
        facecolor="dimgrey",
        edgecolor="black",
        linewidth=1.5,
        zorder=2,
    )
    ax.add_patch(patch)


def draw_obstacle(ax, obstacle) -> None:
    """Dispatch on obstacle type so the scenario can mix shapes."""
    if isinstance(obstacle, Triangle):
        draw_triangle_obstacle(ax, obstacle)
    elif isinstance(obstacle, Rectangle):
        draw_rectangle_obstacle(ax, obstacle)
    elif isinstance(obstacle, Polygon):
        draw_polygon_obstacle(ax, obstacle)
    else:
        raise TypeError(f"Unknown obstacle type: {type(obstacle).__name__}")


def draw_scenario(scenario: dict) -> None:
    """Render the full me595 scenario in a single matplotlib figure."""
    agents = scenario["agents"]
    adjacency = scenario["adjacency"]
    goal = scenario["goal"]

    # Prefer the seamless merged outline for the visual; fall back to the
    # individual convex pieces if the scenario didn't supply one.
    obstacles_to_draw = (
        [scenario["merged_outline"]]
        if "merged_outline" in scenario
        else scenario["obstacles"]
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    # 1. Obstacles (drawn first so robots/edges sit on top).
    for obs in obstacles_to_draw:
        draw_obstacle(ax, obs)

    # 2. Communication graph edges — draw each pair only once.
    n = len(agents)
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                pi = agents[i].position
                pj = agents[j].position
                ax.plot(
                    [pi[0], pj[0]], [pi[1], pj[1]],
                    color="lightgrey", linewidth=1.2, zorder=1,
                )

    # 3. Robot positions and id labels.
    for agent in agents:
        color = ROBOT_COLORS[agent.id % len(ROBOT_COLORS)]
        p = agent.position
        ax.scatter(p[0], p[1], color=color, s=80, zorder=4,
                   edgecolors="black", linewidths=0.5)
        ax.text(p[0] + 0.15, p[1] + 0.15, str(agent.id),
                fontsize=9, zorder=5)

    # 4. Goal as a gold star.
    ax.scatter(goal[0], goal[1], marker="*", s=300, color="gold",
               edgecolors="darkorange", linewidths=0.8, zorder=4, label="Goal")

    # Legend.
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
                   markersize=9, label="Robot"),
        plt.Line2D([0], [0], color="lightgrey", linewidth=1.2, label="Comm. edge"),
        mpatches.Patch(facecolor="dimgrey", edgecolor="black",
                       label="Triangle obstacle"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markeredgecolor="darkorange", markersize=14, label="Goal"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    # Set sensible axis limits with a margin so the whole scene is visible.
    all_x = [a.position[0] for a in agents] + [goal[0]]
    all_y = [a.position[1] for a in agents] + [goal[1]]
    for obs in obstacles_to_draw:
        if isinstance(obs, (Triangle, Polygon)):
            all_x += list(obs.vertices[:, 0])
            all_y += list(obs.vertices[:, 1])
        elif isinstance(obs, Rectangle):
            all_x += [obs.x_min, obs.x_max]
            all_y += [obs.y_min, obs.y_max]
    margin = 1.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("me595 — 5 robots vs. 30-60-90 triangle obstacle")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    scenario = create_triangle_scenario()

    # Print a short summary, in the same spirit as run_scaffold in main.py.
    n_agents = len(scenario["agents"])
    n_edges = int(scenario["adjacency"].sum()) // 2
    print(f"Robots    : {n_agents}")
    print(f"Edges     : {n_edges}")
    print(f"Obstacles : {len(scenario['obstacles'])}  ({', '.join(repr(o) for o in scenario['obstacles'])})")
    print(f"Goal      : {scenario['goal']}")

    draw_scenario(scenario)
