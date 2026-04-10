"""Generate all figures for the README.

Run from repo root:
    ./env/bin/python docs/generate_figures.py
"""
from __future__ import annotations

import os
import sys

# Make python/ and repo root importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_PATH = os.path.join(_REPO_ROOT, "python")
if _PYTHON_PATH not in sys.path:
    sys.path.insert(0, _PYTHON_PATH)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np

from me595.scenario import create_triangle_scenario
from me595.geometry import visible_obstacles, compute_local_free_region, run_direction_consensus
from me595.draw_map import draw_obstacle
from me595.grid_mdp import create_grid, mark_obstacles, value_iteration, extract_direction, grid_to_world
from me595.value_planner import compute_local_value_function, run_value_consensus, plan_direction
from consensus.graph_utils import graph_diameter
from consensus.convex_hull import run_convex_hull_consensus
from consensus.preferred_direction import select_preferred_direction
from consensus.region_consensus import run_region_consensus
from formation.templates import TEMPLATES
from formation.optimizer import optimize_formation
from formation.assignment import solve_assignment
from robots.dynamics import step_double_integrator, compute_pd_acceleration, project_velocity
from me595.dynamics import project_velocity_obstacles
from plotting.draw_team import ROBOT_COLORS

FIGURES_DIR = os.path.join(_REPO_ROOT, "docs", "figures")
WORKSPACE_BOUNDS = (-2.0, -3.0, 14.0, 17.0)


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 1. Scenario map
# ---------------------------------------------------------------------------
def gen_scenario_map():
    print("Generating scenario map...")
    scenario = create_triangle_scenario()
    agents = scenario["agents"]
    adjacency = scenario["adjacency"]
    goal = scenario["goal"]
    merged = scenario["merged_outline"]

    fig, ax = plt.subplots(figsize=(9, 7))
    draw_obstacle(ax, merged)

    n = len(agents)
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                pi, pj = agents[i].position, agents[j].position
                ax.plot([pi[0], pj[0]], [pi[1], pj[1]],
                        color="lightgrey", linewidth=1.2, zorder=1)

    for agent in agents:
        color = ROBOT_COLORS[agent.id % len(ROBOT_COLORS)]
        p = agent.position
        ax.scatter(p[0], p[1], color=color, s=100, zorder=4,
                   edgecolors="black", linewidths=0.6)
        ax.text(p[0] + 0.2, p[1] + 0.2, str(agent.id), fontsize=10, zorder=5)

    ax.scatter(goal[0], goal[1], marker="*", s=400, color="gold",
               edgecolors="darkorange", linewidths=1.0, zorder=4, label="Goal")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
                   markersize=10, label="Robot"),
        plt.Line2D([0], [0], color="lightgrey", linewidth=1.2, label="Comm. edge"),
        mpatches.Patch(facecolor="dimgrey", edgecolor="black", label="Obstacle"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markeredgecolor="darkorange", markersize=16, label="Goal (12, 3)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    ax.set_xlim(-3, 15)
    ax.set_ylim(-4, 18)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Scenario: 5 robots, 30-60-90 triangle + rectangle obstacle")
    fig.tight_layout()
    save(fig, "scenario_map.png")


# ---------------------------------------------------------------------------
# 2. Value function heatmap with gradient-descent path
# ---------------------------------------------------------------------------
def gen_value_heatmap():
    print("Generating value function heatmap...")
    scenario = create_triangle_scenario()
    goal = scenario["goal"]
    merged = scenario["merged_outline"]
    agents = scenario["agents"]

    grid = create_grid(WORKSPACE_BOUNDS, cell_size=0.25, goal=goal)
    mark_obstacles(grid, scenario["obstacles"], inflation=0.15)
    n_iters = value_iteration(grid)

    finite_mask = np.isfinite(grid.values)
    V_plot = np.where(finite_mask, grid.values, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(V_plot, origin="lower", cmap="viridis",
                   extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max])
    fig.colorbar(im, ax=ax, label="Cost-to-go V*(s)")

    ax.imshow(np.where(grid.blocked, 1.0, np.nan), origin="lower", cmap="gray",
              vmin=0, vmax=1,
              extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max], alpha=0.6)
    draw_obstacle(ax, merged)

    ax.scatter(goal[0], goal[1], marker="*", s=400, color="gold",
               edgecolors="darkorange", linewidths=1.0, zorder=8, label="Goal")

    start_positions = np.array([a.position for a in agents])
    centroid = start_positions.mean(axis=0)
    for i, a in enumerate(agents):
        ax.scatter(a.position[0], a.position[1],
                   color=ROBOT_COLORS[i % len(ROBOT_COLORS)], s=80,
                   zorder=7, edgecolors="black", linewidths=0.5)
    ax.scatter(centroid[0], centroid[1], color="red", marker="x",
               s=140, linewidths=3, zorder=8, label="Start centroid")

    # Gradient-descent path preview.
    path = [centroid.copy()]
    step_size = grid.cell_size * 0.5
    goal_world = grid_to_world(grid, *grid.goal_ij)
    p = centroid.copy()
    for _ in range(2000):
        if np.linalg.norm(p - goal_world) < step_size:
            path.append(goal_world.copy())
            break
        theta = extract_direction(grid, p)
        p = p + step_size * np.array([np.cos(theta), np.sin(theta)])
        path.append(p.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], color="red", linewidth=2.5,
            linestyle="--", alpha=0.9, zorder=6, label="Gradient-descent path")

    theta_start = extract_direction(grid, centroid)
    arrow_len = 1.2
    ax.annotate("", xy=(centroid[0] + arrow_len * np.cos(theta_start),
                        centroid[1] + arrow_len * np.sin(theta_start)),
                xytext=(centroid[0], centroid[1]),
                arrowprops=dict(arrowstyle="->", color="red", lw=3), zorder=9)

    ax.set_xlim(grid.x_min - 0.5, grid.x_max + 0.5)
    ax.set_ylim(grid.y_min - 0.5, grid.y_max + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Value function V*(s) — {n_iters} Bellman sweeps, cell_size=0.25")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "value_heatmap.png")


# ---------------------------------------------------------------------------
# Shared simulation runner
# ---------------------------------------------------------------------------
def _run_sim(planner, max_iters=20):
    """Run the full pipeline and return trajectory data."""
    scenario = create_triangle_scenario()
    agents = scenario["agents"]
    adjacency = scenario["adjacency"]
    obstacles = scenario["obstacles"]
    goal = scenario["goal"]
    N = len(agents)

    pos = np.array([a.position for a in agents], dtype=float).copy()
    vel = np.zeros_like(pos)
    initial_pos = pos.copy()
    num_rounds = max(1, graph_diameter(adjacency))
    dt, kp, kd = 0.1, 2.0, 2.0 * float(np.sqrt(2.0))
    GOAL_RADIUS = 1.0

    agreed_grid = None
    if planner == "vi":
        local_grids = [
            compute_local_value_function(
                visible_obstacles(pos[i], obstacles, agents[i].sensing_range),
                workspace_bounds=WORKSPACE_BOUNDS, goal=goal,
                cell_size=0.25, inflation=0.15,
            ) for i in range(N)
        ]
        agreed_grid = run_value_consensus(local_grids, adjacency, num_rounds)

    centroids = []
    thetas = []
    all_pos = [initial_pos.copy()]
    goal_reached = False

    for iteration in range(max_iters):
        hull_history = run_convex_hull_consensus(pos, adjacency, num_rounds)
        hull_vertices = hull_history[-1][0]
        centroid = hull_vertices.mean(axis=0)
        centroids.append(centroid.copy())

        if np.linalg.norm(centroid - goal) < GOAL_RADIUS:
            goal_reached = True
            break

        if planner == "greedy":
            candidate_angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            _, final_utilities = run_direction_consensus(
                pos, adjacency, obstacles, candidate_angles, num_rounds,
                centroid=centroid, goal=goal)
            theta_star = float(select_preferred_direction(final_utilities[0], candidate_angles))
        else:
            dist_to_goal = float(np.linalg.norm(goal - centroid))
            if dist_to_goal < 2.0 * GOAL_RADIUS:
                theta_star = float(np.arctan2(goal[1] - centroid[1], goal[0] - centroid[0]))
            else:
                theta_star = float(plan_direction(agreed_grid, centroid))
        thetas.append(theta_star)

        tau = 3.0
        local_obs = [visible_obstacles(pos[i], obstacles, agents[i].sensing_range) for i in range(N)]
        local_regions = [
            compute_local_free_region(centroid, hull_vertices, local_obs[i],
                                      workspace_bounds=WORKSPACE_BOUNDS,
                                      theta_star=theta_star, tau=tau)
            for i in range(N)
        ]
        _, (A_agreed, b_agreed) = run_region_consensus(local_regions, adjacency, num_rounds)

        if planner == "vi":
            nr_normal = -np.array([np.cos(theta_star), np.sin(theta_star)]).reshape(1, 2)
            nr_offset = np.array([-float(np.cos(theta_star) * centroid[0] + np.sin(theta_star) * centroid[1])])
            A_lp = np.vstack([A_agreed, nr_normal])
            b_lp = np.concatenate([b_agreed, nr_offset])
        else:
            A_lp, b_lp = A_agreed, b_agreed

        direction_vec = np.array([np.cos(theta_star), np.sin(theta_star)])
        dist_to_goal_along_dir = float(direction_vec @ (goal - centroid))
        if planner == "greedy":
            effective_tau = min(tau, max(dist_to_goal_along_dir, 0.0))
        else:
            dist_to_goal = float(np.linalg.norm(goal - centroid))
            effective_tau = min(tau, max(dist_to_goal, GOAL_RADIUS))

        result = optimize_formation(A_lp, b_lp, TEMPLATES["pentagon"], theta_star,
                                    centroid=centroid, tau=effective_tau)
        if result is None:
            for tau_retry in (0.5 * effective_tau, 0.25 * effective_tau, 0.0):
                result = optimize_formation(A_lp, b_lp, TEMPLATES["pentagon"], theta_star,
                                            centroid=centroid, tau=tau_retry)
                if result is not None:
                    break
        if result is None:
            continue
        _, _, _, slots = result

        assignment, _ = solve_assignment(pos, slots)
        targets = slots[assignment]

        for _ in range(200):
            new_pos = np.zeros_like(pos)
            new_vel = np.zeros_like(vel)
            for i in range(N):
                u = compute_pd_acceleration(pos[i], vel[i], targets[i], kp, kd)
                v_cand = vel[i] + dt * u
                v_proj = project_velocity(v_cand, pos[i], A_agreed, b_agreed)
                v_proj = project_velocity_obstacles(v_proj, pos[i], obstacles)
                new_pos[i], new_vel[i] = step_double_integrator(pos[i], v_proj, np.zeros(2), dt)
            pos, vel = new_pos, new_vel
            if all(np.linalg.norm(pos[i] - targets[i]) < 0.05 for i in range(N)):
                break
        all_pos.append(pos.copy())

        if np.linalg.norm(pos.mean(axis=0) - goal) < GOAL_RADIUS:
            goal_reached = True
            break

    return {
        "centroids": centroids,
        "thetas": thetas,
        "all_pos": all_pos,
        "goal_reached": goal_reached,
        "initial_pos": initial_pos,
    }


# ---------------------------------------------------------------------------
# 3. VI trajectory
# ---------------------------------------------------------------------------
def gen_vi_trajectory():
    print("Generating VI trajectory...")
    scenario = create_triangle_scenario()
    merged = scenario["merged_outline"]
    goal = scenario["goal"]
    data = _run_sim("vi")

    fig, ax = plt.subplots(figsize=(10, 8))
    draw_obstacle(ax, merged)
    ax.scatter(goal[0], goal[1], marker="*", s=400, color="gold",
               edgecolors="darkorange", linewidths=1.0, zorder=8, label="Goal")

    # Robot trails.
    positions = np.array(data["all_pos"])  # (iters+1, N, 2)
    N = positions.shape[1]
    for i in range(N):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ax.plot(positions[:, i, 0], positions[:, i, 1],
                color=color, linewidth=1.8, alpha=0.7, zorder=4)
        ax.scatter(positions[0, i, 0], positions[0, i, 1],
                   color=color, s=80, zorder=6, edgecolors="black", linewidths=0.5)
        ax.scatter(positions[-1, i, 0], positions[-1, i, 1],
                   color=color, s=80, marker="D", zorder=6, edgecolors="black", linewidths=0.5)

    # Centroid path with direction arrows.
    centroids = np.array(data["centroids"])
    ax.plot(centroids[:, 0], centroids[:, 1], color="dodgerblue",
            linewidth=2.5, linestyle="-", alpha=0.9, zorder=5, label="Centroid path")
    for k, (c, theta) in enumerate(zip(data["centroids"], data["thetas"])):
        if k % 2 == 0:
            arrow_len = 0.8
            ax.annotate("", xy=(c[0] + arrow_len * np.cos(theta),
                                c[1] + arrow_len * np.sin(theta)),
                        xytext=(c[0], c[1]),
                        arrowprops=dict(arrowstyle="-|>", color="dodgerblue",
                                        lw=2.0, mutation_scale=12), zorder=9)

    ax.set_xlim(-3.5, 15)
    ax.set_ylim(-4, 18)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    n_iters = len(data["centroids"])
    status = "Goal reached" if data["goal_reached"] else "Max iters"
    final_dist = np.linalg.norm(positions[-1].mean(axis=0) - goal)
    ax.set_title(f"VI planner: {n_iters} iterations, {status} (d={final_dist:.2f})")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "vi_trajectory.png")


# ---------------------------------------------------------------------------
# 4. Greedy trajectory
# ---------------------------------------------------------------------------
def gen_greedy_trajectory():
    print("Generating greedy trajectory...")
    scenario = create_triangle_scenario()
    merged = scenario["merged_outline"]
    goal = scenario["goal"]
    data = _run_sim("greedy", max_iters=10)

    fig, ax = plt.subplots(figsize=(10, 8))
    draw_obstacle(ax, merged)
    ax.scatter(goal[0], goal[1], marker="*", s=400, color="gold",
               edgecolors="darkorange", linewidths=1.0, zorder=8, label="Goal")

    positions = np.array(data["all_pos"])
    N = positions.shape[1]
    for i in range(N):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ax.plot(positions[:, i, 0], positions[:, i, 1],
                color=color, linewidth=1.8, alpha=0.7, zorder=4)
        ax.scatter(positions[0, i, 0], positions[0, i, 1],
                   color=color, s=80, zorder=6, edgecolors="black", linewidths=0.5)
        ax.scatter(positions[-1, i, 0], positions[-1, i, 1],
                   color=color, s=80, marker="D", zorder=6, edgecolors="black", linewidths=0.5)

    centroids = np.array(data["centroids"])
    ax.plot(centroids[:, 0], centroids[:, 1], color="crimson",
            linewidth=2.5, linestyle="-", alpha=0.9, zorder=5, label="Centroid path")
    for k, (c, theta) in enumerate(zip(data["centroids"], data["thetas"])):
        arrow_len = 0.8
        ax.annotate("", xy=(c[0] + arrow_len * np.cos(theta),
                            c[1] + arrow_len * np.sin(theta)),
                    xytext=(c[0], c[1]),
                    arrowprops=dict(arrowstyle="-|>", color="crimson",
                                    lw=2.0, mutation_scale=12), zorder=9)

    ax.set_xlim(-3.5, 15)
    ax.set_ylim(-4, 18)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    n_iters = len(data["centroids"])
    final_dist = np.linalg.norm(positions[-1].mean(axis=0) - goal)
    ax.set_title(f"Greedy planner: {n_iters} iterations, stuck (d={final_dist:.2f})")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "greedy_trajectory.png")


# ---------------------------------------------------------------------------
# 5. Side-by-side comparison
# ---------------------------------------------------------------------------
def gen_comparison():
    print("Generating side-by-side comparison...")
    scenario = create_triangle_scenario()
    merged = scenario["merged_outline"]
    goal = scenario["goal"]

    vi_data = _run_sim("vi")
    greedy_data = _run_sim("greedy", max_iters=10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for ax, data, label, color, title in [
        (ax1, greedy_data, "Greedy", "crimson", "Greedy (myopic) -- STUCK"),
        (ax2, vi_data, "VI", "dodgerblue", "Value Iteration -- GOAL REACHED"),
    ]:
        draw_obstacle(ax, merged)
        ax.scatter(goal[0], goal[1], marker="*", s=350, color="gold",
                   edgecolors="darkorange", linewidths=1.0, zorder=8)

        positions = np.array(data["all_pos"])
        N = positions.shape[1]
        for i in range(N):
            c = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            ax.plot(positions[:, i, 0], positions[:, i, 1],
                    color=c, linewidth=1.5, alpha=0.6, zorder=4)
            ax.scatter(positions[0, i, 0], positions[0, i, 1],
                       color=c, s=60, zorder=6, edgecolors="black", linewidths=0.4)
            ax.scatter(positions[-1, i, 0], positions[-1, i, 1],
                       color=c, s=60, marker="D", zorder=6, edgecolors="black", linewidths=0.4)

        centroids = np.array(data["centroids"])
        ax.plot(centroids[:, 0], centroids[:, 1], color=color,
                linewidth=2.5, alpha=0.9, zorder=5)
        for k, (cen, theta) in enumerate(zip(data["centroids"], data["thetas"])):
            if k % 2 == 0 or label == "Greedy":
                ax.annotate("", xy=(cen[0] + 0.7 * np.cos(theta),
                                    cen[1] + 0.7 * np.sin(theta)),
                            xytext=(cen[0], cen[1]),
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=1.8, mutation_scale=10), zorder=9)

        ax.set_xlim(-3.5, 15)
        ax.set_ylim(-4, 18)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle("Before & After: Greedy vs. Value Iteration Direction Planning",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "comparison.png")


if __name__ == "__main__":
    gen_scenario_map()
    gen_value_heatmap()
    gen_vi_trajectory()
    gen_greedy_trajectory()
    gen_comparison()
    print("\nAll figures generated.")
