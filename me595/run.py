"""Closed-loop distributed-formation runner for the me595 alternate map.

Mirrors ``run_closed_loop`` from ``python/main.py`` but:

    * uses the me595 scenario (5 robots vs. a 30-60-90 triangle merged
      with a tall rectangle stem),
    * uses the polygon-aware geometry / dynamics from ``me595/geometry.py``
      and ``me595/dynamics.py`` so triangle obstacles are handled,
    * draws the obstacle as the seamless merged polygon instead of the
      raw convex pieces,
    * always animates the trajectory.

Usage (from the repo root):

    ./env/bin/python -m me595.run
    ./env/bin/python -m me595.run --max-iters 10 --steps 200 --dt 0.05

Pipeline per planning iteration:
    1. distributed convex-hull consensus     (python/consensus/convex_hull.py)
    2. preferred-direction min-consensus     (me595/geometry.py)
    3. local convex free-region per robot    (me595/geometry.py)
    4. distributed region-intersection       (python/consensus/region_consensus.py)
    5. formation optimization (LP)           (python/formation/optimizer.py)
    6. robot-to-slot assignment              (python/formation/assignment.py)
    7. PD control + half-plane / polygon velocity projection
       (python/robots/dynamics.py + me595/dynamics.py)

Iterations repeat until the team centroid arrives within ``goal_radius``
of the goal, or ``--max-iters`` is exhausted.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib.cm as cm

# Make python/ importable so we can reuse the consensus / formation /
# dynamics pieces that don't care which obstacle shapes we use.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_PATH = os.path.join(_REPO_ROOT, "python")
if _PYTHON_PATH not in sys.path:
    sys.path.insert(0, _PYTHON_PATH)

# --- python/ pipeline pieces (obstacle-shape-agnostic) ---
from consensus.graph_utils import graph_diameter                 # noqa: E402
from consensus.convex_hull import run_convex_hull_consensus       # noqa: E402
from consensus.preferred_direction import select_preferred_direction  # noqa: E402
from consensus.region_consensus import run_region_consensus       # noqa: E402
from formation.templates import TEMPLATES                         # noqa: E402
from formation.optimizer import optimize_formation                # noqa: E402
from formation.assignment import solve_assignment                 # noqa: E402
from robots.dynamics import (                                     # noqa: E402
    step_double_integrator,
    compute_pd_acceleration,
    project_velocity,
)
from plotting.draw_team import ROBOT_COLORS                       # noqa: E402
from plotting.draw_obstacles import draw_free_region              # noqa: E402

# --- me595 polygon-aware pieces ---
from me595.scenario import create_triangle_scenario               # noqa: E402
from me595.geometry import (                                       # noqa: E402
    compute_local_free_region,
    visible_obstacles,
    run_direction_consensus,
)
from me595.dynamics import project_velocity_obstacles             # noqa: E402
from me595.draw_map import draw_obstacle                          # noqa: E402


# Workspace bounds large enough to contain the swarm, the merged obstacle
# (x in [1, 8], y in [-2, ~15.66]), and the goal at (12, 3), with some
# breathing room for free-region construction.
WORKSPACE_BOUNDS = (-2.0, -3.0, 14.0, 17.0)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="me595 closed-loop runner")
    p.add_argument("--dt", type=float, default=0.1,
                   help="Simulation timestep in seconds (default 0.1).")
    p.add_argument("--steps", type=int, default=200,
                   help="Max sim steps per planning iteration (default 200).")
    p.add_argument("--max-iters", type=int, default=20, dest="max_iters",
                   help="Max planning iterations (default 20).")
    p.add_argument("--rounds", type=int, default=None,
                   help="Override consensus rounds (default = graph diameter).")
    p.add_argument("--no-animate", action="store_true",
                   help="Skip the animation and show a static summary plot.")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    scenario       = create_triangle_scenario()
    agents         = scenario["agents"]
    adjacency      = scenario["adjacency"]
    obstacles      = scenario["obstacles"]        # convex pieces (triangle, rectangle)
    merged_outline = scenario["merged_outline"]   # union polygon for plotting only
    goal           = scenario["goal"]

    N = len(agents)
    pos = np.array([a.position for a in agents], dtype=float).copy()
    vel = np.zeros_like(pos)
    initial_pos = pos.copy()

    num_rounds = args.rounds if args.rounds else max(1, graph_diameter(adjacency))
    dt = args.dt

    # Critically-damped PD: Kd = 2 * sqrt(Kp).
    kp, kd = 2.0, 2.0 * float(np.sqrt(2.0))

    TARGET_TOL  = 0.05  # robots within this distance of slot targets → converged
    GOAL_RADIUS = 1.0   # centroid within this distance of goal → success

    # Per-iteration records used by both the static and animated plots.
    iter_pos_histories: list[np.ndarray] = []                  # each (T, N, 2)
    iter_slots:         list[np.ndarray] = []
    iter_regions:       list[tuple[np.ndarray, np.ndarray]] = []
    iter_assignments:   list[np.ndarray] = []
    iter_thetas:        list[float]      = []

    goal_reached = False

    print(f"\nme595 closed-loop run  |  N={N} robots  |  diameter={num_rounds}")
    print(f"Goal: {goal}    Workspace: {WORKSPACE_BOUNDS}")

    for iteration in range(args.max_iters):
        # ---- Phase 1: convex-hull consensus ----
        hull_history  = run_convex_hull_consensus(pos, adjacency, num_rounds)
        hull_vertices = hull_history[-1][0]
        centroid      = hull_vertices.mean(axis=0)

        if np.linalg.norm(centroid - goal) < GOAL_RADIUS:
            print(f"Goal reached BEFORE iteration {iteration + 1}.")
            goal_reached = True
            break

        # ---- Phase 2: preferred-direction min-consensus ----
        candidate_angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        _, final_utilities = run_direction_consensus(
            pos, adjacency, obstacles, candidate_angles, num_rounds,
            centroid=centroid, goal=goal,
        )
        theta_star = float(select_preferred_direction(
            final_utilities[0], candidate_angles
        ))

        # ---- Phase 3 + 4: local free regions + agreed intersection ----
        tau = 3.0
        local_obs = [
            visible_obstacles(pos[i], obstacles, agents[i].sensing_range)
            for i in range(N)
        ]
        local_regions = [
            compute_local_free_region(
                centroid,
                hull_vertices,
                local_obs[i],
                workspace_bounds=WORKSPACE_BOUNDS,
                theta_star=theta_star,
                tau=tau,
            )
            for i in range(N)
        ]
        _, (A_agreed, b_agreed) = run_region_consensus(
            local_regions, adjacency, num_rounds
        )

        # ---- Phase 5: formation optimization ----
        direction_vec = np.array([np.cos(theta_star), np.sin(theta_star)])
        dist_to_goal_along_dir = float(direction_vec @ (goal - centroid))
        effective_tau = min(tau, max(dist_to_goal_along_dir, 0.0))
        result = optimize_formation(
            A_agreed, b_agreed, TEMPLATES["pentagon"], theta_star,
            centroid=centroid, tau=effective_tau,
        )
        if result is None:
            print(f"Iter {iteration + 1}: no feasible formation, stopping.")
            break
        _, _, _, slots = result

        # ---- Phase 6: robot-to-slot assignment ----
        assignment, _ = solve_assignment(pos, slots)
        targets = slots[assignment]

        print(
            f"Iter {iteration + 1:2d} | θ*={np.degrees(theta_star):6.1f}° | "
            f"centroid=({centroid[0]:.2f},{centroid[1]:.2f}) | "
            f"slots x∈[{slots[:, 0].min():.2f},{slots[:, 0].max():.2f}]"
        )

        # ---- Phase 7: closed-loop double-integrator until converged ----
        iter_hist = [pos.copy()]
        for _ in range(args.steps):
            new_pos = np.zeros_like(pos)
            new_vel = np.zeros_like(vel)
            for i in range(N):
                u = compute_pd_acceleration(pos[i], vel[i], targets[i], kp, kd)
                v_cand = vel[i] + dt * u
                # Half-plane projection (region constraints).
                v_proj = project_velocity(v_cand, pos[i], A_agreed, b_agreed)
                # Polygon-aware obstacle face projection.
                v_proj = project_velocity_obstacles(v_proj, pos[i], obstacles)
                new_pos[i], new_vel[i] = step_double_integrator(
                    pos[i], v_proj, np.zeros(2), dt
                )
            pos = new_pos
            vel = new_vel
            iter_hist.append(pos.copy())
            if all(
                np.linalg.norm(pos[i] - targets[i]) < TARGET_TOL
                for i in range(N)
            ):
                break

        iter_pos_histories.append(np.array(iter_hist))
        iter_slots.append(slots)
        iter_regions.append((A_agreed, b_agreed))
        iter_assignments.append(assignment)
        iter_thetas.append(theta_star)

        if np.linalg.norm(pos.mean(axis=0) - goal) < GOAL_RADIUS:
            print(f"Goal reached after iteration {iteration + 1}.")
            goal_reached = True
            break

    if not goal_reached:
        print(f"Max iters ({args.max_iters}) reached without arriving at goal.")

    n_iters = len(iter_pos_histories)
    if n_iters == 0:
        print("No iterations ran — nothing to plot.")
        return

    print(f"\nIterations run  : {n_iters}")
    print(f"Goal reached    : {goal_reached}")
    print(f"Final centroid  : {pos.mean(axis=0)}")
    print(f"Distance to goal: {np.linalg.norm(pos.mean(axis=0) - goal):.3f}")

    _render(
        merged_outline=merged_outline,
        obstacles=obstacles,
        goal=goal,
        initial_pos=initial_pos,
        iter_pos_histories=iter_pos_histories,
        iter_slots=iter_slots,
        iter_regions=iter_regions,
        n_iters=n_iters,
        animate=not args.no_animate,
        dt=dt,
        goal_reached=goal_reached,
    )


def _render(
    merged_outline,
    obstacles,
    goal: np.ndarray,
    initial_pos: np.ndarray,
    iter_pos_histories: list[np.ndarray],
    iter_slots: list[np.ndarray],
    iter_regions: list[tuple[np.ndarray, np.ndarray]],
    n_iters: int,
    animate: bool,
    dt: float,
    goal_reached: bool,
) -> None:
    """Render either an animated or static view of the closed-loop run."""
    iter_colors = [cm.tab10(k % 10) for k in range(n_iters)]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Obstacle: prefer the seamless merged outline.
    draw_obstacle(ax, merged_outline)

    # Goal marker.
    ax.scatter(goal[0], goal[1], marker="*", s=300, color="gold",
               edgecolors="darkorange", linewidths=1.0, zorder=8, label="Goal")

    # Axis bounds: include all visited positions, the obstacle, and the goal.
    all_pos = np.vstack([h.reshape(-1, 2) for h in iter_pos_histories])
    all_x = list(all_pos[:, 0]) + list(merged_outline.vertices[:, 0]) + [goal[0]]
    all_y = list(all_pos[:, 1]) + list(merged_outline.vertices[:, 1]) + [goal[1]]
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect("equal")

    if not animate:
        # Static: draw all trajectories colored by iteration.
        for k, hist in enumerate(iter_pos_histories):
            c = iter_colors[k]
            for i in range(hist.shape[1]):
                ax.plot(hist[:, i, 0], hist[:, i, 1],
                        color=c, linewidth=1.5, alpha=0.8, zorder=4)
            A_k, b_k = iter_regions[k]
            draw_free_region(ax, A_k, b_k, color=c, alpha=0.07)
            for s in iter_slots[k]:
                ax.scatter(s[0], s[1], marker="*", s=120, color=c,
                           edgecolors="black", linewidths=0.4,
                           zorder=6, alpha=0.7)

        # Initial robot positions.
        for i in range(initial_pos.shape[0]):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            ax.scatter(initial_pos[i, 0], initial_pos[i, 1],
                       color=color, s=100, zorder=7,
                       edgecolors="black", linewidths=0.8)
            ax.text(initial_pos[i, 0] + 0.1, initial_pos[i, 1] + 0.1,
                    str(i), fontsize=9, zorder=8)

        # Final robot positions.
        final_pos = iter_pos_histories[-1][-1]
        for i in range(final_pos.shape[0]):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            ax.scatter(final_pos[i, 0], final_pos[i, 1],
                       color=color, s=100, marker="D", zorder=7,
                       edgecolors="black", linewidths=0.8)

        ax.set_title(
            f"me595 closed-loop: {n_iters} iter(s)  "
            f"({'goal reached' if goal_reached else 'max iters'})"
        )
        plt.tight_layout()
        plt.show()
        return

    # ---- Animated path ----
    N = iter_pos_histories[0].shape[1]

    # Flatten all frames so a single FuncAnimation can scrub through them.
    frames_pos:  list[np.ndarray] = []
    frames_iter: list[int]        = []
    for k, hist in enumerate(iter_pos_histories):
        for t in range(len(hist)):
            frames_pos.append(hist[t])
            frames_iter.append(k)
    total_frames = len(frames_pos)

    region_patch_holder: list = [None]

    def _draw_region(k: int) -> None:
        if region_patch_holder[0] is not None:
            try:
                region_patch_holder[0].remove()
            except Exception:
                pass
            region_patch_holder[0] = None
        A_k, b_k = iter_regions[k]
        # ``draw_free_region`` adds a patch to the axes; we don't get a
        # handle back, so we leave it stacked (with low alpha each pass
        # is barely visible anyway).
        draw_free_region(ax, A_k, b_k, color=iter_colors[k], alpha=0.15)

    robot_dots = [
        ax.scatter([], [], color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                   s=80, zorder=6, edgecolors="black", linewidths=0.5)
        for i in range(N)
    ]
    trails = [
        ax.plot([], [], color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                linewidth=1.2, alpha=0.6, zorder=4)[0]
        for i in range(N)
    ]
    slot_dots = ax.scatter([], [], marker="*", s=180, color="black",
                           edgecolors="black", linewidths=0.4, zorder=7)
    info_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9)

    current_iter = [-1]

    def init():
        for dot in robot_dots:
            dot.set_offsets(np.empty((0, 2)))
        for trail in trails:
            trail.set_data([], [])
        slot_dots.set_offsets(np.empty((0, 2)))
        info_text.set_text("")
        return robot_dots + trails + [slot_dots, info_text]

    def update(frame: int):
        k = frames_iter[frame]
        p = frames_pos[frame]
        hist = iter_pos_histories[k]
        # Step within iteration k.
        t_in_iter = frame - sum(len(iter_pos_histories[j]) for j in range(k))

        if k != current_iter[0]:
            current_iter[0] = k
            _draw_region(k)
            slot_dots.set_offsets(iter_slots[k])
            slot_dots.set_color(iter_colors[k])

        for i in range(N):
            robot_dots[i].set_offsets([p[i]])
            trail_x = hist[:t_in_iter + 1, i, 0]
            trail_y = hist[:t_in_iter + 1, i, 1]
            trails[i].set_data(trail_x, trail_y)

        info_text.set_text(
            f"Iteration {k + 1}/{n_iters}   step {t_in_iter}   "
            f"t = {t_in_iter * dt:.2f}s"
        )
        return robot_dots + trails + [slot_dots, info_text]

    ax.set_title("me595 closed-loop — distributed formation control on triangle map")
    # Stash the animation on the figure so the GC doesn't collect it before
    # the window finishes drawing.
    fig._me595_anim = animation.FuncAnimation(
        fig, update, frames=total_frames,
        init_func=init, interval=40, blit=False, repeat=False,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
