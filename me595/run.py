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
from me595.value_planner import (                                 # noqa: E402
    compute_local_value_function,
    run_value_consensus,
    plan_direction,
)
from me595.grid_mdp import extract_direction, grid_to_world       # noqa: E402


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
    p.add_argument("--show-value-map", action="store_true", dest="show_value_map",
                   help=(
                       "Display the VI cost-to-go heatmap with a gradient-"
                       "descent path preview before the main loop starts.  "
                       "Only meaningful with --planner vi."
                   ))
    p.add_argument("--planner", choices=("greedy", "vi"), default="vi",
                   help=(
                       "Phase-2 direction selector.  'greedy' uses the "
                       "myopic min-consensus on ray clearance "
                       "(me595/geometry.py).  'vi' uses value iteration "
                       "on a grid MDP (me595/value_planner.py); see "
                       "Kochenderfer et al., Algorithms for Decision "
                       "Making, ch. 7.  Default: vi."
                   ))
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

    print(
        f"\nme595 closed-loop run  |  N={N} robots  |  "
        f"planner={args.planner}  |  diameter={num_rounds}"
    )
    print(f"Goal: {goal}    Workspace: {WORKSPACE_BOUNDS}")

    # ---- Phase 2 setup: build the value-iteration grid once if needed ----
    #
    # Obstacles are static so the cost-to-go field never changes during
    # the run.  We solve VI exactly once before the main loop and reuse
    # the agreed grid every iteration.  Each robot computes its own V_i
    # from its visible obstacles, then max-consensus reconciles them
    # over the communication graph (no-op here since sensing_range=20
    # gives every robot full visibility, but the structure matches the
    # paper's distributed pattern).
    agreed_grid = None
    if args.planner == "vi":
        local_grids = [
            compute_local_value_function(
                visible_obstacles(pos[i], obstacles, agents[i].sensing_range),
                workspace_bounds=WORKSPACE_BOUNDS,
                goal=goal,
                cell_size=0.25,
                inflation=0.15,
            )
            for i in range(N)
        ]
        agreed_grid = run_value_consensus(local_grids, adjacency, num_rounds)
        finite_cells = int(np.isfinite(agreed_grid.values).sum())
        print(
            f"VI: solved on {agreed_grid.rows}x{agreed_grid.cols} grid, "
            f"{finite_cells} reachable cells"
        )
        if args.show_value_map:
            _show_value_map(agreed_grid, merged_outline, goal, initial_pos)

    for iteration in range(args.max_iters):
        # ---- Phase 1: convex-hull consensus ----
        hull_history  = run_convex_hull_consensus(pos, adjacency, num_rounds)
        hull_vertices = hull_history[-1][0]
        centroid      = hull_vertices.mean(axis=0)

        if np.linalg.norm(centroid - goal) < GOAL_RADIUS:
            print(f"Goal reached BEFORE iteration {iteration + 1}.")
            goal_reached = True
            break

        # ---- Phase 2: preferred direction ----
        # Two implementations behind the same θ* interface:
        #   greedy: myopic min-consensus on per-direction ray clearance
        #           (Alonso-Mora et al. 2019, §3.2 / Alg. 2).
        #   vi:     gradient of the agreed cost-to-go field from value
        #           iteration on a grid MDP (Kochenderfer et al.,
        #           Algorithms for Decision Making, ch. 7).
        if args.planner == "greedy":
            candidate_angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            _, final_utilities = run_direction_consensus(
                pos, adjacency, obstacles, candidate_angles, num_rounds,
                centroid=centroid, goal=goal,
            )
            theta_star = float(select_preferred_direction(
                final_utilities[0], candidate_angles
            ))
        else:
            # 'vi' branch.  ``agreed_grid`` was built once before the
            # main loop; we just sample its gradient at the current
            # centroid.  Near the goal the 8-connected grid quantizes
            # angles into 22.5° steps, causing limit-cycle oscillation.
            # Since there are no obstacles to route around close to the
            # goal, we switch to the direct bearing for the final
            # approach (within 2× the goal radius).
            dist_to_goal = float(np.linalg.norm(goal - centroid))
            if dist_to_goal < 2.0 * GOAL_RADIUS:
                theta_star = float(
                    np.arctan2(goal[1] - centroid[1], goal[0] - centroid[0])
                )
            else:
                theta_star = float(plan_direction(agreed_grid, centroid))

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

        # ---- Phase 4b (VI only): no-retreat half-plane for the LP ----
        # The LP cap inside optimize_formation only bounds forward
        # advancement along theta*; it has no lower bound, so when the
        # agreed region is more open behind the centroid than ahead of
        # it (typical when VI says "go around the obstacle the long
        # way") the scale-maximizing LP places the entire pentagon
        # behind us.  We pin the LP to forward-only motion by adding a
        # half-plane that requires every slot to project at least as
        # far along theta* as the current hull centroid.
        #
        # Important: this extra half-plane is used **only** by
        # ``optimize_formation``.  We do NOT add it to ``A_agreed`` /
        # ``b_agreed`` because those also feed ``project_velocity`` in
        # Phase 7, where applying it to each individual robot would be
        # non-physical: a formation naturally spans both sides of its
        # centroid along theta*, so robots behind the centroid would
        # get their velocities clamped and the team would collapse.
        # Greedy uses the original region untouched.
        if args.planner == "vi":
            no_retreat_normal = -np.array(
                [np.cos(theta_star), np.sin(theta_star)]
            ).reshape(1, 2)
            no_retreat_offset = np.array([
                -float(np.cos(theta_star) * centroid[0]
                       + np.sin(theta_star) * centroid[1])
            ])
            A_lp = np.vstack([A_agreed, no_retreat_normal])
            b_lp = np.concatenate([b_agreed, no_retreat_offset])
        else:
            A_lp = A_agreed
            b_lp = b_agreed

        # ---- Phase 5: formation optimization ----
        direction_vec = np.array([np.cos(theta_star), np.sin(theta_star)])
        dist_to_goal_along_dir = float(direction_vec @ (goal - centroid))
        if args.planner == "greedy":
            # Original behavior: clamp tau to the goal-projected
            # distance so the formation never overshoots the goal.
            # Safe for greedy because its theta* is goal-weighted.
            effective_tau = min(tau, max(dist_to_goal_along_dir, 0.0))
        else:
            # VI may legitimately produce a theta* that points away
            # from the goal (going around an obstacle), so we cannot
            # use the goal-projected distance like the greedy branch
            # does — that would zero out tau and freeze the team.
            # Instead, cap the travel window by the straight-line
            # distance to the goal so the formation doesn't overshoot
            # on the final approach, with a floor of GOAL_RADIUS so it
            # can always make at least one final step.
            dist_to_goal = float(np.linalg.norm(goal - centroid))
            effective_tau = min(tau, max(dist_to_goal, GOAL_RADIUS))
        result = optimize_formation(
            A_lp, b_lp, TEMPLATES["pentagon"], theta_star,
            centroid=centroid, tau=effective_tau,
        )
        # The narrow gap below the triangle (~0.7 units after inflation)
        # can defeat the LP at full tau / pentagon scale.  Try
        # progressively smaller travel windows before giving up — this
        # lets the formation shrink to fit the squeeze instead of
        # aborting the run.
        if result is None:
            for tau_retry in (0.5 * effective_tau, 0.25 * effective_tau, 0.0):
                result = optimize_formation(
                    A_lp, b_lp, TEMPLATES["pentagon"], theta_star,
                    centroid=centroid, tau=tau_retry,
                )
                if result is not None:
                    print(
                        f"Iter {iteration + 1}: formation LP recovered "
                        f"with tau={tau_retry:.2f}."
                    )
                    break
        if result is None:
            print(
                f"Iter {iteration + 1}: no feasible formation even at "
                f"tau=0, skipping iteration."
            )
            continue
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
        iter_thetas=iter_thetas,
        n_iters=n_iters,
        animate=not args.no_animate,
        dt=dt,
        goal_reached=goal_reached,
        planner=args.planner,
    )


def _show_value_map(
    grid,
    merged_outline,
    goal: np.ndarray,
    initial_pos: np.ndarray,
) -> None:
    """Display the VI cost-to-go heatmap with gradient-descent path preview.

    The heatmap shows V*(s) for every reachable cell (blocked cells are
    transparent grey).  A red curve traces the gradient-descent path from
    the starting centroid to the goal — this is the "ideal" trajectory
    the formation centroid would follow if it were a single point.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    finite_mask = np.isfinite(grid.values)
    V_plot = np.where(finite_mask, grid.values, np.nan)

    im = ax.imshow(
        V_plot,
        origin="lower",
        cmap="viridis",
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
    )
    fig.colorbar(im, ax=ax, label="Cost-to-go V*(s)")

    # Overlay blocked cells in grey.
    ax.imshow(
        np.where(grid.blocked, 1.0, np.nan),
        origin="lower",
        cmap="gray",
        vmin=0.0, vmax=1.0,
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        alpha=0.6,
    )

    # Merged obstacle outline.
    draw_obstacle(ax, merged_outline)

    # Goal marker.
    ax.scatter(goal[0], goal[1], marker="*", s=300, color="gold",
               edgecolors="darkorange", linewidths=1.0, zorder=8, label="Goal")

    # Initial robot positions.
    for i in range(initial_pos.shape[0]):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ax.scatter(initial_pos[i, 0], initial_pos[i, 1],
                   color=color, s=80, zorder=7, edgecolors="black",
                   linewidths=0.5)

    centroid = initial_pos.mean(axis=0)
    ax.scatter(centroid[0], centroid[1], color="red", marker="x",
               s=120, linewidths=2.5, zorder=8, label="Start centroid")

    # Gradient-descent path preview: follow -grad(V) from the starting
    # centroid in small steps until we reach the goal or run out of steps.
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
    ax.plot(path[:, 0], path[:, 1], color="red", linewidth=2.0,
            linestyle="--", alpha=0.9, zorder=6, label="Gradient-descent path")

    # Direction arrow at starting centroid.
    theta_start = extract_direction(grid, centroid)
    arrow_len = 1.0
    ax.annotate(
        "", xy=(
            centroid[0] + arrow_len * np.cos(theta_start),
            centroid[1] + arrow_len * np.sin(theta_start),
        ),
        xytext=(centroid[0], centroid[1]),
        arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
        zorder=9,
    )

    ax.set_xlim(grid.x_min - 0.5, grid.x_max + 0.5)
    ax.set_ylim(grid.y_min - 0.5, grid.y_max + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Value function heatmap — gradient-descent path preview")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


def _render(
    merged_outline,
    obstacles,
    goal: np.ndarray,
    initial_pos: np.ndarray,
    iter_pos_histories: list[np.ndarray],
    iter_slots: list[np.ndarray],
    iter_regions: list[tuple[np.ndarray, np.ndarray]],
    iter_thetas: list[float],
    n_iters: int,
    animate: bool,
    dt: float,
    goal_reached: bool,
    planner: str = "vi",
) -> None:
    """Render either an animated or static view of the closed-loop run."""
    iter_colors = [cm.tab10(k % 10) for k in range(n_iters)]
    # Direction arrow color: blue for VI, red for greedy.
    arrow_color = "dodgerblue" if planner == "vi" else "crimson"

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
            # Direction arrow at the iteration centroid.
            iter_centroid = hist[0].mean(axis=0)
            theta_k = iter_thetas[k]
            arrow_len = 1.2
            ax.annotate(
                "", xy=(
                    iter_centroid[0] + arrow_len * np.cos(theta_k),
                    iter_centroid[1] + arrow_len * np.sin(theta_k),
                ),
                xytext=(iter_centroid[0], iter_centroid[1]),
                arrowprops=dict(arrowstyle="-|>", color=arrow_color,
                                lw=2.0, mutation_scale=12),
                zorder=9,
            )

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
    # Direction arrow at the centroid, updated per iteration.
    dir_arrow = ax.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=arrow_color,
                        lw=2.5, mutation_scale=14),
        zorder=9,
    )
    planner_label = "VI" if planner == "vi" else "Greedy"
    info_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9)

    current_iter = [-1]

    def init():
        for dot in robot_dots:
            dot.set_offsets(np.empty((0, 2)))
        for trail in trails:
            trail.set_data([], [])
        slot_dots.set_offsets(np.empty((0, 2)))
        dir_arrow.set_visible(False)
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
            # Update direction arrow for this iteration.
            c = hist[0].mean(axis=0)
            theta_k = iter_thetas[k]
            arrow_len = 1.2
            dir_arrow.set_visible(True)
            dir_arrow.xy = (
                c[0] + arrow_len * np.cos(theta_k),
                c[1] + arrow_len * np.sin(theta_k),
            )
            dir_arrow.set_position((c[0], c[1]))

        for i in range(N):
            robot_dots[i].set_offsets([p[i]])
            trail_x = hist[:t_in_iter + 1, i, 0]
            trail_y = hist[:t_in_iter + 1, i, 1]
            trails[i].set_data(trail_x, trail_y)

        info_text.set_text(
            f"Iteration {k + 1}/{n_iters}   step {t_in_iter}   "
            f"t = {t_in_iter * dt:.2f}s   "
            f"θ*={np.degrees(iter_thetas[k]):6.1f}°  [{planner_label}]"
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
