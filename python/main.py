"""Entry point for the distributed formation control simulation.

Each --mode corresponds to one paper phase.  Run a mode to load a scenario,
execute one algorithmic chunk, and see something visible.

Usage examples:
    python main.py --mode scaffold
    python main.py --mode hull
    python main.py --mode hull --rounds 3
    python main.py --mode direction
    python main.py --mode local-free-space
    python main.py --mode region-consensus
    python main.py --mode formation
    python main.py --mode assignment
    python main.py --mode control --steps 200 --animate
    python main.py --mode full-demo
"""

import argparse
import sys


# ---------------------------------------------------------------------------
# Phase 0 — Simulation scaffold
# ---------------------------------------------------------------------------

def run_scaffold(args: argparse.Namespace) -> None:
    """Display the initial scenario: robots, graph, obstacles, and goal.

    What this mode shows:
        - 5 robot positions as colored dots.
        - Velocity arrows (all zero at start, so arrows are invisible — that
          is correct and expected).
        - Communication graph edges between neighbors.
        - Rectangular obstacles.
        - Goal position as a star.

    Args:
        args: Parsed command-line arguments.

    Notes:
        - Import create_default_scenario from simulation.scenario.
        - Import draw_team_state and draw_rect_obstacles from plotting.draw_team.
        - Print a short summary: number of robots, number of edges, graph
          diameter, number of obstacles.
        - Call plt.show() at the end.
    """
    from simulation.scenario import create_default_scenario

    from plotting.draw_team import draw_team_state, draw_rect_obstacles
    from consensus.graph_utils import graph_diameter
    import matplotlib.pyplot as plt

    scenario = create_default_scenario()
    agents    = scenario['agents']
    adjacency = scenario['adjacency']
    obstacles = scenario['obstacles']
    goal      = scenario['goal']

    fig, ax = plt.subplots(figsize=(8, 6))
    draw_rect_obstacles(ax, obstacles)
    draw_team_state(ax, agents, adjacency, goal=goal)

    d = graph_diameter(adjacency)
    n_edges = int(adjacency.sum()) // 2
    print(f"Robots : {len(agents)}")
    print(f"Edges  : {n_edges}")
    print(f"Diameter: {d}")
    print(f"Obstacles: {len(obstacles)}")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Phase 1 — Distributed convex hull consensus
# ---------------------------------------------------------------------------

def run_hull(args: argparse.Namespace) -> None:
    """Run distributed convex hull consensus and visualize convergence.

    What this mode shows:
        - A multi-panel figure: one panel per consensus round.
        - In each panel, every robot's local hull estimate is drawn as a
          semi-transparent filled polygon.
        - By the final panel all polygons should be identical (= global hull).

    Args:
        args: Parsed command-line arguments.
            args.rounds overrides the default number of rounds (graph diameter).

    Notes:
        - Import run_convex_hull_consensus from consensus.convex_hull.
        - Import draw_hull_convergence_summary from plotting.draw_hulls.
        - Import graph_diameter from consensus.graph_utils.
        - Print which round convergence was detected.
    """
    from simulation.scenario import create_default_scenario
    from consensus.graph_utils import graph_diameter
    from consensus.convex_hull import run_convex_hull_consensus, did_hulls_converge
    from plotting.draw_hulls import draw_hull_convergence_summary
    import matplotlib.pyplot as plt
    import numpy as np

    scenario  = create_default_scenario()
    agents    = scenario['agents']
    adjacency = scenario['adjacency']

    positions = np.array([a.position for a in agents])

    num_rounds = args.rounds if args.rounds else graph_diameter(adjacency)

    history = run_convex_hull_consensus(positions, adjacency, num_rounds)
    converged, k = did_hulls_converge(history)
    print(f"Rounds run : {num_rounds}")
    print(f"Converged  : {converged} at round {k}")

    fig = draw_hull_convergence_summary(history, positions, adjacency=adjacency)
    plt.show()


# ---------------------------------------------------------------------------
# Remaining modes — stubs for later phases
# ---------------------------------------------------------------------------

def run_direction(args: argparse.Namespace) -> None:
    """Phase 2: preferred direction of motion consensus.

    What this mode shows:
        - Robot positions and obstacles.
        - All candidate directions as arrows from the hull centroid,
          coloured red (blocked) to green (clear) by global utility.
        - The agreed team direction θ* as a bold black arrow.
    """
    from simulation.scenario import create_default_scenario
    from consensus.graph_utils import graph_diameter
    from consensus.convex_hull import run_convex_hull_consensus
    from consensus.preferred_direction import (
        run_direction_consensus, select_preferred_direction
    )
    from plotting.draw_team import draw_direction_result
    import matplotlib.pyplot as plt
    import numpy as np

    scenario  = create_default_scenario()
    agents    = scenario['agents']
    adjacency = scenario['adjacency']
    obstacles = scenario['obstacles']
    goal      = scenario['goal']

    positions  = np.array([a.position for a in agents])
    num_rounds = args.rounds if args.rounds else graph_diameter(adjacency)

    # Phase 1 result: hull centroid is the arrow origin
    hull_history = run_convex_hull_consensus(positions, adjacency, num_rounds)
    hull_vertices = hull_history[-1][0]          # all robots agree after d rounds
    centroid = hull_vertices.mean(axis=0)

    # 16 candidate angles evenly spaced around the full circle
    candidate_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)

    _, final_utilities = run_direction_consensus(
        positions, adjacency, obstacles, candidate_angles, num_rounds,
        centroid=centroid, goal=goal,
    )

    # All rows of final_utilities are identical after consensus — use robot 0's
    theta_star = select_preferred_direction(final_utilities[0], candidate_angles)

    print(f"Rounds run : {num_rounds}")
    print(f"θ*         : {np.degrees(theta_star):.1f}°")

    fig, ax = plt.subplots(figsize=(9, 7))
    draw_direction_result(ax, agents, obstacles, centroid,
                          candidate_angles, final_utilities[0], theta_star,
                          goal=goal)
    plt.tight_layout()
    plt.show()


def run_local_free_space(args: argparse.Namespace) -> None:
    """Phase 3: each robot computes a local obstacle-free convex region.

    What this mode shows:
        - All 5 robots' individual safe regions as filled polygons.
        - Obstacle boundaries overlaid.
        - Hull centroid and preferred direction arrow.
        - Robot positions.
    """
    from simulation.scenario import create_default_scenario
    from consensus.graph_utils import graph_diameter
    from consensus.convex_hull import run_convex_hull_consensus
    from consensus.preferred_direction import (
        run_direction_consensus, select_preferred_direction
    )
    from geometry.free_space import compute_local_free_region, visible_obstacles
    from plotting.draw_team import draw_rect_obstacles, ROBOT_COLORS
    from plotting.draw_obstacles import draw_free_region, draw_halfspace_boundaries
    import matplotlib.pyplot as plt
    import numpy as np

    REGION_COLORS = ['steelblue', 'tomato', 'seagreen', 'mediumpurple', 'darkorange']

    scenario  = create_default_scenario()
    agents    = scenario['agents']
    adjacency = scenario['adjacency']
    obstacles = scenario['obstacles']
    goal      = scenario['goal']

    positions  = np.array([a.position for a in agents])
    num_rounds = args.rounds if args.rounds else graph_diameter(adjacency)

    # Phase 1 — hull centroid
    hull_history  = run_convex_hull_consensus(positions, adjacency, num_rounds)
    hull_vertices = hull_history[-1][0]
    centroid      = hull_vertices.mean(axis=0)

    # Phase 2 — preferred direction
    candidate_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    _, final_utilities = run_direction_consensus(
        positions, adjacency, obstacles, candidate_angles, num_rounds,
        centroid=centroid, goal=goal,
    )
    theta_star = select_preferred_direction(final_utilities[0], candidate_angles)

    # Phase 3 — each robot's local free region (based on what it can see)
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_rect_obstacles(ax, obstacles)

    tau = 3.0  # look-ahead distance for computing χ = c + τ·[cos θ*, sin θ*]
    local_obstacle_lists = [
        visible_obstacles(agent.position, obstacles, agent.sensing_range)
        for agent in agents
    ]

    for i, agent in enumerate(agents):
        A, b = compute_local_free_region(centroid, hull_vertices, local_obstacle_lists[i],
                                         theta_star=theta_star, tau=tau)
        draw_free_region(ax, A, b,
                         color=REGION_COLORS[i % len(REGION_COLORS)],
                         alpha=0.15,
                         label=f"Robot {i} region")

    # Robot positions, sensing circles, and id labels
    for agent in agents:
        color = ROBOT_COLORS[agent.id % len(ROBOT_COLORS)]
        # Sensing range circle — dashed, same color as robot
        circle = plt.Circle(agent.position, agent.sensing_range,
                            color=color, fill=False, linestyle='--',
                            linewidth=1.0, alpha=0.5, zorder=3)
        ax.add_patch(circle)
        ax.scatter(agent.position[0], agent.position[1],
                   color=color, s=80, zorder=5, edgecolors='black', linewidths=0.5)
        ax.text(agent.position[0] + 0.1, agent.position[1] + 0.1,
                str(agent.id), fontsize=9, zorder=6)

    # Centroid and preferred direction arrow
    ax.scatter(centroid[0], centroid[1], color='black', s=80,
               marker='D', zorder=6, label='Centroid')
    tip = centroid + 1.5 * np.array([np.cos(theta_star), np.sin(theta_star)])
    ax.annotate("", xy=tip, xytext=centroid,
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5), zorder=6)

    # Axis limits to show full scenario
    all_x = list(positions[:, 0]) + [o.x_min for o in obstacles] + [o.x_max for o in obstacles]
    all_y = list(positions[:, 1]) + [o.y_min for o in obstacles] + [o.y_max for o in obstacles]
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Draw half-space boundary lines for agent 0 only
    A0, b0 = compute_local_free_region(centroid, hull_vertices, local_obstacle_lists[0],
                                        theta_star=theta_star, tau=tau)
    draw_halfspace_boundaries(ax, A0, b0)

    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f"Phase 3: local obstacle-free regions  (θ* = {np.degrees(theta_star):.1f}°)")
    plt.tight_layout()

    # --- Per-robot figure: one panel per robot ---
    fig2, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    # Consistent axis limits across all panels
    margin = 1.0
    x_lo = min(all_x) - margin
    x_hi = max(all_x) + margin
    y_lo = min(all_y) - margin
    y_hi = max(all_y) + margin

    for i, agent in enumerate(agents):
        panel = axes[i]
        color = ROBOT_COLORS[agent.id % len(ROBOT_COLORS)]
        region_color = REGION_COLORS[i % len(REGION_COLORS)]

        # Obstacles
        draw_rect_obstacles(panel, obstacles)

        # Free region for this robot
        A, b = compute_local_free_region(centroid, hull_vertices, local_obstacle_lists[i],
                                         theta_star=theta_star, tau=tau)
        draw_free_region(panel, A, b, color=region_color, alpha=0.3)

        # Half-space boundary lines
        panel.set_xlim(x_lo, x_hi)
        panel.set_ylim(y_lo, y_hi)
        draw_halfspace_boundaries(panel, A, b)

        # Sensing circle
        circle = plt.Circle(agent.position, agent.sensing_range,
                             color=color, fill=False, linestyle='--',
                             linewidth=1.5, alpha=0.7, zorder=4)
        panel.add_patch(circle)

        # Other robots as small grey dots
        for other in agents:
            if other.id != agent.id:
                panel.scatter(other.position[0], other.position[1],
                              color='lightgrey', s=40, zorder=4,
                              edgecolors='grey', linewidths=0.5)

        # This robot as a large colored dot
        panel.scatter(agent.position[0], agent.position[1],
                      color=color, s=100, zorder=5,
                      edgecolors='black', linewidths=0.8)
        panel.text(agent.position[0] + 0.1, agent.position[1] + 0.1,
                   str(agent.id), fontsize=9, zorder=6)

        n_seen = len(local_obstacle_lists[i])
        panel.set_aspect('equal')
        panel.set_title(f"Robot {i} — sees {n_seen} obstacle(s)", fontsize=10)

    # Hide the unused 6th panel
    axes[-1].set_visible(False)

    fig2.suptitle(f"Phase 3: per-robot local free regions  (θ* = {np.degrees(theta_star):.1f}°)",
                  fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"Rounds     : {num_rounds}")
    print(f"θ*         : {np.degrees(theta_star):.1f}°")
    print(f"Centroid   : {centroid}")
    for i, agent in enumerate(agents):
        seen = local_obstacle_lists[i]
        A, b = compute_local_free_region(centroid, hull_vertices, seen, theta_star=theta_star, tau=tau)
        print(f"Robot {i}: sees {len(seen)} obstacle(s), {A.shape[0]} half-space constraints")


def run_region_consensus(args: argparse.Namespace) -> None:
    """Phase 4: distributed consensus to intersect local safe regions.

    What this mode shows:
        - Left panel:  each robot's individual local region (from Phase 3),
                       colored by robot id.
        - Right panel: the single agreed intersection region after consensus,
                       with obstacle and robot positions overlaid.

    Notes:
        - Run phases 1, 2, and 3 first (same pattern as run_local_free_space).
        - Import run_region_consensus from consensus.region_consensus.
        - The consensus returns a single agreed (A, b) — draw it with draw_free_region.
        - Print the number of constraints in the final agreed region per robot.
    """
    from simulation.scenario import create_default_scenario
    from consensus.graph_utils import graph_diameter
    from consensus.convex_hull import run_convex_hull_consensus
    from consensus.preferred_direction import (
        run_direction_consensus, select_preferred_direction
    )
    from geometry.free_space import compute_local_free_region, visible_obstacles
    from consensus.region_consensus import run_region_consensus as _run_consensus
    from plotting.draw_team import draw_rect_obstacles, ROBOT_COLORS
    from plotting.draw_obstacles import draw_free_region, draw_halfspace_boundaries
    import matplotlib.pyplot as plt
    import numpy as np

    REGION_COLORS = ['steelblue', 'tomato', 'seagreen', 'mediumpurple', 'darkorange']

    scenario  = create_default_scenario()
    agents    = scenario['agents']
    adjacency = scenario['adjacency']
    obstacles = scenario['obstacles']
    goal      = scenario['goal']

    positions  = np.array([a.position for a in agents])
    num_rounds = args.rounds if args.rounds else graph_diameter(adjacency)

    # Phase 1 — hull centroid
    hull_history  = run_convex_hull_consensus(positions, adjacency, num_rounds)
    hull_vertices = hull_history[-1][0]
    centroid      = hull_vertices.mean(axis=0)

    # Phase 2 — preferred direction
    candidate_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    _, final_utilities = run_direction_consensus(
        positions, adjacency, obstacles, candidate_angles, num_rounds,
        centroid=centroid, goal=goal,
    )
    theta_star = select_preferred_direction(final_utilities[0], candidate_angles)

    # Phase 3 — per-robot local free regions
    tau = 3.0
    local_obstacle_lists = [
        visible_obstacles(agent.position, obstacles, agent.sensing_range)
        for agent in agents
    ]
    local_regions = [
        compute_local_free_region(centroid, hull_vertices, local_obstacle_lists[i],
                                  theta_star=theta_star, tau=tau)
        for i in range(len(agents))
    ]

    # Phase 4 — consensus to intersect all local regions
    _, (A_agreed, b_agreed) = _run_consensus(local_regions, adjacency, num_rounds)

    # --- Plot ---
    fig, (ax_ind, ax_agreed) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: individual regions
    draw_rect_obstacles(ax_ind, obstacles)
    for i, (A, b) in enumerate(local_regions):
        draw_free_region(ax_ind, A, b,
                         color=REGION_COLORS[i % len(REGION_COLORS)],
                         alpha=0.2, label=f"Robot {i}")
    for agent in agents:
        color = ROBOT_COLORS[agent.id % len(ROBOT_COLORS)]
        ax_ind.scatter(agent.position[0], agent.position[1],
                       color=color, s=80, zorder=5,
                       edgecolors='black', linewidths=0.5)
        ax_ind.text(agent.position[0] + 0.1, agent.position[1] + 0.1,
                    str(agent.id), fontsize=9, zorder=6)
    ax_ind.set_aspect('equal')
    ax_ind.legend(loc='upper right', fontsize=8)
    ax_ind.set_title("Individual local regions (Phase 3)")

    # Right: agreed intersection
    draw_rect_obstacles(ax_agreed, obstacles)
    draw_free_region(ax_agreed, A_agreed, b_agreed,
                     color='mediumseagreen', alpha=0.35, label='Agreed region P')
    draw_halfspace_boundaries(ax_agreed, A_agreed, b_agreed)
    for agent in agents:
        color = ROBOT_COLORS[agent.id % len(ROBOT_COLORS)]
        ax_agreed.scatter(agent.position[0], agent.position[1],
                          color=color, s=80, zorder=5,
                          edgecolors='black', linewidths=0.5)
        ax_agreed.text(agent.position[0] + 0.1, agent.position[1] + 0.1,
                       str(agent.id), fontsize=9, zorder=6)
    ax_agreed.scatter(centroid[0], centroid[1], color='black', s=80,
                      marker='D', zorder=6, label='Centroid')
    tip = centroid + 1.5 * np.array([np.cos(theta_star), np.sin(theta_star)])
    ax_agreed.annotate("", xy=tip, xytext=centroid,
                       arrowprops=dict(arrowstyle='->', color='black', lw=2.5), zorder=6)
    ax_agreed.set_aspect('equal')
    ax_agreed.legend(loc='upper right', fontsize=8)
    ax_agreed.set_title(f"Agreed intersection P  (θ* = {np.degrees(theta_star):.1f}°)")

    # Shared axis limits
    all_x = list(positions[:, 0]) + [o.x_min for o in obstacles] + [o.x_max for o in obstacles]
    all_y = list(positions[:, 1]) + [o.y_min for o in obstacles] + [o.y_max for o in obstacles]
    margin = 1.0
    for ax in (ax_ind, ax_agreed):
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    plt.suptitle("Phase 4: distributed region consensus", fontsize=13)
    plt.tight_layout()
    plt.show()

    print(f"Rounds          : {num_rounds}")
    print(f"θ*              : {np.degrees(theta_star):.1f}°")
    print(f"Agreed region   : {A_agreed.shape[0]} half-space constraints")
    for i in range(len(agents)):
        print(f"  Robot {i}: {local_regions[i][0].shape[0]} constraints  "
              f"(sees {len(local_obstacle_lists[i])} obstacle(s))")


def run_formation(args: argparse.Namespace) -> None:
    """Phase 5: formation template optimization inside the agreed region.

    What this mode shows:
        - The agreed safe region as a filled polygon.
        - The best-fit formation template overlaid inside it.
        - Target slot positions as colored markers.
        - Lines connecting slot positions to show the formation shape.
    """
    from simulation.scenario import create_default_scenario
    from consensus.graph_utils import graph_diameter
    from consensus.convex_hull import run_convex_hull_consensus
    from consensus.preferred_direction import (
        run_direction_consensus, select_preferred_direction
    )
    from geometry.free_space import compute_local_free_region, visible_obstacles
    from consensus.region_consensus import run_region_consensus as _run_consensus
    from formation.templates import TEMPLATES
    from formation.optimizer import optimize_formation
    from plotting.draw_team import draw_rect_obstacles, ROBOT_COLORS
    from plotting.draw_obstacles import draw_free_region, draw_halfspace_boundaries
    import matplotlib.pyplot as plt
    import numpy as np

    scenario  = create_default_scenario()
    agents    = scenario['agents']
    adjacency = scenario['adjacency']
    obstacles = scenario['obstacles']
    goal      = scenario['goal']

    positions  = np.array([a.position for a in agents])
    num_rounds = args.rounds if args.rounds else graph_diameter(adjacency)

    # Phase 1 — hull centroid
    hull_history  = run_convex_hull_consensus(positions, adjacency, num_rounds)
    hull_vertices = hull_history[-1][0]
    centroid      = hull_vertices.mean(axis=0)

    # Phase 2 — preferred direction
    candidate_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    _, final_utilities = run_direction_consensus(
        positions, adjacency, obstacles, candidate_angles, num_rounds,
        centroid=centroid, goal=goal,
    )
    theta_star = select_preferred_direction(final_utilities[0], candidate_angles)

    # Phase 3 — per-robot local free regions
    tau = 3.0
    local_obstacle_lists = [
        visible_obstacles(agent.position, obstacles, agent.sensing_range)
        for agent in agents
    ]
    local_regions = [
        compute_local_free_region(centroid, hull_vertices, local_obstacle_lists[i],
                                  theta_star=theta_star, tau=tau)
        for i in range(len(agents))
    ]

    # Phase 4 — agreed intersection
    _, (A_agreed, b_agreed) = _run_consensus(local_regions, adjacency, num_rounds)

    # Phase 5 — optimize formation template inside agreed region
    template_name = 'vee'
    template = TEMPLATES[template_name]
    result = optimize_formation(A_agreed, b_agreed, template, theta_star)

    if result is None:
        print("No feasible formation found inside the agreed region.")
        return

    best_center, best_scale, best_angle, slots = result

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_rect_obstacles(ax, obstacles)

    # Agreed region
    draw_free_region(ax, A_agreed, b_agreed,
                     color='mediumseagreen', alpha=0.25, label='Agreed region P')

    # Formation slot positions — colored by robot id, connected by lines
    slot_colors = [ROBOT_COLORS[i % len(ROBOT_COLORS)] for i in range(len(slots))]
    for i, slot in enumerate(slots):
        ax.scatter(slot[0], slot[1], color=slot_colors[i], s=120,
                   zorder=6, edgecolors='black', linewidths=0.8,
                   marker='*', label=f"Slot {i}")
    # Close the formation polygon
    loop = np.vstack([slots, slots[0]])
    ax.plot(loop[:, 0], loop[:, 1], 'k--', linewidth=1.0, alpha=0.5, zorder=5)

    # Formation center
    ax.scatter(best_center[0], best_center[1], color='black', s=60,
               marker='D', zorder=7, label='Formation center')

    # Current robot positions
    for agent in agents:
        color = ROBOT_COLORS[agent.id % len(ROBOT_COLORS)]
        ax.scatter(agent.position[0], agent.position[1],
                   color=color, s=80, zorder=5,
                   edgecolors='black', linewidths=0.5, alpha=0.5)
        ax.text(agent.position[0] + 0.1, agent.position[1] + 0.1,
                str(agent.id), fontsize=9, zorder=6)

    # Preferred direction arrow from centroid
    tip = centroid + 1.5 * np.array([np.cos(theta_star), np.sin(theta_star)])
    ax.annotate("", xy=tip, xytext=centroid,
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5), zorder=6)

    # Goal marker
    ax.scatter(goal[0], goal[1], marker='*', s=250, color='gold',
               edgecolors='darkorange', linewidths=0.8, zorder=6, label='Goal')

    all_x = list(positions[:, 0]) + [o.x_min for o in obstacles] + [o.x_max for o in obstacles]
    all_y = list(positions[:, 1]) + [o.y_min for o in obstacles] + [o.y_max for o in obstacles]
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(
        f"Phase 5: '{template_name}' formation  "
        f"(scale={best_scale:.2f}, φ={np.degrees(best_angle):.1f}°)"
    )
    plt.tight_layout()
    plt.show()

    print(f"Template        : {template_name}")
    print(f"Optimal scale   : {best_scale:.3f}")
    print(f"Optimal angle   : {np.degrees(best_angle):.1f}°")
    print(f"Formation center: {best_center}")
    print("Slot positions:")
    for i, slot in enumerate(slots):
        print(f"  Slot {i}: {slot}")


def run_assignment(args: argparse.Namespace) -> None:
    """Phase 6: assign robots to formation slots.

    What this mode shows:
        - Agreed safe region.
        - Target slot positions as star markers.
        - Current robot positions as colored dots.
        - A line from each robot to its assigned slot.
        - Total assignment cost printed to console.
    """
    from simulation.scenario import create_default_scenario
    from consensus.graph_utils import graph_diameter
    from consensus.convex_hull import run_convex_hull_consensus
    from consensus.preferred_direction import (
        run_direction_consensus, select_preferred_direction
    )
    from geometry.free_space import compute_local_free_region, visible_obstacles
    from consensus.region_consensus import run_region_consensus as _run_consensus
    from formation.templates import TEMPLATES
    from formation.optimizer import optimize_formation
    from formation.assignment import solve_assignment
    from plotting.draw_team import draw_rect_obstacles, ROBOT_COLORS
    from plotting.draw_obstacles import draw_free_region
    import matplotlib.pyplot as plt
    import numpy as np

    scenario  = create_default_scenario()
    agents    = scenario['agents']
    adjacency = scenario['adjacency']
    obstacles = scenario['obstacles']
    goal      = scenario['goal']

    positions  = np.array([a.position for a in agents])
    num_rounds = args.rounds if args.rounds else graph_diameter(adjacency)

    # Phases 1–4
    hull_history  = run_convex_hull_consensus(positions, adjacency, num_rounds)
    hull_vertices = hull_history[-1][0]
    centroid      = hull_vertices.mean(axis=0)

    candidate_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    _, final_utilities = run_direction_consensus(
        positions, adjacency, obstacles, candidate_angles, num_rounds,
        centroid=centroid, goal=goal,
    )
    theta_star = select_preferred_direction(final_utilities[0], candidate_angles)

    tau = 3.0
    local_obstacle_lists = [
        visible_obstacles(agent.position, obstacles, agent.sensing_range)
        for agent in agents
    ]
    local_regions = [
        compute_local_free_region(centroid, hull_vertices, local_obstacle_lists[i],
                                  theta_star=theta_star, tau=tau)
        for i in range(len(agents))
    ]
    _, (A_agreed, b_agreed) = _run_consensus(local_regions, adjacency, num_rounds)

    # Phase 5 — formation optimization
    template_name = 'pentagon'
    template = TEMPLATES[template_name]
    result = optimize_formation(A_agreed, b_agreed, template, theta_star)
    if result is None:
        print("No feasible formation found.")
        return
    _, _, _, slots = result

    # Phase 6 — assignment
    assignment, total_cost = solve_assignment(positions, slots)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_rect_obstacles(ax, obstacles)
    draw_free_region(ax, A_agreed, b_agreed,
                     color='mediumseagreen', alpha=0.2, label='Agreed region P')

    # Assignment lines
    for i, agent in enumerate(agents):
        slot_idx = assignment[i]
        ax.plot([agent.position[0], slots[slot_idx, 0]],
                [agent.position[1], slots[slot_idx, 1]],
                color='grey', linewidth=1.2, linestyle='--', zorder=3)

    # Slot positions
    for j, slot in enumerate(slots):
        ax.scatter(slot[0], slot[1], marker='*', s=200, zorder=6,
                   color=ROBOT_COLORS[assignment.tolist().index(j) % len(ROBOT_COLORS)],
                   edgecolors='black', linewidths=0.5)

    # Robot positions
    for i, agent in enumerate(agents):
        color = ROBOT_COLORS[agent.id % len(ROBOT_COLORS)]
        ax.scatter(agent.position[0], agent.position[1],
                   color=color, s=80, zorder=5,
                   edgecolors='black', linewidths=0.5)
        ax.text(agent.position[0] + 0.1, agent.position[1] + 0.1,
                str(agent.id), fontsize=9, zorder=7)

    # Goal
    ax.scatter(goal[0], goal[1], marker='*', s=250, color='gold',
               edgecolors='darkorange', linewidths=0.8, zorder=6, label='Goal')

    all_x = list(positions[:, 0]) + [o.x_min for o in obstacles] + [o.x_max for o in obstacles]
    all_y = list(positions[:, 1]) + [o.y_min for o in obstacles] + [o.y_max for o in obstacles]
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f"Phase 6: robot-to-slot assignment  (total cost = {total_cost:.3f})")
    plt.tight_layout()
    plt.show()

    print(f"Template     : {template_name}")
    print(f"Total cost   : {total_cost:.3f}")
    for i in range(len(agents)):
        print(f"  Robot {i} → Slot {assignment[i]}  "
              f"(dist = {np.linalg.norm(positions[i] - slots[assignment[i]]):.3f})")


def run_control(args: argparse.Namespace) -> None:
    """Phase 7: closed-loop double-integrator motion toward assigned targets.

    What this mode shows (static):
        - Obstacle-free region and obstacles.
        - Each robot's trajectory as a colored line.
        - Final positions and slot positions.

    With --animate:
        - Live animation of robots moving toward their assigned slots.
        - Velocity arrows updated each frame.
    """
    from simulation.scenario import create_default_scenario
    from consensus.graph_utils import graph_diameter
    from consensus.convex_hull import run_convex_hull_consensus
    from consensus.preferred_direction import (
        run_direction_consensus, select_preferred_direction
    )
    from geometry.free_space import compute_local_free_region, visible_obstacles
    from consensus.region_consensus import run_region_consensus as _run_consensus
    from formation.templates import TEMPLATES
    from formation.optimizer import optimize_formation
    from formation.assignment import solve_assignment
    from robots.dynamics import (
        step_double_integrator, compute_pd_acceleration,
        project_velocity, project_velocity_obstacles
    )
    from plotting.draw_team import draw_rect_obstacles, ROBOT_COLORS
    from plotting.draw_obstacles import draw_free_region
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np

    scenario  = create_default_scenario()
    agents    = scenario['agents']
    adjacency = scenario['adjacency']
    obstacles = scenario['obstacles']
    goal      = scenario['goal']

    positions  = np.array([a.position for a in agents])
    num_rounds = args.rounds if args.rounds else graph_diameter(adjacency)
    dt         = args.dt
    n_steps    = args.steps

    # Phases 1–4
    hull_history  = run_convex_hull_consensus(positions, adjacency, num_rounds)
    hull_vertices = hull_history[-1][0]
    centroid      = hull_vertices.mean(axis=0)

    candidate_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    _, final_utilities = run_direction_consensus(
        positions, adjacency, obstacles, candidate_angles, num_rounds,
        centroid=centroid, goal=goal,
    )
    theta_star = select_preferred_direction(final_utilities[0], candidate_angles)

    tau = 3.0
    local_obstacle_lists = [
        visible_obstacles(agent.position, obstacles, agent.sensing_range)
        for agent in agents
    ]
    local_regions = [
        compute_local_free_region(centroid, hull_vertices, local_obstacle_lists[i],
                                  theta_star=theta_star, tau=tau)
        for i in range(len(agents))
    ]
    _, (A_agreed, b_agreed) = _run_consensus(local_regions, adjacency, num_rounds)

    # Phase 5 — formation optimization
    template = TEMPLATES['pentagon']
    result = optimize_formation(A_agreed, b_agreed, template, theta_star)
    if result is None:
        print("No feasible formation found.")
        return
    _, _, _, slots = result

    # Phase 6 — assignment
    assignment, _ = solve_assignment(positions, slots)
    targets = slots[assignment]  # targets[i] is the slot robot i should go to

    # PD gains — critical damping: kd = 2*sqrt(kp)
    kp, kd = 2.0, 2.0 * np.sqrt(2.0)

    # Simulation state
    pos = positions.copy().astype(float)
    vel = np.zeros_like(pos)

    # Record full trajectory history: shape (n_steps+1, N, 2)
    pos_history = [pos.copy()]
    vel_history = [vel.copy()]

    for _ in range(n_steps):
        new_pos = np.zeros_like(pos)
        new_vel = np.zeros_like(vel)
        for i in range(len(agents)):
            u = compute_pd_acceleration(pos[i], vel[i], targets[i], kp, kd)
            # Apply acceleration, then project velocity onto safe region
            v_candidate = vel[i] + dt * u
            v_projected = project_velocity(v_candidate, pos[i], A_agreed, b_agreed)
            v_projected = project_velocity_obstacles(v_projected, pos[i], obstacles)
            new_pos[i], new_vel[i] = step_double_integrator(
                pos[i], v_projected, np.zeros(2), dt
            )
        pos = new_pos
        vel = new_vel
        pos_history.append(pos.copy())
        vel_history.append(vel.copy())

    pos_history = np.array(pos_history)  # (n_steps+1, N, 2)
    vel_history = np.array(vel_history)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(9, 7))
    draw_rect_obstacles(ax, obstacles)
    draw_free_region(ax, A_agreed, b_agreed,
                     color='mediumseagreen', alpha=0.15, label='Agreed region P')

    # Slot positions
    for j, slot in enumerate(slots):
        ax.scatter(slot[0], slot[1], marker='*', s=200, zorder=6,
                   color=ROBOT_COLORS[j % len(ROBOT_COLORS)],
                   edgecolors='black', linewidths=0.5)

    all_x = list(positions[:, 0]) + [o.x_min for o in obstacles] + [o.x_max for o in obstacles]
    all_y = list(positions[:, 1]) + [o.y_min for o in obstacles] + [o.y_max for o in obstacles]
    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.set_aspect('equal')

    if not args.animate:
        # Static: draw full trajectories as lines
        for i in range(len(agents)):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            ax.plot(pos_history[:, i, 0], pos_history[:, i, 1],
                    color=color, linewidth=1.5, zorder=4)
            ax.scatter(pos_history[-1, i, 0], pos_history[-1, i, 1],
                       color=color, s=80, zorder=5,
                       edgecolors='black', linewidths=0.5)
        ax.set_title(f"Phase 7: trajectories over {n_steps} steps  (dt={dt})")
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

    else:
        # Animated: robots move in real time
        robot_dots = [
            ax.scatter([], [], color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                       s=80, zorder=5, edgecolors='black', linewidths=0.5)
            for i in range(len(agents))
        ]
        trails = [
            ax.plot([], [], color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                    linewidth=1.2, alpha=0.5, zorder=3)[0]
            for i in range(len(agents))
        ]
        time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=9)

        def init():
            for dot in robot_dots:
                dot.set_offsets(np.empty((0, 2)))
            for trail in trails:
                trail.set_data([], [])
            time_text.set_text('')
            return robot_dots + trails + [time_text]

        def update(frame):
            for i in range(len(agents)):
                xy = pos_history[frame, i]
                robot_dots[i].set_offsets([xy])
                trails[i].set_data(pos_history[:frame+1, i, 0],
                                   pos_history[:frame+1, i, 1])
            time_text.set_text(f"t = {frame * dt:.2f}s")
            return robot_dots + trails + [time_text]

        ax.set_title(f"Phase 7: animated control  (dt={dt}, {n_steps} steps)")
        ani = animation.FuncAnimation(
            fig, update, frames=n_steps+1,
            init_func=init, interval=50, blit=True
        )
        plt.tight_layout()
        plt.show()

    # Print final position errors
    print(f"Steps: {n_steps},  dt: {dt}")
    for i in range(len(agents)):
        err = np.linalg.norm(pos_history[-1, i] - targets[i])
        print(f"  Robot {i} → Slot {assignment[i]}  final error: {err:.4f}")


def run_closed_loop(args: argparse.Namespace) -> None:
    """Iterative closed-loop: replan after each formation convergence until goal reached.

    Each iteration:
        1. Run phases 1–6 with current robot positions to get new targets.
        2. Simulate robots toward those targets until they converge (or max steps).
        3. Check if the team centroid is within goal_radius of the goal.
        4. If not, repeat.

    What this mode shows (static):
        - All robot trajectories across all iterations, colored by iteration.
        - Each iteration's agreed safe region drawn faintly.
        - Final slot positions and the goal marker.

    With --animate:
        - Live animation of all iterations played sequentially.
        - Region patch and slot markers update at each replanning step.

    Args:
        args: Parsed command-line arguments.
            args.steps    — max simulation steps per planning iteration (default 100).
            args.dt       — timestep in seconds.
            args.max_iters — max number of replanning iterations (default 20).
            args.animate  — live animation if set.

    Notes:
        - target_tol: robots are considered converged when all are within 0.15 of targets.
        - goal_radius: centroid within 0.5 of goal counts as success.
    """
    from simulation.scenario import create_default_scenario
    from consensus.graph_utils import graph_diameter
    from consensus.convex_hull import run_convex_hull_consensus
    from consensus.preferred_direction import (
        run_direction_consensus, select_preferred_direction
    )
    from geometry.free_space import compute_local_free_region, visible_obstacles
    from consensus.region_consensus import run_region_consensus as _run_consensus
    from formation.templates import TEMPLATES
    from formation.optimizer import optimize_formation
    from formation.assignment import solve_assignment
    from robots.dynamics import (
        step_double_integrator, compute_pd_acceleration,
        project_velocity, project_velocity_obstacles
    )
    from plotting.draw_team import draw_rect_obstacles, ROBOT_COLORS
    from plotting.draw_obstacles import draw_free_region
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    import numpy as np

    TARGET_TOL  = 0.05   # robots within this of targets → converged
    GOAL_RADIUS = 1.0    # centroid within this of goal → success
    max_iters   = args.max_iters

    scenario  = create_default_scenario()
    agents    = scenario['agents']
    adjacency = scenario['adjacency']
    obstacles = scenario['obstacles']
    goal      = scenario['goal']

    positions  = np.array([a.position for a in agents])
    num_rounds = args.rounds if args.rounds else graph_diameter(adjacency)
    dt         = args.dt
    kp, kd     = 2.0, 2.0 * np.sqrt(2.0)

    pos = positions.copy().astype(float)
    vel = np.zeros_like(pos)

    # Per-iteration records for plotting
    iter_pos_histories: list[np.ndarray]              = []  # each: (T, N, 2)
    iter_slots:         list[np.ndarray]              = []
    iter_regions:       list[tuple[np.ndarray, np.ndarray]] = []
    iter_assignments:   list[np.ndarray]              = []

    goal_reached = False

    for iteration in range(max_iters):
        # ---- Phase 1: hull consensus ----
        hull_history  = run_convex_hull_consensus(pos, adjacency, num_rounds)
        hull_vertices = hull_history[-1][0]
        centroid      = hull_vertices.mean(axis=0)

        # Check goal before replanning
        if np.linalg.norm(centroid - goal) < GOAL_RADIUS:
            print(f"Goal reached before iteration {iteration + 1}!")
            goal_reached = True
            break

        # ---- Phase 2: preferred direction ----
        candidate_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        _, final_utilities = run_direction_consensus(
            pos, adjacency, obstacles, candidate_angles, num_rounds,
            centroid=centroid, goal=goal,
        )
        theta_star = select_preferred_direction(final_utilities[0], candidate_angles)

        # ---- Phase 3+4: local regions → agreed intersection ----
        tau = 3.0
        local_obs = [
            visible_obstacles(pos[i], obstacles, agents[i].sensing_range)
            for i in range(len(agents))
        ]
        local_regions = [
            compute_local_free_region(centroid, hull_vertices, local_obs[i],
                                      theta_star=theta_star, tau=tau)
            for i in range(len(agents))
        ]
        _, (A_agreed, b_agreed) = _run_consensus(local_regions, adjacency, num_rounds)

        # ---- Phase 5+6: formation + assignment ----
        # Cap tau so the formation center doesn't overshoot the goal.
        direction_vec = np.array([np.cos(theta_star), np.sin(theta_star)])
        dist_to_goal_along_dir = float(direction_vec @ (goal - centroid))
        effective_tau = min(tau, max(dist_to_goal_along_dir, 0.0))
        result = optimize_formation(A_agreed, b_agreed, TEMPLATES['pentagon'], theta_star,
                                    centroid=centroid, tau=effective_tau)
        if result is None:
            print(f"Iteration {iteration + 1}: no feasible formation — stopping.")
            break
        _, _, _, slots = result
        assignment, _ = solve_assignment(pos, slots)
        targets = slots[assignment]

        print(f"Iter {iteration + 1:2d} | θ*={np.degrees(theta_star):6.1f}° | "
              f"centroid={centroid} | targets x∈[{targets[:,0].min():.2f},{targets[:,0].max():.2f}]")

        # ---- Simulate until converged or max steps ----
        iter_hist = [pos.copy()]
        for _ in range(args.steps):
            new_pos = np.zeros_like(pos)
            new_vel = np.zeros_like(vel)
            for i in range(len(agents)):
                u = compute_pd_acceleration(pos[i], vel[i], targets[i], kp, kd)
                v_cand = vel[i] + dt * u
                v_proj = project_velocity(v_cand, pos[i], A_agreed, b_agreed)
                v_proj = project_velocity_obstacles(v_proj, pos[i], obstacles)
                new_pos[i], new_vel[i] = step_double_integrator(
                    pos[i], v_proj, np.zeros(2), dt
                )
            pos = new_pos
            vel = new_vel
            iter_hist.append(pos.copy())
            if all(np.linalg.norm(pos[i] - targets[i]) < TARGET_TOL
                   for i in range(len(agents))):
                break

        iter_pos_histories.append(np.array(iter_hist))
        iter_slots.append(slots)
        iter_regions.append((A_agreed, b_agreed))
        iter_assignments.append(assignment)

        # Check goal after movement
        centroid_now = pos.mean(axis=0)
        if np.linalg.norm(centroid_now - goal) < GOAL_RADIUS:
            print(f"Goal reached after iteration {iteration + 1}!")
            goal_reached = True
            break

    if not goal_reached:
        print(f"Max iterations ({max_iters}) reached without arriving at goal.")

    n_iters = len(iter_pos_histories)
    if n_iters == 0:
        print("No iterations ran — nothing to plot.")
        return

    # Iteration colors — cycle through a colormap
    iter_colors = [cm.tab10(i % 10) for i in range(n_iters)]

    # Axis bounds from all positions + obstacles + goal
    all_pos = np.vstack([h.reshape(-1, 2) for h in iter_pos_histories])
    all_x = list(all_pos[:, 0]) + [o.x_min for o in obstacles] + [o.x_max for o in obstacles] + [goal[0]]
    all_y = list(all_pos[:, 1]) + [o.y_min for o in obstacles] + [o.y_max for o in obstacles] + [goal[1]]
    margin = 1.0
    x_lo, x_hi = min(all_x) - margin, max(all_x) + margin
    y_lo, y_hi = min(all_y) - margin, max(all_y) + margin

    fig, ax = plt.subplots(figsize=(10, 8))
    draw_rect_obstacles(ax, obstacles)
    ax.scatter(goal[0], goal[1], marker='*', s=300, color='gold',
               edgecolors='darkorange', linewidths=1.0, zorder=8, label='Goal')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect('equal')

    if not args.animate:
        # Static: draw all trajectories colored by iteration
        for k, hist in enumerate(iter_pos_histories):
            c = iter_colors[k]
            for i in range(len(agents)):
                ax.plot(hist[:, i, 0], hist[:, i, 1],
                        color=c, linewidth=1.5, alpha=0.8, zorder=4)
            # Faint agreed region
            A_k, b_k = iter_regions[k]
            draw_free_region(ax, A_k, b_k, color=c, alpha=0.07)
            # Slot markers
            for s in iter_slots[k]:
                ax.scatter(s[0], s[1], marker='*', s=120, color=c,
                           edgecolors='black', linewidths=0.4, zorder=6, alpha=0.7)

        # Starting positions
        for i in range(len(agents)):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            ax.scatter(positions[i, 0], positions[i, 1],
                       color=color, s=100, zorder=7,
                       edgecolors='black', linewidths=0.8)
            ax.text(positions[i, 0] + 0.1, positions[i, 1] + 0.1,
                    str(i), fontsize=9, zorder=8)

        # Final positions
        for i in range(len(agents)):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            ax.scatter(pos[i, 0], pos[i, 1],
                       color=color, s=100, marker='D', zorder=7,
                       edgecolors='black', linewidths=0.8)

        ax.set_title(
            f"Closed-loop: {n_iters} planning iteration(s)  "
            f"({'goal reached' if goal_reached else 'max iters'})"
        )
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

    else:
        # Animated: play all iterations sequentially
        # Flatten all frames with iteration tags
        frames_pos  = []   # (N, 2) per frame
        frames_iter = []   # iteration index for each frame
        for k, hist in enumerate(iter_pos_histories):
            for t in range(len(hist)):
                frames_pos.append(hist[t])
                frames_iter.append(k)
        total_frames = len(frames_pos)

        # Draw initial region
        region_patch = [None]
        def _draw_region(k):
            if region_patch[0] is not None:
                region_patch[0].remove()
                region_patch[0] = None
            A_k, b_k = iter_regions[k]
            draw_free_region(ax, A_k, b_k,
                             color=iter_colors[k], alpha=0.15)

        robot_dots = [
            ax.scatter([], [], color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                       s=80, zorder=6, edgecolors='black', linewidths=0.5)
            for i in range(len(agents))
        ]
        trails = [
            ax.plot([], [], color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                    linewidth=1.2, alpha=0.6, zorder=4)[0]
            for i in range(len(agents))
        ]
        slot_dots = ax.scatter([], [], marker='*', s=180, color='black',
                               edgecolors='black', linewidths=0.4, zorder=7)
        info_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=9)

        current_iter = [-1]

        def init():
            for dot in robot_dots:
                dot.set_offsets(np.empty((0, 2)))
            for trail in trails:
                trail.set_data([], [])
            slot_dots.set_offsets(np.empty((0, 2)))
            info_text.set_text('')
            return robot_dots + trails + [slot_dots, info_text]

        def update(frame):
            k   = frames_iter[frame]
            p   = frames_pos[frame]
            hist = iter_pos_histories[k]
            t_in_iter = frame - sum(len(iter_pos_histories[j]) for j in range(k))

            if k != current_iter[0]:
                current_iter[0] = k
                _draw_region(k)
                slot_dots.set_offsets(iter_slots[k])
                slot_dots.set_color(iter_colors[k])

            for i in range(len(agents)):
                robot_dots[i].set_offsets([p[i]])
                trail_x = hist[:t_in_iter + 1, i, 0]
                trail_y = hist[:t_in_iter + 1, i, 1]
                trails[i].set_data(trail_x, trail_y)

            info_text.set_text(
                f"Iteration {k + 1}/{n_iters}  step {t_in_iter}  "
                f"t={t_in_iter * dt:.2f}s"
            )
            return robot_dots + trails + [slot_dots, info_text]

        ax.set_title("Closed-loop iterative replanning")
        ani = animation.FuncAnimation(
            fig, update, frames=total_frames,
            init_func=init, interval=40, blit=False
        )
        plt.tight_layout()
        plt.show()

    print(f"\nIterations run : {n_iters}")
    print(f"Goal reached   : {goal_reached}")
    print(f"Final centroid : {pos.mean(axis=0)}")
    print(f"Distance to goal: {np.linalg.norm(pos.mean(axis=0) - goal):.3f}")


def run_circular(args: argparse.Namespace) -> None:
    """Circular path scenario: 5-robot pentagon loops through 6 waypoints.

    The team starts near the south waypoint and follows a counterclockwise
    circle.  Two choke obstacles on the east side force the formation to shrink
    in order to squeeze through, then re-expand on the far side.

    Each iteration:
        1. Run phases 1–6 with current robot positions to get new formation targets.
        2. Simulate robots toward those targets until convergence (or max steps).
        3. Advance to the next waypoint when the centroid is within GOAL_RADIUS.
        4. Stop after visiting all 6 waypoints once (full loop complete).

    What this mode shows (static):
        - Circular waypoints as numbered markers on the path.
        - All robot trajectories colored by planning iteration.
        - Each iteration's agreed safe region drawn faintly.
        - Choke obstacles in red.

    With --animate:
        - Live animation of all iterations played sequentially.

    Args:
        args: Parsed command-line arguments.
            args.steps     — max simulation steps per planning iteration (default 100).
            args.dt        — timestep in seconds.
            args.max_iters — max replanning iterations across all waypoints (default 60).
            args.animate   — live animation if set.
    """
    from simulation.scenario import create_circular_choke_scenario
    from consensus.graph_utils import graph_diameter
    from consensus.convex_hull import run_convex_hull_consensus
    from consensus.preferred_direction import (
        run_direction_consensus, select_preferred_direction
    )
    from geometry.free_space import compute_local_free_region, visible_obstacles
    from consensus.region_consensus import run_region_consensus as _run_consensus
    from formation.templates import TEMPLATES
    from formation.optimizer import optimize_formation
    from formation.assignment import solve_assignment
    from robots.dynamics import (
        step_double_integrator, compute_pd_acceleration,
        project_velocity, project_velocity_obstacles
    )
    from plotting.draw_team import draw_rect_obstacles, ROBOT_COLORS
    from plotting.draw_obstacles import draw_free_region
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    import numpy as np

    TARGET_TOL  = 0.05   # robots within this of their targets → converged
    GOAL_RADIUS = 1.0    # centroid within this of current waypoint → advance
    WORKSPACE   = (-2.0, -2.0, 12.0, 12.0)
    tau         = 3.0
    kp, kd      = 2.0, 2.0 * np.sqrt(2.0)
    max_iters   = args.max_iters

    scenario  = create_circular_choke_scenario()
    agents    = scenario['agents']
    adjacency = scenario['adjacency']
    obstacles = scenario['obstacles']
    waypoints = scenario['waypoints']

    positions  = np.array([a.position for a in agents])
    num_rounds = args.rounds if args.rounds else graph_diameter(adjacency)
    dt         = args.dt

    pos = positions.copy().astype(float)
    vel = np.zeros_like(pos)

    # Waypoint cycling state
    current_wp       = 0
    waypoints_visited = 0
    loop_complete    = False

    # Per-iteration records for plotting
    iter_pos_histories: list[np.ndarray]                    = []
    iter_slots:         list[np.ndarray]                    = []
    iter_regions:       list[tuple[np.ndarray, np.ndarray]] = []
    iter_assignments:   list[np.ndarray]                    = []
    iter_wp_index:      list[int]                           = []  # waypoint at each iter

    for iteration in range(max_iters):
        goal = waypoints[current_wp]

        # ---- Phase 1: hull consensus ----
        hull_history  = run_convex_hull_consensus(pos, adjacency, num_rounds)
        hull_vertices = hull_history[-1][0]
        centroid      = hull_vertices.mean(axis=0)

        # Check if already at this waypoint before replanning
        if np.linalg.norm(centroid - goal) < GOAL_RADIUS:
            waypoints_visited += 1
            current_wp = (current_wp + 1) % len(waypoints)
            if waypoints_visited == len(waypoints):
                print("Full loop complete!")
                loop_complete = True
                break
            goal = waypoints[current_wp]

        # ---- Phase 2: preferred direction ----
        candidate_angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        _, final_utilities = run_direction_consensus(
            pos, adjacency, obstacles, candidate_angles, num_rounds,
            centroid=centroid, goal=goal,
        )
        theta_star = select_preferred_direction(final_utilities[0], candidate_angles)

        # ---- Phase 3+4: local regions → agreed intersection ----
        local_obs = [
            visible_obstacles(pos[i], obstacles, agents[i].sensing_range)
            for i in range(len(agents))
        ]
        local_regions = [
            compute_local_free_region(
                centroid, hull_vertices, local_obs[i],
                workspace_bounds=WORKSPACE,
                theta_star=theta_star, tau=tau,
            )
            for i in range(len(agents))
        ]
        _, (A_agreed, b_agreed) = _run_consensus(local_regions, adjacency, num_rounds)

        # ---- Phase 5+6: formation + assignment ----
        # Cap tau so the formation center doesn't overshoot the current waypoint.
        direction_vec = np.array([np.cos(theta_star), np.sin(theta_star)])
        dist_to_goal_along_dir = float(direction_vec @ (goal - centroid))
        effective_tau = min(tau, max(dist_to_goal_along_dir, 0.0))
        result = optimize_formation(A_agreed, b_agreed, TEMPLATES['pentagon'], theta_star,
                                    centroid=centroid, tau=effective_tau)
        if result is None:
            print(f"Iteration {iteration + 1}: no feasible formation — stopping.")
            break
        _, _, _, slots = result
        assignment, _ = solve_assignment(pos, slots)
        targets = slots[assignment]

        print(f"Iter {iteration + 1:2d} | WP {current_wp} | θ*={np.degrees(theta_star):6.1f}° | "
              f"centroid=({centroid[0]:.2f},{centroid[1]:.2f}) | "
              f"visited={waypoints_visited}/{len(waypoints)}")

        # ---- Simulate until converged or max steps ----
        iter_hist = [pos.copy()]
        for _ in range(args.steps):
            new_pos = np.zeros_like(pos)
            new_vel = np.zeros_like(vel)
            for i in range(len(agents)):
                u = compute_pd_acceleration(pos[i], vel[i], targets[i], kp, kd)
                v_cand = vel[i] + dt * u
                v_proj = project_velocity(v_cand, pos[i], A_agreed, b_agreed)
                v_proj = project_velocity_obstacles(v_proj, pos[i], obstacles)
                new_pos[i], new_vel[i] = step_double_integrator(
                    pos[i], v_proj, np.zeros(2), dt
                )
            pos = new_pos
            vel = new_vel
            iter_hist.append(pos.copy())
            if all(np.linalg.norm(pos[i] - targets[i]) < TARGET_TOL
                   for i in range(len(agents))):
                break

        iter_pos_histories.append(np.array(iter_hist))
        iter_slots.append(slots)
        iter_regions.append((A_agreed, b_agreed))
        iter_assignments.append(assignment)
        iter_wp_index.append(current_wp)

        # Check waypoint after movement
        centroid_now = pos.mean(axis=0)
        if np.linalg.norm(centroid_now - goal) < GOAL_RADIUS:
            waypoints_visited += 1
            current_wp = (current_wp + 1) % len(waypoints)
            if waypoints_visited == len(waypoints):
                print("Full loop complete!")
                loop_complete = True
                break

    if not loop_complete:
        print(f"Max iterations ({max_iters}) reached — "
              f"{waypoints_visited}/{len(waypoints)} waypoints visited.")

    n_iters = len(iter_pos_histories)
    if n_iters == 0:
        print("No iterations ran — nothing to plot.")
        return

    # Iteration colors cycling through tab10
    iter_colors = [cm.tab10(i % 10) for i in range(n_iters)]

    # Axis bounds: workspace + small margin
    x_lo, y_lo, x_hi, y_hi = WORKSPACE
    margin = 0.5

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_rect_obstacles(ax, obstacles)

    # Draw circular waypoints as numbered markers
    wp_arr = np.array(waypoints)
    ax.scatter(wp_arr[:, 0], wp_arr[:, 1], marker='o', s=200,
               color='gold', edgecolors='darkorange', linewidths=1.5,
               zorder=8, label='Waypoints')
    for idx, wp in enumerate(waypoints):
        ax.text(wp[0] + 0.15, wp[1] + 0.15, str(idx), fontsize=10,
                fontweight='bold', zorder=9, color='darkorange')

    # Faint circle guide
    theta_circ = np.linspace(0, 2 * np.pi, 200)
    cx, cy, cr = 5.0, 5.0, 3.5
    ax.plot(cx + cr * np.cos(theta_circ), cy + cr * np.sin(theta_circ),
            '--', color='gray', linewidth=0.8, alpha=0.5, zorder=2)

    ax.set_xlim(x_lo - margin, x_hi + margin)
    ax.set_ylim(y_lo - margin, y_hi + margin)
    ax.set_aspect('equal')

    if not args.animate:
        for k, hist in enumerate(iter_pos_histories):
            c = iter_colors[k]
            for i in range(len(agents)):
                ax.plot(hist[:, i, 0], hist[:, i, 1],
                        color=c, linewidth=1.5, alpha=0.8, zorder=4)
            A_k, b_k = iter_regions[k]
            draw_free_region(ax, A_k, b_k, color=c, alpha=0.07)
            for s in iter_slots[k]:
                ax.scatter(s[0], s[1], marker='*', s=120, color=c,
                           edgecolors='black', linewidths=0.4, zorder=6, alpha=0.7)

        # Starting robot positions
        for i in range(len(agents)):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            ax.scatter(positions[i, 0], positions[i, 1],
                       color=color, s=100, zorder=7,
                       edgecolors='black', linewidths=0.8)
            ax.text(positions[i, 0] + 0.1, positions[i, 1] + 0.1,
                    str(i), fontsize=9, zorder=8)

        # Final robot positions
        for i in range(len(agents)):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            ax.scatter(pos[i, 0], pos[i, 1],
                       color=color, s=100, marker='D', zorder=7,
                       edgecolors='black', linewidths=0.8)

        ax.set_title(
            f"Circular choke: {n_iters} planning iters  |  "
            f"{waypoints_visited}/{len(waypoints)} waypoints visited  "
            f"({'loop complete' if loop_complete else 'incomplete'})"
        )
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()

    else:
        # Animated: flatten all frames with iteration tags
        frames_pos  = []
        frames_iter = []
        for k, hist in enumerate(iter_pos_histories):
            for t in range(len(hist)):
                frames_pos.append(hist[t])
                frames_iter.append(k)
        total_frames = len(frames_pos)

        region_patch = [None]

        def _draw_region(k):
            if region_patch[0] is not None:
                region_patch[0].remove()
                region_patch[0] = None
            A_k, b_k = iter_regions[k]
            draw_free_region(ax, A_k, b_k, color=iter_colors[k], alpha=0.15)

        robot_dots = [
            ax.scatter([], [], color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                       s=80, zorder=6, edgecolors='black', linewidths=0.5)
            for i in range(len(agents))
        ]
        trails = [
            ax.plot([], [], color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                    linewidth=1.2, alpha=0.6, zorder=4)[0]
            for i in range(len(agents))
        ]
        slot_dots = ax.scatter([], [], marker='*', s=180, color='black',
                               edgecolors='black', linewidths=0.4, zorder=7)
        info_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=9)

        current_iter = [-1]

        def init():
            for dot in robot_dots:
                dot.set_offsets(np.empty((0, 2)))
            for trail in trails:
                trail.set_data([], [])
            slot_dots.set_offsets(np.empty((0, 2)))
            info_text.set_text('')
            return robot_dots + trails + [slot_dots, info_text]

        def update(frame):
            k   = frames_iter[frame]
            p   = frames_pos[frame]
            hist = iter_pos_histories[k]
            t_in_iter = frame - sum(len(iter_pos_histories[j]) for j in range(k))

            if k != current_iter[0]:
                current_iter[0] = k
                _draw_region(k)
                slot_dots.set_offsets(iter_slots[k])
                slot_dots.set_color(iter_colors[k])

            for i in range(len(agents)):
                robot_dots[i].set_offsets([p[i]])
                trail_x = hist[:t_in_iter + 1, i, 0]
                trail_y = hist[:t_in_iter + 1, i, 1]
                trails[i].set_data(trail_x, trail_y)

            wp_idx = iter_wp_index[k]
            info_text.set_text(
                f"Iter {k + 1}/{n_iters}  WP {wp_idx}  step {t_in_iter}  "
                f"t={t_in_iter * dt:.2f}s"
            )
            return robot_dots + trails + [slot_dots, info_text]

        ax.set_title("Circular choke — iterative replanning")
        ani = animation.FuncAnimation(
            fig, update, frames=total_frames,
            init_func=init, interval=40, blit=False
        )
        plt.tight_layout()
        plt.show()

    print(f"\nIterations run    : {n_iters}")
    print(f"Waypoints visited : {waypoints_visited}/{len(waypoints)}")
    print(f"Loop complete     : {loop_complete}")
    print(f"Final centroid    : {pos.mean(axis=0)}")


def run_full_demo(args: argparse.Namespace) -> None:
    """Phase 8: run the complete paper pipeline end-to-end.

    Stages: hull consensus → direction consensus → local regions →
            region consensus → formation optimization → assignment →
            closed-loop motion.
    """
    # TODO: implement in Phase 8
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------

MODES: dict = {
    "scaffold":         run_scaffold,
    "hull":             run_hull,
    "direction":        run_direction,
    "local-free-space": run_local_free_space,
    "region-consensus": run_region_consensus,
    "formation":        run_formation,
    "assignment":       run_assignment,
    "control":          run_control,
    "closed-loop":      run_closed_loop,
    "circular":         run_circular,
    "full-demo":        run_full_demo,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distributed multi-robot formation control — teaching implementation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available modes:\n" + "\n".join(
            f"  python main.py --mode {m}" for m in MODES
        ),
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=list(MODES.keys()),
        help="Which phase / demo to run.",
    )
    parser.add_argument(
        "--scenario", type=str, default="default",
        help="Scenario preset name (default: 'default').",
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="Override number of consensus rounds (default: graph diameter).",
    )
    parser.add_argument(
        "--dt", type=float, default=0.1,
        help="Simulation timestep in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of simulation steps for control/animation modes (default: 100).",
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Show an animation instead of a static plot where applicable.",
    )
    parser.add_argument(
        "--max-iters", type=int, default=20, dest="max_iters",
        help="Max number of replanning iterations for closed-loop mode (default: 20).",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    MODES[args.mode](args)
