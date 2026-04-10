"""Distributed value-iteration planner for the me595 pipeline.

This module is **Phase C** of the value-iteration replacement for the
myopic preferred-direction step (Section 3.2 of Alonso-Mora et al.
2019, "Distributed multi-robot formation control in dynamic
environments").  It wraps the single-agent grid MDP from
``me595/grid_mdp.py`` in a distributed-consensus interface that mirrors
the structure of the paper's other consensus algorithms (convex hull,
direction, free region).

Distributed framing
-------------------
Each robot ``i`` independently computes a local cost-to-go grid
``V_i`` from the obstacles it can see (Phase B's value iteration on
its own GridMDP).  The robots then run a **max-consensus** on the
flattened V arrays:

    V_i(k+1) = element-wise max over j in N_i ∪ {i}  of V_j(k)

This is the cost-space dual of the paper's min-consensus on utilities
(Alg. 2): higher V means a more conservative cost-to-go, so taking the
elementwise max respects every robot's obstacle knowledge — if any
robot in the team has marked a cell as expensive (or impassable), the
team agrees to treat it that way.  After ``d`` rounds (graph diameter)
all robots converge to a common ``V_agreed``, exactly as Algorithms
1-3 of the paper converge their respective quantities.

In the current me595 scenario every robot has ``sensing_range = 20`` and
sees the entire obstacle, so the consensus is a structural no-op — every
local grid is identical from the start.  The code still runs the
consensus loop so the architecture is correct under partial
observability (the case the paper actually targets).

Reference: Kochenderfer, Wheeler & Wray, *Algorithms for Decision
Making* (MIT Press, 2022), Chapter 7 (Exact Solution Methods),
Section 7.5 (Value Iteration).  The single-robot solver is the
canonical exact MDP value iteration; the max-consensus wrapper is
the same flooding pattern Alonso-Mora et al. use for their other
distributed steps.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from me595.geometry import ConvexObstacle, polygon_distance_to_point
from me595.grid_mdp import (
    GridMDP,
    create_grid,
    extract_direction,
    mark_obstacles,
    value_iteration,
)


# ---------------------------------------------------------------------------
# Local (single-robot) value function
# ---------------------------------------------------------------------------


def compute_local_value_function(
    obstacles: Sequence[ConvexObstacle],
    workspace_bounds: tuple[float, float, float, float],
    goal: np.ndarray,
    cell_size: float = 0.25,
    inflation: float = 0.15,
    sensing_range: float | None = None,
    tol: float = 1e-3,
    max_iters: int = 500,
) -> GridMDP:
    """Build and solve a single robot's local grid MDP.

    Pipeline:

        1. Allocate a fresh GridMDP for ``workspace_bounds``.
        2. Mark obstacle cells (optionally restricted to those within
           ``sensing_range`` of the workspace center, as a stand-in for
           per-robot limited visibility).
        3. Run synchronous value iteration to convergence.

    The result is a fully-solved cost-to-go field that this robot would
    use *on its own*, without any input from teammates.  The
    distributed reconciliation across the team happens in
    :func:`run_value_consensus`.

    Args:
        obstacles:        Convex obstacles known to this robot.  In the
                          me595 scenario callers may pass the already-
                          filtered list returned by
                          ``me595.geometry.visible_obstacles``; this
                          function does not assume which obstacles
                          belong to which robot.
        workspace_bounds: ``(x_min, y_min, x_max, y_max)``.  Same for
                          every robot — the team plans in a shared
                          frame.
        goal:             World-coordinate goal point, shape (2,).
        cell_size:        Grid resolution in world units.  Default 0.25
                          gives a 64x80 grid for the me595 workspace.
        inflation:        Obstacle safety margin.  See ``mark_obstacles``.
        sensing_range:    Optional secondary filter on obstacles.  If
                          provided, only obstacles whose nearest point
                          is within ``sensing_range`` of the workspace
                          *center* are marked.  This is a coarse
                          stand-in for per-robot visibility used to
                          exercise the consensus path — actual robot
                          positions are not threaded through here so
                          the function stays a pure
                          (obstacles -> grid) operation.
        tol, max_iters:   Forwarded to ``value_iteration``.

    Returns:
        A solved GridMDP whose ``values`` field holds V*(s) for every
        cell.
    """
    grid = create_grid(workspace_bounds, cell_size, goal)

    if sensing_range is not None:
        x_center = 0.5 * (workspace_bounds[0] + workspace_bounds[2])
        y_center = 0.5 * (workspace_bounds[1] + workspace_bounds[3])
        center = np.array([x_center, y_center])
        obstacles = [
            obs for obs in obstacles
            if polygon_distance_to_point(obs, center) <= sensing_range
        ]

    mark_obstacles(grid, obstacles, inflation=inflation)
    value_iteration(grid, tol=tol, max_iters=max_iters)
    return grid


# ---------------------------------------------------------------------------
# Distributed max-consensus on cost-to-go
# ---------------------------------------------------------------------------


def run_value_consensus(
    local_grids: Sequence[GridMDP],
    adjacency: np.ndarray,
    num_rounds: int,
) -> GridMDP:
    """Reconcile per-robot value functions by element-wise max-consensus.

    Each robot starts with its own ``V_i`` and runs the synchronous
    update

        V_i(k+1) = max( V_i(k),  max_{j in N_i} V_j(k) )

    elementwise across the cost-to-go grid, mirroring Algorithm 2 of
    Alonso-Mora et al. (2019) but with **max** instead of **min**
    because we are propagating costs (higher = more conservative)
    rather than utilities (higher = better).  After ``num_rounds``
    rounds — typically the graph diameter ``d`` — every robot holds
    the same value field, equal to the elementwise max of all initial
    grids.

    The Bellman equation is **not** re-solved between rounds: this is a
    pure consensus, not a distributed value iteration.  Each robot's
    initial ``V_i`` is already an exact local solve; consensus only
    reconciles disagreements coming from differing obstacle knowledge.

    Args:
        local_grids: One GridMDP per robot.  All grids must share the
                     same shape, bounds, cell size, and goal cell —
                     they only differ in which cells are blocked / how
                     V was filled in.
        adjacency:   Symmetric (N, N) binary adjacency matrix.
        num_rounds:  Number of consensus rounds to execute.  Use the
                     graph diameter for guaranteed convergence under
                     synchronous flooding.

    Returns:
        A single GridMDP holding the consensus value field.  All
        per-robot grids are also updated in place so callers can
        continue to use them as agreed grids.

    Raises:
        ValueError: If grid shapes / bounds / goal mismatch, or if
                    adjacency has the wrong shape.
    """
    if len(local_grids) == 0:
        raise ValueError("run_value_consensus needs at least one local grid.")

    N = len(local_grids)
    A = np.asarray(adjacency, dtype=float)
    if A.shape != (N, N):
        raise ValueError(
            f"adjacency must have shape ({N}, {N}), got {A.shape}"
        )

    # Sanity-check that all grids agree on the basic geometry.  Without
    # this the elementwise max would silently broadcast or fail.
    ref = local_grids[0]
    for k, g in enumerate(local_grids):
        if (g.rows, g.cols) != (ref.rows, ref.cols):
            raise ValueError(
                f"Grid {k} has shape ({g.rows}, {g.cols}); "
                f"expected ({ref.rows}, {ref.cols})."
            )
        if g.cell_size != ref.cell_size:
            raise ValueError(
                f"Grid {k} cell_size {g.cell_size} != {ref.cell_size}."
            )
        if g.goal_ij != ref.goal_ij:
            raise ValueError(
                f"Grid {k} goal_ij {g.goal_ij} != {ref.goal_ij}."
            )

    # Snapshot of every robot's current V — we update one buffer per
    # robot per round to keep the consensus synchronous (no robot sees
    # a neighbor's already-updated value within the same round).
    values = [g.values.copy() for g in local_grids]

    for _ in range(num_rounds):
        new_values = [v.copy() for v in values]
        for i in range(N):
            neighbors = np.where(A[i] > 0.0)[0]
            merged = values[i]
            for j in neighbors:
                merged = np.maximum(merged, values[j])
            new_values[i] = merged
        values = new_values

    # Write the agreed field back into every grid so all robots
    # downstream see the consensus result, and return one of them as
    # the canonical "agreed grid" for the caller.
    for g, v in zip(local_grids, values):
        g.values[:] = v
        # Blocked cells stay inf; consensus on inf is still inf, so the
        # blocked mask remains consistent with the values array.

    return local_grids[0]


# ---------------------------------------------------------------------------
# Direction extraction wrapper
# ---------------------------------------------------------------------------


def plan_direction(grid: GridMDP, centroid: np.ndarray) -> float:
    """Return ``theta_star`` at ``centroid`` from the agreed value field.

    Thin wrapper around :func:`me595.grid_mdp.extract_direction`,
    provided so the rest of the pipeline can talk to a single
    "planner" entry point that mirrors the way ``geometry.py``
    exposes :func:`run_direction_consensus` for the greedy planner.

    Args:
        grid:     A GridMDP whose values have been solved (and ideally
                  reconciled across the team via
                  :func:`run_value_consensus`).
        centroid: World coordinates of the team hull centroid, shape (2,).

    Returns:
        Preferred direction angle in radians, in ``[-pi, pi]``.
    """
    return extract_direction(grid, np.asarray(centroid, dtype=float))


# ---------------------------------------------------------------------------
# Phase C standalone sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build per-robot value functions for the me595 scenario, run
    # max-consensus, and verify (a) the agreed grid equals each
    # local grid (no-op since all robots see everything), and (b)
    # plan_direction at the start centroid points around the obstacle.
    from me595.geometry import visible_obstacles
    from me595.scenario import create_triangle_scenario

    scenario = create_triangle_scenario()
    workspace_bounds = (-2.0, -3.0, 14.0, 17.0)
    goal = scenario["goal"]
    agents = scenario["agents"]
    obstacles = scenario["obstacles"]
    adjacency = scenario["adjacency"]
    N = len(agents)

    # Per-robot local solves.  Each robot uses its own visible
    # obstacles, even though in this scenario sensing_range = 20 makes
    # every list identical.
    local_grids = [
        compute_local_value_function(
            visible_obstacles(agents[i].position, obstacles, agents[i].sensing_range),
            workspace_bounds=workspace_bounds,
            goal=goal,
            cell_size=0.25,
            inflation=0.15,
        )
        for i in range(N)
    ]

    def _max_finite_diff(a: np.ndarray, b: np.ndarray) -> float:
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            return 0.0
        return float(np.max(np.abs(a[mask] - b[mask])))

    # Sanity: all local grids should be identical here.
    ref = local_grids[0].values
    max_local_diff = max(
        _max_finite_diff(g.values, ref) for g in local_grids[1:]
    )
    print(f"Max diff between local grids before consensus: {max_local_diff:.3e}")

    # Run max-consensus across the ring graph (diameter 2).
    from consensus.graph_utils import graph_diameter  # noqa: E402
    d = max(1, graph_diameter(adjacency))
    print(f"Graph diameter: {d}")
    agreed = run_value_consensus(local_grids, adjacency, num_rounds=d)

    # After consensus, every grid must equal the agreed one.
    max_post = max(
        _max_finite_diff(g.values, agreed.values) for g in local_grids
    )
    print(f"Max diff between local grids after consensus:  {max_post:.3e}")

    centroid = np.mean([a.position for a in agents], axis=0)
    theta = plan_direction(agreed, centroid)
    print(
        f"Start centroid = ({centroid[0]:.2f}, {centroid[1]:.2f})  "
        f"plan_direction -> theta* = {np.degrees(theta):6.1f} deg"
    )

    # Spot-check a few other points to confirm the planner makes
    # geometric sense.
    for label, p in [
        ("stuck point",   np.array([2.67, 7.57])),
        ("below triangle", np.array([4.0, -2.5])),
        ("east of obstacle", np.array([10.0, 5.0])),
    ]:
        t = plan_direction(agreed, p)
        print(f"  {label:18s} ({p[0]:5.2f}, {p[1]:5.2f}) -> {np.degrees(t):6.1f} deg")
