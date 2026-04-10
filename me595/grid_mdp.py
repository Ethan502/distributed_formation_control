"""Grid Markov Decision Process for me595 value-iteration path planning.

This module implements **Phase A** of the value-iteration planner: a 2D
discretization of the workspace with obstacle marking.  Value iteration
itself (Phase B) is added later in this same file.

Design reference
----------------
The MDP formulation follows Kochenderfer, Wheeler & Wray, *Algorithms
for Decision Making* (MIT Press, 2022), Chapter 7 (Exact Solution
Methods).  Each grid cell is a state; actions are 8-connected moves to
neighboring cells; transitions are deterministic; the per-step cost is
the Euclidean step length.  The goal cell is absorbing with zero
cost-to-go.  Cells inside (or near) an obstacle are impassable and
carry an infinite cost-to-go.

Coordinate convention
---------------------
The grid stores cells in (row, col) order with **row 0 at the bottom**
of the workspace (y_min) and column 0 at the left (x_min).  This matches
``imshow(..., origin="lower")`` so heatmaps render with +y pointing up.

    row = int((y - y_min) / cell_size)
    col = int((x - x_min) / cell_size)

Cell *centers* (used for containment tests) are offset by half a cell:

    x_center = x_min + (col + 0.5) * cell_size
    y_center = y_min + (row + 0.5) * cell_size
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from me595.geometry import ConvexObstacle, polygon_distance_to_point


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class GridMDP:
    """A 2D grid Markov Decision Process for shortest-path planning.

    Attributes:
        x_min, y_min, x_max, y_max: Workspace bounds in world coordinates.
        cell_size: Edge length of one square cell, in world units.
        rows, cols: Grid dimensions.  ``rows = ceil((y_max - y_min)/cell_size)``
            and ``cols = ceil((x_max - x_min)/cell_size)``.
        blocked: Bool array of shape (rows, cols).  True means the cell
            is impassable (inside an obstacle, or within the inflation
            margin of one).
        values: Float array of shape (rows, cols) holding the optimal
            cost-to-go V*(s) for each cell.  Initialized to ``+inf``
            everywhere except the goal cell, which is 0.  Blocked cells
            stay at ``+inf`` forever.
        goal_ij: (row, col) of the goal cell.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    cell_size: float
    rows: int
    cols: int
    blocked: np.ndarray
    values: np.ndarray
    goal_ij: tuple[int, int]


# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------


def world_to_grid(grid: GridMDP, point: np.ndarray) -> tuple[int, int]:
    """Convert a world (x, y) point to (row, col) grid indices.

    The result is clipped to the valid index range so that points on or
    just past the workspace boundary still map to a sensible cell rather
    than raising.  Callers that need to detect out-of-bounds queries
    should test the world coordinates themselves.
    """
    p = np.asarray(point, dtype=float)
    col = int((p[0] - grid.x_min) / grid.cell_size)
    row = int((p[1] - grid.y_min) / grid.cell_size)
    col = max(0, min(grid.cols - 1, col))
    row = max(0, min(grid.rows - 1, row))
    return row, col


def grid_to_world(grid: GridMDP, row: int, col: int) -> np.ndarray:
    """Convert (row, col) grid indices to the world coordinates of that cell's center."""
    x = grid.x_min + (col + 0.5) * grid.cell_size
    y = grid.y_min + (row + 0.5) * grid.cell_size
    return np.array([x, y], dtype=float)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def create_grid(
    workspace_bounds: tuple[float, float, float, float],
    cell_size: float,
    goal: np.ndarray,
) -> GridMDP:
    """Allocate a fresh GridMDP covering ``workspace_bounds``.

    All cells start unblocked.  ``values`` is initialized to ``+inf``
    everywhere except the goal cell, which is 0 (the absorbing
    zero-cost state of the MDP).

    Args:
        workspace_bounds: ``(x_min, y_min, x_max, y_max)`` in world units.
        cell_size:        Edge length of one square cell.  ``0.25`` for
                          the me595 scenario gives a 64x80 grid.
        goal:             World-coordinate goal point, shape (2,).

    Returns:
        A fully populated GridMDP with empty obstacle map.

    Raises:
        ValueError: If the bounds are degenerate or the goal lies
                    outside ``workspace_bounds``.
    """
    x_min, y_min, x_max, y_max = workspace_bounds
    if not (x_max > x_min and y_max > y_min):
        raise ValueError(
            f"Degenerate workspace bounds: {workspace_bounds}"
        )
    if cell_size <= 0.0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")

    # ceil so the grid fully covers the workspace even if the extents
    # are not exact multiples of cell_size.
    cols = int(np.ceil((x_max - x_min) / cell_size))
    rows = int(np.ceil((y_max - y_min) / cell_size))

    g = np.asarray(goal, dtype=float)
    if not (x_min <= g[0] <= x_max and y_min <= g[1] <= y_max):
        raise ValueError(
            f"Goal {tuple(g)} lies outside workspace bounds {workspace_bounds}."
        )

    blocked = np.zeros((rows, cols), dtype=bool)
    values = np.full((rows, cols), np.inf, dtype=float)

    # Absorbing zero-cost goal state.  We compute the goal index by hand
    # (rather than calling world_to_grid) so this function has no
    # dependency on a partially-initialized GridMDP instance.
    goal_col = int((g[0] - x_min) / cell_size)
    goal_row = int((g[1] - y_min) / cell_size)
    goal_col = max(0, min(cols - 1, goal_col))
    goal_row = max(0, min(rows - 1, goal_row))
    values[goal_row, goal_col] = 0.0

    return GridMDP(
        x_min=float(x_min),
        y_min=float(y_min),
        x_max=float(x_max),
        y_max=float(y_max),
        cell_size=float(cell_size),
        rows=rows,
        cols=cols,
        blocked=blocked,
        values=values,
        goal_ij=(goal_row, goal_col),
    )


def mark_obstacles(
    grid: GridMDP,
    obstacles: Iterable[ConvexObstacle],
    inflation: float = 0.15,
) -> None:
    """Mark every cell that lies inside or near an obstacle as blocked.

    A cell is blocked iff its **center** is within ``inflation`` world
    units of any obstacle.  Inflation accounts for the finite size of
    the robots — half the inter-robot safety margin (~0.15 m) keeps the
    planned path clear of obstacle faces.

    Blocked cells have their cost-to-go forced to ``+inf``, even if they
    happen to coincide with the goal cell.  (In that pathological case
    the goal is unreachable; value iteration will leave most of the
    grid at ``+inf`` and the caller should notice.)

    Args:
        grid:       The GridMDP to update in place.
        obstacles:  Convex obstacles (Triangle, Rectangle, ...).  Tested
                    via ``polygon_distance_to_point`` so any shape that
                    me595/geometry.py understands works here.
        inflation:  Safety margin in world units.  ``0.0`` to disable.
    """
    if inflation < 0.0:
        raise ValueError(f"inflation must be non-negative, got {inflation}")

    obstacles = list(obstacles)
    if not obstacles:
        return

    # Iterate over every cell once.  For the me595 64x80 grid this is
    # 5120 cells * a handful of polygon distance checks each — well
    # under a millisecond.  Vectorizing would require per-shape code
    # and isn't worth the complexity.
    for row in range(grid.rows):
        y = grid.y_min + (row + 0.5) * grid.cell_size
        for col in range(grid.cols):
            x = grid.x_min + (col + 0.5) * grid.cell_size
            center = np.array([x, y])
            for obs in obstacles:
                if polygon_distance_to_point(obs, center) <= inflation:
                    grid.blocked[row, col] = True
                    grid.values[row, col] = np.inf
                    break  # one obstacle is enough


# ---------------------------------------------------------------------------
# Phase B — value iteration solver
# ---------------------------------------------------------------------------
#
# We treat the grid as a deterministic shortest-path MDP:
#
#     states  S       = (row, col) cells that are not blocked
#     actions A(s)    = up to 8 moves to neighboring cells
#     T(s'|s,a)       = 1 if s' is the neighbor reached by a, else 0
#     cost(s, a)      = Euclidean step length
#                       cell_size           for N/S/E/W
#                       cell_size * sqrt(2) for the four diagonals
#     terminal state  = goal cell, V*(goal) = 0
#
# The optimal cost-to-go satisfies the Bellman equation
#
#     V*(s) = min_{a in A(s)}  [ cost(s, a) + V*(s') ]
#
# and value iteration computes it by repeated synchronous backups
#     V_{k+1}(s) <- min_a [ cost(s, a) + V_k(s') ]
# until ||V_{k+1} - V_k||_inf < tol.
#
# Reference: Kochenderfer, Wheeler & Wray, *Algorithms for Decision
# Making* (MIT Press, 2022), Chapter 7 (Exact Solution Methods),
# Section 7.5 (Value Iteration).  The deterministic-grid case here is
# the canonical "shortest path on a weighted lattice" textbook example
# of an exact MDP solver.


# Eight neighbor offsets and their per-step Euclidean costs as multiples
# of cell_size.  Order is fixed but irrelevant — value iteration only
# cares about the per-cell minimum.
_NEIGHBOR_OFFSETS: tuple[tuple[int, int, float], ...] = (
    ( 1,  0, 1.0),                  # N
    (-1,  0, 1.0),                  # S
    ( 0,  1, 1.0),                  # E
    ( 0, -1, 1.0),                  # W
    ( 1,  1, float(np.sqrt(2.0))),  # NE
    ( 1, -1, float(np.sqrt(2.0))),  # NW
    (-1,  1, float(np.sqrt(2.0))),  # SE
    (-1, -1, float(np.sqrt(2.0))),  # SW
)


def neighbors_8(
    row: int, col: int, rows: int, cols: int, cell_size: float
) -> list[tuple[int, int, float]]:
    """Return up to eight in-bounds neighbors of cell (row, col).

    Each entry is ``(neighbor_row, neighbor_col, step_cost)`` with the
    step cost expressed in world units (already multiplied by
    ``cell_size``).  Out-of-bounds neighbors are skipped.

    This helper is exported for tests / future asynchronous solvers; the
    main vectorized ``value_iteration`` loop below does not call it.
    """
    out: list[tuple[int, int, float]] = []
    for dr, dc, base_cost in _NEIGHBOR_OFFSETS:
        r, c = row + dr, col + dc
        if 0 <= r < rows and 0 <= c < cols:
            out.append((r, c, base_cost * cell_size))
    return out


def value_iteration(
    grid: GridMDP, tol: float = 1e-3, max_iters: int = 500
) -> int:
    """Solve the grid MDP by synchronous Bellman backups.

    Updates ``grid.values`` in place.  The goal cell stays at 0
    (absorbing terminal state) and blocked cells stay at ``+inf``
    (impassable, infinite cost-to-go).  Every other cell converges to
    its optimal cost-to-go to the goal under the 8-connected,
    deterministic-step model.

    Implementation
    --------------
    Rather than looping cell-by-cell in Python (~5k cells * 8 neighbors
    * 200 iters = 8M operations) we vectorize one full Bellman sweep as
    eight shifted views of a padded V array:

        for each (dr, dc, step_cost):
            shifted = V_padded[1+dr : 1+dr+rows, 1+dc : 1+dc+cols]
            candidates[k] = step_cost + shifted

        V_new = element-wise min over the 8 candidates

    Padding with ``+inf`` makes out-of-bounds neighbors automatically
    "impossibly expensive", so the per-cell min ignores them.  Likewise
    blocked neighbors carry V = inf and are dominated.

    Args:
        grid:      The GridMDP whose ``values`` array is updated in place.
        tol:       Convergence tolerance on ``max |V_{k+1} - V_k|``.
        max_iters: Hard cap on iterations (safety net).

    Returns:
        Number of Bellman sweeps actually performed.
    """
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if max_iters <= 0:
        raise ValueError(f"max_iters must be positive, got {max_iters}")

    rows, cols = grid.rows, grid.cols
    blocked = grid.blocked
    goal_r, goal_c = grid.goal_ij
    h = grid.cell_size

    # Per-action step costs in world units, in the same order as
    # _NEIGHBOR_OFFSETS.
    offsets = _NEIGHBOR_OFFSETS
    step_costs = np.array([h * c for _, _, c in offsets], dtype=float)

    V = grid.values  # alias; we update in place via assignment below

    # We mutate V in place each sweep.  ``updateable`` is the mask of
    # cells that participate in the Bellman backup: not blocked, and
    # not the goal cell (the goal is a fixed terminal state).
    updateable = ~blocked
    updateable[goal_r, goal_c] = False

    # Pre-allocate the padded buffer once and reuse it across sweeps.
    V_padded = np.full((rows + 2, cols + 2), np.inf, dtype=float)
    candidates = np.empty((len(offsets), rows, cols), dtype=float)

    iters = 0
    for _ in range(max_iters):
        iters += 1

        # Refresh the padded interior from the current V.
        V_padded[1:1 + rows, 1:1 + cols] = V

        # Build the 8 candidate cost arrays via shifted slices.
        for k, (dr, dc, _) in enumerate(offsets):
            r0 = 1 + dr
            c0 = 1 + dc
            candidates[k] = step_costs[k] + V_padded[r0:r0 + rows, c0:c0 + cols]

        # Bellman update: per-cell min over actions.
        V_new = candidates.min(axis=0)

        # Preserve invariants: goal stays 0, blocked stay inf.
        V_new[goal_r, goal_c] = 0.0
        V_new[blocked] = np.inf

        # Convergence check on cells we actually update.  ``np.inf -
        # np.inf`` would be NaN, so mask first.
        diff_mask = updateable & np.isfinite(V_new) & np.isfinite(V)
        if np.any(diff_mask):
            delta = float(np.max(np.abs(V_new[diff_mask] - V[diff_mask])))
        else:
            delta = np.inf

        # Also count cells that just transitioned from inf to finite —
        # those are real progress, even though the diff is inf.
        newly_reached = int(np.sum(np.isinf(V) & np.isfinite(V_new)))

        grid.values[:] = V_new
        V = grid.values

        if newly_reached == 0 and delta < tol:
            break

    return iters


# ---------------------------------------------------------------------------
# Direction extraction (gradient of V at the centroid)
# ---------------------------------------------------------------------------


def _bilinear_sample(grid: GridMDP, x: float, y: float) -> float:
    """Bilinearly interpolate ``grid.values`` at world point (x, y).

    Returns ``+inf`` if the point lies outside the workspace or if any
    of the four surrounding cell centers is blocked / infinite.  Used
    by ``extract_direction`` to compute finite-difference gradients
    with sub-cell accuracy.
    """
    # Express the query point in "cell-center index space": the value
    # u = (x - x_min)/h - 0.5 lies at integer u when x is at the center
    # of column int(u).
    h = grid.cell_size
    u = (x - grid.x_min) / h - 0.5
    v = (y - grid.y_min) / h - 0.5

    c0 = int(np.floor(u))
    r0 = int(np.floor(v))
    c1 = c0 + 1
    r1 = r0 + 1

    if c0 < 0 or r0 < 0 or c1 >= grid.cols or r1 >= grid.rows:
        return float("inf")

    fc = u - c0
    fr = v - r0

    v00 = grid.values[r0, c0]
    v01 = grid.values[r0, c1]
    v10 = grid.values[r1, c0]
    v11 = grid.values[r1, c1]

    if not (np.isfinite(v00) and np.isfinite(v01)
            and np.isfinite(v10) and np.isfinite(v11)):
        return float("inf")

    # Standard bilinear interpolation.
    return float(
        (1 - fr) * ((1 - fc) * v00 + fc * v01)
        + fr * ((1 - fc) * v10 + fc * v11)
    )


def extract_direction(grid: GridMDP, point: np.ndarray) -> float:
    """Return the preferred direction angle (radians) at a world point.

    Computes the gradient of V by central finite differences using
    bilinearly-interpolated samples one cell-size away from the query
    point in each axis, then returns

        theta_star = atan2(-dV/dy, -dV/dx)

    which is the angle of the **negative** gradient — pointing
    "downhill" on the cost surface, toward the goal along the globally
    shortest path.

    Edge handling
    -------------
    1. If the centroid is inside (or near) an obstacle, the surrounding
       V samples will be inf and the gradient is undefined.  Falls back
       to the direct angle from the point to the goal cell.
    2. If a central difference is undefined (one side inf), tries a
       one-sided difference using the central sample.
    3. If both axes still come out undefined, falls back to the direct
       angle to the goal.

    Args:
        grid:  A GridMDP whose ``values`` have been solved by
               ``value_iteration``.
        point: World coordinates of the query point, shape (2,).

    Returns:
        ``theta_star`` in radians, in ``[-pi, pi]``.
    """
    p = np.asarray(point, dtype=float)
    h = grid.cell_size

    goal_world = grid_to_world(grid, *grid.goal_ij)

    def _fallback() -> float:
        diff = goal_world - p
        return float(np.arctan2(diff[1], diff[0]))

    v_center = _bilinear_sample(grid, p[0], p[1])
    v_xp = _bilinear_sample(grid, p[0] + h, p[1])
    v_xm = _bilinear_sample(grid, p[0] - h, p[1])
    v_yp = _bilinear_sample(grid, p[0], p[1] + h)
    v_ym = _bilinear_sample(grid, p[0], p[1] - h)

    # dV/dx
    if np.isfinite(v_xp) and np.isfinite(v_xm):
        dvdx = (v_xp - v_xm) / (2.0 * h)
    elif np.isfinite(v_xp) and np.isfinite(v_center):
        dvdx = (v_xp - v_center) / h
    elif np.isfinite(v_xm) and np.isfinite(v_center):
        dvdx = (v_center - v_xm) / h
    else:
        dvdx = None

    # dV/dy
    if np.isfinite(v_yp) and np.isfinite(v_ym):
        dvdy = (v_yp - v_ym) / (2.0 * h)
    elif np.isfinite(v_yp) and np.isfinite(v_center):
        dvdy = (v_yp - v_center) / h
    elif np.isfinite(v_ym) and np.isfinite(v_center):
        dvdy = (v_center - v_ym) / h
    else:
        dvdy = None

    if dvdx is None or dvdy is None:
        return _fallback()

    # Degenerate (flat) gradient — happens at the goal cell or in any
    # plateau.  Fall back to the direct goal heading.
    if abs(dvdx) < 1e-12 and abs(dvdy) < 1e-12:
        return _fallback()

    return float(np.arctan2(-dvdy, -dvdx))


# ---------------------------------------------------------------------------
# Phase A + B standalone visual sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build the me595 grid, mark obstacles, run value iteration, and
    # plot the cost-to-go heatmap with the gradient at the starting
    # centroid overlaid.
    import time

    import matplotlib.pyplot as plt

    from me595.scenario import create_triangle_scenario

    scenario = create_triangle_scenario()
    workspace_bounds = (-2.0, -3.0, 14.0, 17.0)  # matches me595/run.py
    goal = scenario["goal"]

    grid = create_grid(workspace_bounds, cell_size=0.25, goal=goal)
    mark_obstacles(grid, scenario["obstacles"], inflation=0.15)

    n_blocked = int(grid.blocked.sum())
    print(
        f"Grid: {grid.rows} rows x {grid.cols} cols "
        f"({grid.rows * grid.cols} cells)"
    )
    print(f"Blocked cells: {n_blocked} ({100 * n_blocked / grid.blocked.size:.1f}%)")
    print(f"Goal cell (row, col) = {grid.goal_ij}")

    t0 = time.perf_counter()
    n_iters = value_iteration(grid)
    t1 = time.perf_counter()
    finite_mask = np.isfinite(grid.values)
    print(f"Value iteration: {n_iters} sweeps in {(t1 - t0) * 1e3:.1f} ms")
    print(
        f"Reachable cells: {int(finite_mask.sum())} / {grid.values.size}  "
        f"V_max={grid.values[finite_mask].max():.3f}"
    )

    # Sample direction at the swarm starting centroid.
    start_positions = np.array([a.position for a in scenario["agents"]])
    start_centroid = start_positions.mean(axis=0)
    theta_start = extract_direction(grid, start_centroid)
    print(
        f"Start centroid = ({start_centroid[0]:.2f}, {start_centroid[1]:.2f})  "
        f"theta* = {np.degrees(theta_start):6.1f} deg"
    )

    # Plot the value field, masking blocked cells so they show through.
    V_plot = np.where(finite_mask, grid.values, np.nan)

    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(
        V_plot,
        origin="lower",
        cmap="viridis",
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
    )
    fig.colorbar(im, ax=ax, label="Cost-to-go V*(s)")

    # Overlay the obstacle silhouette in gray for reference.
    ax.imshow(
        np.where(grid.blocked, 1.0, np.nan),
        origin="lower",
        cmap="gray",
        vmin=0.0, vmax=1.0,
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        alpha=0.6,
    )

    ax.scatter(
        goal[0], goal[1],
        marker="*", s=300, color="gold",
        edgecolors="darkorange", linewidths=1.0, zorder=5, label="Goal",
    )
    ax.scatter(
        start_positions[:, 0], start_positions[:, 1],
        color="tab:blue", s=60, zorder=5, label="Robots",
    )
    ax.scatter(
        start_centroid[0], start_centroid[1],
        color="red", marker="x", s=120, linewidths=2.5, zorder=6,
        label="Centroid",
    )

    # Gradient arrow at the centroid (length = 1 m for visibility).
    arrow_len = 1.0
    ax.annotate(
        "", xy=(
            start_centroid[0] + arrow_len * np.cos(theta_start),
            start_centroid[1] + arrow_len * np.sin(theta_start),
        ),
        xytext=(start_centroid[0], start_centroid[1]),
        arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
        zorder=7,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"me595 grid MDP value function  "
        f"({n_iters} sweeps, theta*={np.degrees(theta_start):.1f} deg)"
    )
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
