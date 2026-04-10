# PLAN.md — Value Iteration Path Planning for me595

## Problem Statement

The current me595 closed-loop runner gets stuck. The preferred-direction
consensus (Phase 2) selects the direction with the best one-step ray
clearance weighted toward the goal. This is **myopic**: it cannot reason
about multi-step detours. When the merged triangle+rectangle obstacle
blocks the direct path to the goal, the algorithm locks onto a direction
with high clearance (away from the obstacle) that makes no progress.

**Observed failure**: the swarm starts at ~(1, 4), needs to reach (12, 3).
The obstacle spans x in [1, 8], y in [-2, 15.66]. The greedy direction
consensus picks 258.8 deg (south-southwest — lots of clearance, moderate
goal weight) and the centroid stays fixed at (2.67, 7.57) for all 20
iterations.

**Required fix**: replace the myopic direction selection with a planner
that can see "go south around the bottom of the obstacle, then east to the
goal" — a path that requires committing to a direction that temporarily
moves away from the goal.

---

## Chosen Algorithm: Value Iteration on a Grid MDP

Reference: Kochenderfer, "Algorithms for Decision Making", Chapter 7
(Exact Solution Methods), Section 7.4 (Value Iteration).

### Core idea

Discretize the 2D workspace into a grid. Each cell is a **state** in an
MDP. Actions are moves to the 8 neighboring cells (N, NE, E, SE, S, SW, W,
NW). Obstacle cells are impassable. The goal cell has cost 0; every move
costs its Euclidean distance (1.0 for cardinal, sqrt(2) for diagonal).

**Value iteration** (Bellman updates) computes V(s) = minimum cost-to-go
from every cell s to the goal:

    V(s) <- min_a [ cost(s, a) + V(s') ]

where s' is the cell reached by action a from s. Impassable cells get
V = infinity. After convergence, V encodes the globally optimal distance
to the goal around all obstacles.

### Direction extraction

At the hull centroid c, the preferred direction theta_star is extracted
from the gradient of V:

    grad_V = ( V(c + dx) - V(c - dx),  V(c + dy) - V(c - dy) )  /  (2 * cell_size)
    theta_star = atan2(-grad_V[1], -grad_V[0])

This points "downhill" on the value surface — toward the goal along the
globally shortest path. The gradient is estimated by finite differences on
the grid, with bilinear interpolation for non-grid-aligned centroids.

### Distributed framing

The distributed structure mirrors the existing utility consensus:

1. Each robot i computes V_i on the full grid using only its **locally
   visible obstacles** (those within sensing_range).
2. Robots run **max-consensus** on V vectors:
       V_i(k+1) = max_{j in N_i} ( V_i(k), V_j(k) )   element-wise.
   Higher V = higher cost = more conservative. Taking the max respects
   every robot's obstacle knowledge.
3. After d rounds (graph diameter), all robots agree on V_agreed.
4. Each robot extracts theta_star from grad(V_agreed) at the centroid.

In the current me595 scenario all robots have sensing_range=20 and see
every obstacle, so the consensus is a no-op. But the code structure
supports the general case where robots have limited sensing.

### Why this works

Value iteration "floods" cost outward from the goal. The obstacle creates
a cost shadow: cells behind it have high V because the shortest path goes
around. The gradient of V at any point gives the locally optimal step
toward the goal. Even at (2.67, 7.57) — where the greedy planner gets
stuck — the VI gradient will point south-southeast (toward the bottom of
the triangle), because that is the start of the shortest path to (12, 3).

---

## Integration with the Existing Pipeline

Only **Phase 2** (direction selection) changes. Phases 1, 3-7 are
untouched:

| Phase | Current (greedy) | New (VI) |
|-------|-----------------|----------|
| 1. Hull consensus | unchanged | unchanged |
| 2. Direction | ray-clearance min-consensus | gradient of V from grid MDP |
| 3. Local free regions | unchanged | unchanged |
| 4. Region consensus | unchanged | unchanged |
| 5. Formation optimizer | unchanged | unchanged |
| 6. Assignment | unchanged | unchanged |
| 7. PD control + projection | unchanged | unchanged |

The `--planner` flag in `me595/run.py` selects between `greedy` (current)
and `vi` (new). Both paths produce a single float `theta_star` that the
rest of the pipeline consumes.

---

## Preserving the Current Algorithm

The user needs to demonstrate before/after. The current greedy planner
must remain runnable:

    ./env/bin/python -m me595.run --planner greedy     # current (gets stuck)
    ./env/bin/python -m me595.run --planner vi          # new (navigates around)
    ./env/bin/python -m me595.run                       # default = vi

No existing code should be deleted. The greedy path in `me595/geometry.py`
(`run_direction_consensus`, `compute_local_utilities`, `ray_clearance`)
stays intact.

---

## Implementation Phases

### Phase A: Grid MDP Infrastructure

**New file**: `me595/grid_mdp.py`

Build the grid representation and obstacle marking.

#### Data structures

```python
@dataclass
class GridMDP:
    x_min: float          # workspace left edge
    y_min: float          # workspace bottom edge
    x_max: float          # workspace right edge
    y_max: float          # workspace top edge
    cell_size: float      # meters per cell (default 0.25)
    blocked: np.ndarray   # bool array (rows, cols) — True = impassable
    values: np.ndarray    # float array (rows, cols) — V(s), inf for blocked
    goal_ij: tuple[int, int]  # grid indices of the goal cell
```

#### Functions to implement

1. `create_grid(workspace_bounds, cell_size) -> GridMDP`
   - Allocate the grid arrays.
   - Initialize `values` to inf everywhere except the goal cell (0).
   - `blocked` starts all-False.

2. `mark_obstacles(grid, obstacles) -> None`
   - For each cell, test if its center is inside any obstacle.
   - Use `polygon_contains_point` from `me595/geometry.py`.
   - Optionally inflate obstacles by a safety margin (half the robot
     inter-collision radius, ~0.15 m).
   - Set `blocked[i, j] = True` and `values[i, j] = inf`.

3. `world_to_grid(grid, point) -> (int, int)`
   - Convert world (x, y) to grid row, col indices.

4. `grid_to_world(grid, i, j) -> np.ndarray`
   - Convert grid (row, col) back to world (x, y).

#### Done when

- A grid can be created for the me595 workspace.
- Obstacle cells match the triangle + rectangle footprint.
- A simple matplotlib `imshow` of `blocked` looks correct.

---

### Phase B: Value Iteration Solver

**Same file**: `me595/grid_mdp.py`

Implement the Bellman update loop.

#### Functions to implement

1. `neighbors_8(i, j, rows, cols) -> list[tuple[int, int, float]]`
   - Return up-to-8 neighbor cells and their step costs.
   - Cardinal neighbors: cost = cell_size.
   - Diagonal neighbors: cost = cell_size * sqrt(2).
   - Skip out-of-bounds cells.

2. `value_iteration(grid, tol=1e-3, max_iters=500) -> int`
   - Standard synchronous Bellman update:
     ```
     for each non-blocked cell (i, j):
         V_new[i,j] = min over neighbors (r,c,cost):
             cost + V_old[r, c]   if not blocked[r, c]
     ```
   - Iterate until max change < tol or max_iters reached.
   - Update `grid.values` in-place.
   - Return the number of iterations used.

3. `extract_direction(grid, point) -> float`
   - Given a world-coordinate point (the hull centroid), compute
     theta_star from the gradient of V using finite differences.
   - Use bilinear interpolation for sub-cell accuracy.
   - Return the angle in radians.

#### Performance note

For cell_size=0.25, the grid is 64x80 = 5120 cells. Value iteration on
this grid converges in ~200 iterations, each sweeping 5120 cells.
Total: ~1M cell updates — well under 1 second in NumPy. No need for
compiled code.

#### Done when

- `value_iteration` converges on the me595 grid.
- V values near the goal are low, V values behind the obstacle are high.
- `extract_direction` at (2.67, 7.57) returns a south-ish angle (toward
  the bottom of the triangle), not 258.8 deg.
- A heatmap of V with the obstacle overlaid looks like a distance field
  that wraps around the obstacle.

---

### Phase C: Distributed Value Consensus

**New file**: `me595/value_planner.py`

Wrap the grid MDP in a distributed-consensus-compatible interface.

#### Functions to implement

1. `compute_local_value_function(obstacles, workspace_bounds, goal, cell_size, sensing_range=None) -> GridMDP`
   - Create grid, mark obstacles, run VI, return the solved grid.
   - If sensing_range is provided, only mark obstacles within that range
     of the grid center (for future partial-observability support).

2. `run_value_consensus(local_grids, adjacency, num_rounds) -> GridMDP`
   - Each robot has a local GridMDP with its own `values` array.
   - Max-consensus on the flattened value vectors:
     ```
     V_i(k+1) = element-wise max over j in N_i of V_j(k)
     ```
   - After num_rounds, all grids agree. Return the agreed grid.
   - (When all robots see the same obstacles, this is an identity
     operation. The code still runs it for structural consistency.)

3. `plan_direction(grid, centroid) -> float`
   - Thin wrapper: call `extract_direction(grid, centroid)`.
   - Returns theta_star.

#### Done when

- `plan_direction` at the me595 starting centroid returns a direction
  that, if followed iteratively, traces a path around the obstacle
  bottom toward (12, 3).
- The distributed consensus produces the same result as a single robot
  running VI alone (since all robots see everything).

---

### Phase D: Integration with run.py

**Modify**: `me595/run.py`

Add the `--planner` CLI flag and wire the VI planner into the main loop.

#### Changes

1. Add argument: `--planner {greedy,vi}` with default `vi`.

2. If `planner == "vi"`:
   - Before the main loop, call `compute_local_value_function` once for
     each robot. Run `run_value_consensus` to get `agreed_grid`.
   - Since obstacles are static, this only needs to happen **once**
     (not every iteration). The value function doesn't change.
   - In the main loop, replace the direction-consensus block (Phase 2)
     with:
     ```python
     theta_star = plan_direction(agreed_grid, centroid)
     ```

3. If `planner == "greedy"`:
   - Run the existing `run_direction_consensus` + `select_preferred_direction`
     code, unchanged.

4. Print the planner type in the header line:
   ```
   me595 closed-loop run  |  N=5 robots  |  planner=vi  |  diameter=2
   ```

#### Done when

- `./env/bin/python -m me595.run --planner greedy` reproduces the stuck
  behavior (centroid at ~(2.67, 7.57), 20 iterations, no progress).
- `./env/bin/python -m me595.run --planner vi` navigates the swarm
  around the bottom of the obstacle toward (12, 3).
- `./env/bin/python -m me595.run` defaults to vi.

---

### Phase E: Visualization and Testing

Add diagnostic plots so the user can see what VI is doing.

#### Additions

1. **Value function heatmap**: add a `--show-value-map` flag to
   `me595/run.py` that displays the V grid as a color map with the
   obstacle silhouette, goal marker, and the gradient-arrow at the
   centroid overlaid. This is shown once before the main loop starts.

2. **Path preview**: on the heatmap, trace the gradient-descent path
   from the starting centroid to the goal. This is the "ideal" centroid
   trajectory that VI would produce if the formation were a point.

3. **Direction arrow in animation**: during the animated run, draw a
   small arrow at the centroid showing theta_star for each iteration.
   Color it by planner type (blue for VI, red for greedy) so the
   before/after comparison is visually obvious.

#### Done when

- The heatmap clearly shows the cost-to-go wrapping around the obstacle.
- The gradient-descent path visually goes around the obstacle bottom.
- The animated run with `--planner vi` shows the swarm tracking that path.

---

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| `me595/grid_mdp.py` | NEW | Grid MDP: creation, obstacle marking, value iteration, direction extraction |
| `me595/value_planner.py` | NEW | Distributed VI wrapper: local V computation, max-consensus, plan_direction |
| `me595/run.py` | MODIFY | Add `--planner {greedy,vi}` flag, wire VI into Phase 2 |
| `me595/geometry.py` | UNCHANGED | Greedy planner code stays for `--planner greedy` |
| `me595/dynamics.py` | UNCHANGED | Velocity projection unchanged |
| `me595/scenario.py` | UNCHANGED | Scenario unchanged |

---

## Suggested Implementation Order

1. **Phase A** first — get the grid and obstacle marking working, verify
   with an imshow plot.
2. **Phase B** next — implement VI and verify the value heatmap looks
   correct (smooth cost field wrapping around obstacle).
3. **Phase C** — wrap in distributed consensus interface.
4. **Phase D** — wire into run.py with `--planner` flag.
5. **Phase E** — add visualization, run both planners, verify before/after.

Each phase should be testable independently. Phase A+B can be verified
with a standalone script that creates the grid, runs VI, and shows the
heatmap. Phase C+D add the integration. Phase E is polish.

---

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `cell_size` | 0.25 | 64x80 grid, fast VI, sufficient resolution |
| `vi_tol` | 1e-3 | Convergence tolerance for Bellman updates |
| `vi_max_iters` | 500 | Safety cap; should converge in ~200 |
| `obstacle_inflation` | 0.15 | Half the inter-robot safety margin |
| `move_cost_cardinal` | cell_size | Cost of N/S/E/W step |
| `move_cost_diagonal` | cell_size * sqrt(2) | Cost of NE/SE/SW/NW step |

---

## Expected Behavior After Implementation

Starting configuration: 5 robots clustered around (1, 4). Goal at (12, 3).
Obstacle: merged triangle+rectangle spanning x in [1, 8], y in [-2, 15.66].

**With `--planner greedy`** (before):
- theta_star locks to 258.8 deg after iteration 1
- Centroid stuck at (2.67, 7.57)
- Never reaches goal

**With `--planner vi`** (after):
- theta_star at the starting position points ~south-southeast (toward the
  gap below the triangle at y = -2)
- As the centroid moves south past the triangle bottom, theta_star rotates
  to east
- As the centroid clears x = 8, theta_star rotates to north-northeast
  toward (12, 3)
- Swarm reaches goal within ~10-15 iterations

The formation may need to shrink when passing through the narrow gap below
the triangle (y between -3 and -2). The existing formation optimizer
handles this automatically — it maximizes scale within the agreed free
region, which will be small near the obstacle.
