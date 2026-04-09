# CLAUDE.md — Agent Instructions for me595 Value Iteration Planner

## Context

This repo implements a 2D distributed formation control algorithm (Alonso-
Mora 2019). There are two implementations:

- `python/` — the original teaching implementation with axis-aligned
  rectangular obstacles. Fully working, do not modify.
- `me595/` — an alternate scenario with a 30-60-90 triangle merged with a
  tall rectangle obstacle. Has polygon-aware geometry and dynamics. The
  closed-loop runner (`me595/run.py`) works end-to-end but gets stuck
  because the direction selection is myopic (one-step ray clearance).

**Your task**: implement Value Iteration on a Grid MDP to replace the
direction selection (Phase 2) in the me595 pipeline. Read `plan.md` for
the full design.

---

## What to Build

Read `plan.md` thoroughly before writing any code. It contains the
complete algorithm specification, data structures, function signatures,
and integration plan.

### New files to create

1. **`me595/grid_mdp.py`** — Grid MDP infrastructure and value iteration
   solver. Contains: `GridMDP` dataclass, `create_grid`,
   `mark_obstacles`, `value_iteration`, `extract_direction`, and helpers.

2. **`me595/value_planner.py`** — Distributed wrapper that connects the
   grid MDP to the formation control pipeline. Contains:
   `compute_local_value_function`, `run_value_consensus`,
   `plan_direction`.

### File to modify

3. **`me595/run.py`** — Add `--planner {greedy,vi}` argument (default
   `vi`). When `planner == "vi"`, compute the value function once before
   the main loop, then use `plan_direction(grid, centroid)` in place of
   the direction consensus block. When `planner == "greedy"`, run the
   existing code unchanged.

### Files NOT to modify

- `me595/geometry.py` — greedy planner code, keep for `--planner greedy`
- `me595/dynamics.py` — velocity projection, unchanged
- `me595/scenario.py` — scenario definition, unchanged
- `me595/triangle.py`, `me595/rectangle.py`, `me595/polygon.py` — shapes
- Everything under `python/` — do not touch

---

## Critical Design Decisions

### Grid parameters

- `cell_size = 0.25` (64x80 grid for the me595 workspace)
- Workspace bounds: `(-2.0, -3.0, 14.0, 17.0)` — match `WORKSPACE_BOUNDS`
  in `me595/run.py`
- Inflate obstacles by 0.15 units when marking blocked cells

### Value iteration

- 8-connected grid (cardinal + diagonal moves)
- Move cost = Euclidean distance (1.0 * cell_size for cardinal,
  sqrt(2) * cell_size for diagonal)
- Goal cell: V = 0. Blocked cells: V = inf. All others: V = inf initially.
- Synchronous Bellman updates until max change < 1e-3 or 500 iterations
- The VI runs **once** before the main loop (obstacles are static)

### Direction extraction

- At the centroid, compute the gradient of V by finite differences
- Use bilinear interpolation for sub-cell positions
- `theta_star = atan2(-dV/dy, -dV/dx)` — negative gradient points toward
  lower cost (toward the goal)
- Handle edge cases: if centroid is inside an obstacle or gradient is
  zero, fall back to direct angle toward goal

### Distributed consensus on V

- Each robot computes V locally with its visible obstacles
- Max-consensus on flattened V arrays (higher V = more conservative):
  `V_i(k+1) = element-wise max over j in N_i of V_j(k)`
- In the current scenario all robots see all obstacles, so consensus is
  a no-op. Still run it for structural consistency.

### Preserving the greedy planner

- The `--planner greedy` path must produce **identical output** to the
  current `me595/run.py`. Do not refactor the greedy code path.
- Default planner should be `vi`.

---

## How the Pipeline Uses theta_star

Only Phase 2 changes. The rest of the pipeline consumes `theta_star`
(a single float, radians) exactly as before:

```python
# Phase 2 produces theta_star (either greedy or VI)

# Phase 3: compute_local_free_region(..., theta_star=theta_star, tau=tau)
# Phase 5: optimize_formation(..., theta_star, centroid=centroid, tau=effective_tau)
# Phase 7: PD control toward assigned slots
```

Do not change any function signatures in the pipeline. The VI planner's
only job is to produce a better `theta_star`.

---

## Coding Standards

- Python 3, type hints, descriptive variable names
- Use `numpy` for array operations, `dataclasses` for data structures
- Every function needs a clear docstring explaining inputs, outputs,
  algorithm, and math reference
- Include comments explaining **why**, not just restating code
- Validate array shapes at function entry
- No new dependencies beyond numpy/matplotlib (no scipy needed for VI)
- Keep functions short and single-purpose
- Match the style of existing me595/ code (see `geometry.py` for reference)

---

## Testing Strategy

### Unit tests (informal, run manually)

After Phase A+B, create a small test at the bottom of `grid_mdp.py`:

```python
if __name__ == "__main__":
    # Build grid, mark obstacles, run VI, show heatmap
    from me595.scenario import create_triangle_scenario
    scenario = create_triangle_scenario()
    grid = create_grid((-2, -3, 14, 17), cell_size=0.25)
    mark_obstacles(grid, scenario["obstacles"])
    n_iters = value_iteration(grid)
    print(f"VI converged in {n_iters} iterations")
    # Show heatmap
    import matplotlib.pyplot as plt
    plt.imshow(grid.values, origin="lower", cmap="viridis",
               extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max])
    plt.colorbar(label="Cost-to-go")
    plt.title("Value function")
    plt.show()
```

### Integration test

```bash
# Greedy (should get stuck — same output as before)
./env/bin/python -m me595.run --planner greedy --max-iters 5

# VI (should make progress toward goal)
./env/bin/python -m me595.run --planner vi

# Default (should use VI)
./env/bin/python -m me595.run
```

### Visual verification

- The value function heatmap should show a smooth distance field that
  wraps around the obstacle. Low values (dark) near the goal, high
  values (bright) behind the obstacle.
- `extract_direction` at (1.0, 4.12) should return a south-ish angle
  (toward the gap below the triangle), NOT east toward the goal.
- The animated run with `--planner vi` should show the swarm curving
  around the bottom of the obstacle.

---

## Known Gotchas

1. **Grid indexing**: row 0 = y_min (bottom), not y_max. Use `origin="lower"`
   in imshow. The mapping is:
   - `row = int((y - y_min) / cell_size)`
   - `col = int((x - x_min) / cell_size)`

2. **Obstacle containment test**: use `polygon_contains_point` from
   `me595/geometry.py` for both Triangle and Rectangle obstacles. Don't
   write a separate containment test.

3. **Inf handling in VI**: blocked cells stay at inf forever. When
   computing the Bellman update, skip neighbors that are blocked (their
   V = inf would propagate). Use `np.isinf` checks.

4. **Gradient at grid boundary**: if the centroid is near the edge of the
   grid, one-sided finite differences may be needed. Handle this.

5. **The narrow gap**: the bottom of the triangle is at y = -2,
   workspace y_min = -3. That's only 1 unit (4 cells at 0.25 resolution)
   of clearance. With 0.15 inflation, it's ~0.7 units. The formation
   will need to shrink to fit. The existing optimizer handles this — it
   maximizes scale within the agreed free region. But if the gap is too
   narrow for any valid pentagon formation, `optimize_formation` returns
   None. Handle this in run.py (skip the iteration and continue, or
   reduce tau).

6. **VI convergence**: for a 64x80 grid, VI should converge in ~200
   iterations, well under 1 second. If it takes much longer, check that
   blocked cells are properly skipped in updates.

---

## Repo Structure (relevant files)

```
me595/
    __init__.py          # package marker
    scenario.py          # create_triangle_scenario()
    triangle.py          # Triangle dataclass
    rectangle.py         # Rectangle dataclass
    polygon.py           # Polygon dataclass (visual only)
    geometry.py          # polygon-aware free-space, ray clearance, direction consensus
    dynamics.py          # polygon-aware velocity projection
    draw_map.py          # matplotlib scenario drawing
    run.py               # closed-loop runner (MODIFY: add --planner flag)
    grid_mdp.py          # NEW: grid MDP + value iteration
    value_planner.py     # NEW: distributed VI wrapper + direction extraction

python/                  # original implementation — DO NOT MODIFY
    consensus/
        convex_hull.py
        preferred_direction.py
        region_consensus.py
        graph_utils.py
    formation/
        optimizer.py     # optimize_formation (used by me595/run.py)
        assignment.py    # solve_assignment (used by me595/run.py)
        templates.py     # TEMPLATES["pentagon"]
    robots/
        agent.py         # Agent dataclass
        dynamics.py      # step_double_integrator, compute_pd_acceleration, project_velocity
    geometry/
        free_space.py
        rectangles.py
    plotting/
        draw_team.py     # ROBOT_COLORS
        draw_obstacles.py # draw_free_region
```

---

## Definition of Done

The implementation is complete when:

1. `./env/bin/python -m me595.run --planner vi` navigates the swarm
   around the obstacle and reaches the goal (centroid within 1.0 of
   (12, 3)).
2. `./env/bin/python -m me595.run --planner greedy` still reproduces the
   stuck behavior (identical to current output).
3. The animated trajectory with `--planner vi` visually shows the swarm
   curving around the bottom of the obstacle.
4. A value function heatmap can be displayed (via `--show-value-map` or
   by running `grid_mdp.py` directly).
5. All existing me595 code still works unchanged.
