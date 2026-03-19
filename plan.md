# PLAN.md

## Goal
Implement the paper in a **Python-only 2D simulation** in small, visible, paper-aligned chunks. The plan should help the user learn the algorithm, not just finish it.

The implementation assumptions for this phase are:
- 2D simulation
- 5 robots
- double-integrator robot model
- adjacency-matrix communication
- perfect knowledge of robot states
- axis-aligned rectangular obstacles
- one `main.py` with command-line modes
- reusable shared modules

---

## High-level paper pipeline
The paper’s implementation will be developed in this order:
1. simulation scaffolding,
2. distributed convex hull consensus,
3. preferred direction consensus,
4. local safe-region construction,
5. consensus intersection of safe regions,
6. formation template and optimization,
7. robot-to-slot assignment,
8. closed-loop robot motion to assigned targets,
9. integrated full-paper demo in Python.

The paper introduces exactly these major distributed ingredients: consensus on the convex hull of robot positions, consensus on a preferred direction of motion through a max-min utility rule, and consensus on a convex free position-time region by intersecting individual regions. fileciteturn3file1L170-L176 fileciteturn3file3L14-L27 The algorithm then computes a target formation inside the agreed region, assigns robots to target slots, and runs a high-frequency control loop toward those targets. fileciteturn3file4L1-L16

---

## Global implementation strategy

### Philosophy
Each phase should produce one of the following:
- a plot,
- an animation,
- a printed consensus result,
- a target formation overlay,
- trajectories.

### One-mode-per-phase style
The repository will use a single `main.py` with modes such as:

```bash
python main.py --mode scaffold
python main.py --mode hull
python main.py --mode direction
python main.py --mode local-free-space
python main.py --mode region-consensus
python main.py --mode formation
python main.py --mode assignment
python main.py --mode control
python main.py --mode full-demo
```

### Expected dependencies
Primary:
- `numpy`
- `matplotlib`
- `scipy`
- `cvxpy`

Optional later only if clearly useful:
- `shapely`

For the first implementation, avoid optional dependencies unless the geometry becomes too cumbersome.

---

## Phase 0 — Simulation scaffold

### Objective
Build the smallest possible reusable simulation environment for 5 robots in 2D.

### Why this comes first
Before implementing any consensus or geometry, the user needs a way to:
- place robots in the plane,
- define an adjacency matrix,
- draw obstacles,
- run repeatable demos.

### Math/Model
For robot \(i\), use the 2D double-integrator model:
\[
\dot p_i = v_i, \qquad \dot v_i = u_i
\]
with discrete simulation step \(\Delta t\):
\[
p_i^{k+1} = p_i^k + \Delta t\, v_i^k,
\qquad
v_i^{k+1} = v_i^k + \Delta t\, u_i^k.
\]

### Deliverables
- `robots/agent.py`
- `robots/dynamics.py`
- `consensus/graph_utils.py`
- `geometry/rectangles.py`
- `plotting/draw_team.py`
- `simulation/scenario.py`
- `main.py --mode scaffold`

### What `--mode scaffold` should show
- 5 robot positions
- velocity arrows
- adjacency graph edges
- one or two rectangular obstacles
- goal point

### Functions to scaffold
- `create_default_scenario()`
- `validate_adjacency_matrix()`
- `get_neighbors()`
- `step_double_integrator()`
- `draw_team_state()`
- `draw_rect_obstacles()`

### Done when
- the plot renders cleanly,
- adjacency is easy to inspect,
- changing robot positions or obstacle rectangles is easy.

---

## Phase 1 — Distributed convex hull consensus

### Objective
Implement the first true paper algorithmic block: all robots reach agreement on the convex hull of robot positions using only local communication.

### Paper idea
The paper begins by having each robot propagate hull information through the communication graph. Each robot starts with only its own position, updates its local convex hull by combining it with neighbors’ hull information, and converges in at most the graph diameter when the graph is connected. fileciteturn3file0L1-L17 fileciteturn3file0L18-L31

### Math summary
Let \(C_i(k)\) be robot \(i\)'s local hull estimate after round \(k\). The paper shows the update idea as:
\[
C_i(k+1) = \operatorname{convhull}(C_i(k), C_j(k))
\]
across neighbor exchanges, and proves convergence to the global convex hull in at most the graph diameter for connected graphs. fileciteturn3file0L1-L17

### Implementation simplification
For the first version:
- store sets of candidate points locally,
- union with neighbor sets,
- recompute hull each round,
- convergence can be checked by equality of hull vertex sets.

### Deliverables
- `geometry/hulls.py`
- `consensus/convex_hull.py`
- `plotting/draw_hulls.py`
- `main.py --mode hull`

### What `--mode hull` should show
Option A:
- all robots in one figure,
- true global hull,
- one selected robot’s current local hull estimate,
- iteration count.

Option B:
- subplots for all robots’ local hull estimates by round.

### Functions to scaffold
- `compute_convex_hull(points)`
- `extract_hull_vertices(points)`
- `run_convex_hull_consensus(points, adjacency, num_rounds)`
- `did_hulls_converge(hull_history)`
- `draw_local_hull_estimates(...)`

### Done when
- every robot ends with the same hull,
- the result matches the centralized hull,
- the user can see convergence round by round.

---

## Phase 2 — Preferred direction consensus

### Objective
Implement the paper’s optional but important max-min consensus step for choosing a preferred direction of motion.

### Paper idea
After the hull is known, the team computes a preferred direction from the hull centroid toward the goal. Because obstacles may block some directions for some robots, the paper considers a discrete set of candidate directions \(\Theta\), lets each robot assign a utility to each direction, then performs a componentwise min-consensus so the team chooses the direction with the best worst-case utility. fileciteturn3file2L27-L41 fileciteturn3file3L1-L27

### Math summary
Each robot defines a utility function \(u_i(\theta)\) over candidate angles. The global utility is
\[
u(\theta) = \min_{i \in I} u_i(\theta)
\]
and the team selects
\[
\theta^* = \arg\max_{\theta \in \Theta} u(\theta).
\]
Distributed update:
\[
\mathbf{u}_i(k+1) = \min_{j \in N_i} \big(\mathbf{u}_i(k), \mathbf{u}_j(k)\big)
\]
with componentwise minimum. fileciteturn3file3L11-L27

### First implementation choice
Use a simple utility function based on distance from the robot to the nearest obstacle boundary along a ray in candidate direction \(\theta\).

This is not the only possible utility, but it is intuitive and visual.

### Deliverables
- `consensus/preferred_direction.py`
- `geometry/free_space.py` (first small helpers only)
- `plotting/draw_direction.py` or extend existing plotting module
- `main.py --mode direction`

### What `--mode direction` should show
- robot positions and obstacles,
- hull centroid,
- goal point,
- candidate rays,
- local per-robot utilities,
- chosen team direction.

### Functions to scaffold
- `compute_hull_centroid(hull_vertices)`
- `sample_candidate_directions(num_angles)`
- `ray_rectangle_intersection_distance(origin, direction, rect)`
- `compute_direction_utility(robot_position, obstacles, theta)`
- `run_min_consensus_on_utilities(local_utilities, adjacency)`
- `select_best_direction(global_utilities)`

### Done when
- all robots agree on the same direction,
- the chosen direction visibly avoids blocked directions better than a naive straight-to-goal direction in the test scenario.

---

## Phase 3 — Local obstacle-free region construction

### Objective
Give each robot a local safe region that encodes where the formation could move without intersecting the obstacles seen by that robot.

### Paper idea
The paper computes convex regions in free position-time space and later intersects them across robots. fileciteturn3file1L170-L176 This is one of the harder parts of the method, so the first implementation should use a teaching-first stepping stone.

### Recommended first simplification
Instead of jumping to a full generic polytope representation in position-time space, start by constructing **simple convex half-space constraints in 2D** around rectangular obstacles for a chosen motion direction and planning horizon.

Possible stepping-stone options:
1. 2D position-only safe corridor approximation,
2. 2D position-time prism approximation,
3. full half-space polytope representation.

Recommended order:
- start with position-only convex safe region,
- then augment with time if needed for paper fidelity.

### Deliverables
- `geometry/free_space.py`
- `geometry/polygons.py`
- `main.py --mode local-free-space`

### What `--mode local-free-space` should show
- one robot’s local obstacle perception,
- its resulting local convex safe region,
- the current hull / centroid / preferred direction overlaid for context.

### Functions to scaffold
- `build_local_free_region(robot_state, obstacles, preferred_direction, horizon)`
- `polygon_from_halfspaces(...)`
- `clip_region_with_obstacle_constraints(...)`
- `draw_local_free_region(...)`

### Done when
- each robot can compute a local region,
- the region is convex and drawable,
- the output changes in sensible ways when obstacles move.

---

## Phase 4 — Consensus intersection of local safe regions

### Objective
Have all robots agree on a common safe region by intersecting their local regions through distributed communication.

### Paper idea
A central part of the paper is distributed consensus on a convex free region in position-time space, obtained by intersecting individual regions. fileciteturn3file1L170-L176 In Python, a first version can represent regions explicitly with half-space matrices or polygon vertices.

### Representation choice
Prefer half-space form when practical:
\[
P = \{x \mid Ax \le b\}
\]
because intersection is then just stacking constraints.

### Deliverables
- `geometry/free_space.py`
- `consensus/region_consensus.py` or extend `free_space.py`
- `main.py --mode region-consensus`

### What `--mode region-consensus` should show
- each robot’s local safe region,
- the agreed common region,
- any regions that are empty or overly restrictive.

### Functions to scaffold
- `region_to_halfspaces(region)`
- `intersect_halfspace_regions(region_a, region_b)`
- `run_region_consensus(local_regions, adjacency, num_rounds)`
- `is_region_empty(region)`

### Done when
- all robots converge to the same region,
- the agreed region is the same as the centralized intersection,
- the region can be visualized reliably.

---

## Phase 5 — Formation templates and formation optimization

### Objective
Compute a target formation that fits inside the agreed safe region and is oriented according to the preferred direction.

### Paper idea
Once the common region is known, the paper solves a formation optimization problem to find the best template and configuration. fileciteturn3file4L1-L8 In the Python learning version, start with a small library of templates.

### First template library
Use 2D versions of:
- line,
- V-shape,
- pentagon,
- rectangle-like formation.

### First optimization scope
Optimize a reduced parameter vector such as:
- formation center,
- scale,
- rotation.

If needed later, extend to additional shape parameters.

### Deliverables
- `formation/templates.py`
- `formation/optimizer.py`
- `main.py --mode formation`

### What `--mode formation` should show
- agreed safe region,
- candidate formations,
- best formation selected,
- target slot positions.

### Functions to scaffold
- `generate_template_points(template_name, num_robots)`
- `apply_similarity_transform(template_points, center, scale, angle)`
- `formation_constraints_within_region(template_points, region)`
- `solve_best_formation(region, preferred_direction, templates)`

### Done when
- a valid formation is found inside the region,
- the selected result is visually reasonable,
- changing the obstacle layout changes the chosen formation or scale.

---

## Phase 6 — Robot-to-slot assignment

### Objective
Assign the current robots to target formation slots.

### Paper idea
The paper formulates an assignment problem minimizing the sum of squared traveled distances to formation slots. fileciteturn3file4L8-L16 For the Python phase, a centralized solver is acceptable.

### Math summary
Given current robot positions \(p_i\) and target slot positions \(r_j^*\), solve:
\[
\min_X \sum_i \sum_j x_{ij}\|p_i-r_j^*\|^2
\]
where \(X\) is a permutation matrix. fileciteturn3file4L8-L16

### Deliverables
- `formation/assignment.py`
- `main.py --mode assignment`

### What `--mode assignment` should show
- current robot positions,
- target slot positions,
- assignment lines,
- total assignment cost.

### Functions to scaffold
- `compute_assignment_cost_matrix(robot_positions, target_positions)`
- `solve_assignment(cost_matrix)`
- `draw_assignment(robot_positions, target_positions, assignment)`

### Done when
- each robot gets one slot,
- visual lines match intuition,
- assignment cost is printed.

---

## Phase 7 — Closed-loop motion with double-integrator robots

### Objective
Move the robots to their assigned targets using the double-integrator model.

### First controller
Use a simple PD tracking law:
\[
u_i = -K_p(p_i-r_i^*) - K_d(v_i-v_i^*)
\]
with \(v_i^*=0\) for the first version.

### Deliverables
- `robots/dynamics.py`
- `simulation/runner.py`
- `main.py --mode control`

### What `--mode control` should show
- animated trajectories,
- current robot positions,
- target slot positions,
- velocity vectors,
- optional collision checks.

### Functions to scaffold
- `compute_pd_acceleration(state, target_position, gains)`
- `simulate_team_to_targets(team_state, target_positions, controller, dt, steps)`
- `check_robot_robot_collisions(...)`
- `check_robot_obstacle_collisions(...)`

### Done when
- robots visibly converge toward targets,
- trajectories are smooth,
- no obvious numerical instability occurs.

---

## Phase 8 — Full integrated demo

### Objective
Run the full Python pipeline in one scenario:
1. hull consensus,
2. preferred direction consensus,
3. local region generation,
4. region intersection,
5. formation optimization,
6. assignment,
7. closed-loop motion.

### Deliverables
- `main.py --mode full-demo`
- scenario presets with at least one moving-obstacle example

### What `--mode full-demo` should show
- step-by-step or staged animation,
- printed summaries for each algorithm block,
- optional pause between stages.

### Done when
- the whole paper pipeline is visible in one run,
- each step can still be run independently,
- the codebase is still readable.

---

## Suggested `main.py` CLI structure

```python
parser.add_argument("--mode", type=str, required=True)
parser.add_argument("--scenario", type=str, default="default")
parser.add_argument("--rounds", type=int, default=None)
parser.add_argument("--dt", type=float, default=0.1)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--animate", action="store_true")
```

Suggested modes:
- `scaffold`
- `hull`
- `direction`
- `local-free-space`
- `region-consensus`
- `formation`
- `assignment`
- `control`
- `full-demo`

---

## Recommended first coding session
The first session should only cover **Phase 0** and set up placeholders for **Phase 1**.

### Session 1 tasks
1. Create the folder structure.
2. Implement `create_default_scenario()`.
3. Implement adjacency validation and neighbor lookup.
4. Implement 2D robot state container.
5. Implement a basic plotting function for robots, graph edges, obstacles, and goal.
6. Add `python main.py --mode scaffold`.
7. Add stub files and stub functions for hull consensus.

### Why this is the right first step
It is small, visible, and immediately useful. It also prepares the exact data structures needed for the first real paper step: convex hull consensus.

---

## Recommended second coding session
The second session should implement **Phase 1** fully.

### Session 2 tasks
1. Add centralized convex hull helper.
2. Add local robot hull storage.
3. Implement one consensus round.
4. Implement repeated rounds until convergence.
5. Plot local hull evolution.
6. Add `python main.py --mode hull`.

---

## Learning checklist for each phase
For every phase, the user should be able to answer:
1. What information does each robot store locally?
2. What does each robot send to its neighbors?
3. What update rule is applied each round?
4. What object is the team converging to?
5. How can I verify correctness visually?
6. What module will reuse this result later?

---

## Stretch goals for later, but not now
Do not implement these yet unless requested:
- noisy localization,
- asynchronous communication,
- packet drops,
- 3D extension,
- ROS2 nodes,
- Gazebo integration,
- real quadrotor dynamics,
- distributed assignment solver,
- dynamic obstacle prediction beyond a simple first version.

---

## Immediate next step
Start by implementing **Phase 0: Simulation scaffold** and create enough shared structure that **Phase 1: convex hull consensus** can be added cleanly in the next iteration.
