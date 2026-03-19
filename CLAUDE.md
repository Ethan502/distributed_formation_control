# AGENTS.md

## Purpose
This repository is for a **teaching-first Python implementation** of the paper on distributed formation control in dynamic environments. The goal is not to jump straight to a finished simulator. The goal is to learn the paper by building it in **small, runnable chunks** with visible progress at each step.

The implementation target for now is:
- **2D only**
- **5 robots**
- **double-integrator robot model**
- **adjacency-matrix communication**
- **perfect state knowledge**
- **axis-aligned rectangular obstacles**
- **single `main.py` with command-line modes**
- **shared modules that grow over time**

Do **not** introduce ROS2, Gazebo, 3D dynamics, or sensor noise unless the user explicitly asks for it later.

---

## Core development philosophy

### 1) Teach before optimizing
When adding a new chunk of the algorithm:
1. Explain the math and the purpose of the chunk first.
2. Keep the first implementation small and direct.
3. Prefer correctness and readability over cleverness.
4. Add plots or printed diagnostics so the user can see what changed.

### 2) Small milestones only
Each coding task should be small enough that the user can:
- understand the equations being implemented,
- fill in the function bodies themselves,
- run `main.py` in a specific mode,
- visually confirm progress.

### 3) Preserve a paper-first ordering
Stay aligned with the paper’s high-level pipeline:
1. distributed convex hull consensus,
2. preferred direction consensus,
3. obstacle-free convex region construction,
4. formation optimization,
5. assignment to formation slots,
6. motion/control simulation.

Warm-up utilities are allowed, but the plan should always return to this pipeline quickly.

### 4) Keep modules reusable
The repository should evolve around shared modules rather than duplicating logic in many scripts. `main.py` should switch between modes and demos, while the actual logic lives in reusable files.

---

## What the implementation should mirror from the paper
The implementation should reflect these major ideas from the paper:
- robots share local information over a graph,
- the team reaches consensus on the convex hull of robot positions,
- the team reaches consensus on a preferred direction of motion using a max-min utility rule,
- each robot computes a local safe region in position-time space,
- the team intersects those local regions through consensus,
- all robots solve the same formation optimization using the agreed region,
- robots are assigned to target slots,
- a lower-level simulation moves robots toward assigned targets.

For now, keep the implementation in 2D even if the paper is written in 3D terms. Use the same structure but simplify dimensions.

---

## Required coding style

### General rules
- Use Python 3.
- Prefer `numpy`, `matplotlib`, `scipy`, and `cvxpy` when needed.
- Avoid adding new dependencies unless they clearly reduce complexity.
- Use type hints where practical.
- Use descriptive variable names.
- Keep functions short and single-purpose.
- Avoid hidden state when possible.

### Function header policy
This project is intended to help the user learn by filling in parts of the implementation. Therefore:
- Every nontrivial function should have a **clear docstring**.
- Every function should include **descriptive comments** about:
  - inputs,
  - outputs,
  - algorithm idea,
  - math reference when relevant,
  - expected edge cases.
- It is acceptable, and often preferred, to create **stubs/TODOs** for functions that the user will fill in.

### Comment style
Comments should explain **why**, not just restate code.
Good comment:
> Exchange only newly discovered hull vertices so the message passing matches the paper’s communication-saving idea.

Bad comment:
> Increment k by 1.

### Error handling
- Validate shapes of arrays.
- Validate adjacency matrix symmetry when an undirected graph is assumed.
- Fail with informative error messages.

---

## Repository architecture
Use this structure unless there is a strong reason to change it:

```text
.
├── AGENTS.md
├── PLAN.md
├── README.md                  # optional later
├── main.py
├── config.py
├── robots/
│   ├── __init__.py
│   ├── agent.py
│   ├── team.py
│   └── dynamics.py
├── consensus/
│   ├── __init__.py
│   ├── convex_hull.py
│   ├── preferred_direction.py
│   └── graph_utils.py
├── geometry/
│   ├── __init__.py
│   ├── hulls.py
│   ├── polygons.py
│   ├── rectangles.py
│   └── free_space.py
├── formation/
│   ├── __init__.py
│   ├── templates.py
│   ├── optimizer.py
│   └── assignment.py
├── simulation/
│   ├── __init__.py
│   ├── scenario.py
│   ├── runner.py
│   └── logging_utils.py
└── plotting/
    ├── __init__.py
    ├── draw_team.py
    ├── draw_hulls.py
    ├── draw_obstacles.py
    └── animate.py
```

This structure may be introduced gradually. At the beginning, it is okay if many of these modules are only partial.

---

## `main.py` policy
There should be a **single** `main.py` that selects a mode from the command line.

Suggested style:
```bash
python main.py --mode hull
python main.py --mode direction
python main.py --mode free-space
python main.py --mode formation
python main.py --mode assignment
python main.py --mode closed-loop
```

Each mode should:
1. create or load a scenario,
2. run one small algorithmic chunk,
3. display something notable,
4. print a short summary of what happened.

Do not build giant all-in-one demos too early.

---

## Implementation standards by module

### `robots/agent.py`
Should define a robot agent with at least:
- identifier,
- position,
- velocity,
- current target,
- local neighbor list or adjacency access,
- local storage for consensus variables.

For now the robot model is a **2D double integrator**.
That means the state should usually look like:
\[
 x_i = [p_{x,i},\ p_{y,i},\ v_{x,i},\ v_{y,i}]^\top
\]
with acceleration input.

### `consensus/graph_utils.py`
Should provide utilities for:
- validating adjacency matrices,
- extracting neighbors,
- computing graph diameter when needed,
- checking connectedness.

### `consensus/convex_hull.py`
Should implement the distributed convex hull consensus process in a way that mirrors the paper:
- initialize each robot with its own point,
- exchange only newly discovered hull points if desired,
- update local hull estimate,
- repeat until convergence or graph-diameter rounds.

### `consensus/preferred_direction.py`
Should implement the paper’s max-min consensus over a discrete set of candidate angles:
- each robot computes local utility for each angle,
- robots take componentwise minima across neighbors,
- after convergence, each robot selects the angle with best global worst-case utility.

### `geometry/free_space.py`
Should start simple. For the first version:
- treat obstacles as axis-aligned rectangles,
- represent local safe information as geometry that can be visualized and debugged,
- do not rush into full complexity if a simpler stepping stone is needed.

### `formation/optimizer.py`
Should compute formation scale/pose/configuration inside the agreed safe region.
Start with a small template library and clear constraints.

### `formation/assignment.py`
Should solve the robot-to-slot assignment problem. A centralized implementation is acceptable for the Python phase unless the user explicitly asks for a distributed version.

### `robots/dynamics.py`
Should simulate the 2D double-integrator motion:
\[
\dot p_i = v_i, \qquad \dot v_i = u_i
\]
with a discrete-time update for simulation.

Start with a simple PD-style target-tracking controller unless a later step requires a more paper-faithful controller.

---

## How to write function stubs
When scaffolding a new chunk, functions should be easy for the user to complete.
Use this style:

```python
def run_convex_hull_consensus(points: np.ndarray, adjacency: np.ndarray, num_rounds: int) -> list:
    """Run the distributed convex-hull consensus algorithm for a team of robots.

    Args:
        points: Array of shape (N, 2) with robot positions.
        adjacency: Binary adjacency matrix of shape (N, N).
        num_rounds: Number of consensus rounds to execute.

    Returns:
        history: Per-round history of each robot's local hull estimate.

    Notes:
        - This function should mirror the paper's convex-hull information propagation idea.
        - In the simplest version, each robot may store raw candidate points before computing a hull.
        - Later versions can optimize communication by sending only newly discovered hull vertices.
    """
    # Step 1: Validate input sizes and graph assumptions.
    # Step 2: Initialize each robot's local hull estimate with its own position.
    # Step 3: For each round, exchange local hull information with neighbors.
    # Step 4: Update each robot's hull estimate.
    # Step 5: Save history for plotting.
    raise NotImplementedError
```

This pattern is encouraged throughout the repository.

---

## Plotting expectations
Every meaningful milestone should have an associated visualization.
Examples:
- communication graph over robot positions,
- local hull estimates by robot,
- candidate direction utilities,
- local safe rectangles/polytopes,
- target formation overlaid on safe region,
- assignment lines from current to target positions,
- trajectories over time.

Plots should be readable and labeled. Use legends only when they genuinely help.

---

## Math explanation policy
Before implementing a new section of the paper, provide:
1. a short intuitive explanation,
2. the main equation(s),
3. what each variable means,
4. what the code will need to store/update.

Keep the explanation scoped to the chunk being implemented. Do not front-load the whole paper every time.

---

## What to avoid
- Do not jump straight to ROS2 or Gazebo.
- Do not introduce unnecessary abstractions too early.
- Do not hide the paper logic inside black-box libraries.
- Do not optimize communication or runtime before the basic algorithm is visible and correct.
- Do not mix several major paper stages into one first implementation.
- Do not replace paper steps with unrelated alternatives unless the user explicitly asks.

---

## Preferred development rhythm
For each new chunk, follow this order:
1. explain the math,
2. define the data structures,
3. scaffold the functions with docstrings/comments,
4. implement the smallest runnable version,
5. add a `main.py --mode ...` demo,
6. verify visually,
7. refine if the user wants more realism.

---

## Definition of done for a chunk
A chunk is complete when:
- the user can describe what it does,
- the code has clear stubs or implementations,
- `main.py` has a runnable mode for it,
- the output includes something visible or measurable,
- the code is ready to be reused by the next chunk.

---

## Current scope lock
Until the user says otherwise, the active scope is:
- Python only
- 2D only
- five robots
- paper-aligned chunks
- educational scaffolding with comments/docstrings
