"""3D preferred direction consensus.

Implements the max-min consensus over a discrete set of candidate directions
on the unit sphere. Each robot evaluates local utility for each direction
(based on obstacle clearance and goal alignment), then robots exchange
utilities taking componentwise minimums. After convergence, every robot
selects the direction with the highest worst-case utility.

This is the 3D extension of the paper's preferred direction algorithm:
candidate directions are sampled on the sphere (instead of a circle in 2D).
"""

import numpy as np


def generate_candidate_directions(num_directions: int) -> np.ndarray:
    """Generate roughly uniform unit vectors on the sphere using Fibonacci sampling.

    The Fibonacci sphere produces a near-uniform distribution of points on
    the unit sphere without requiring optimization. This is used to discretize
    the set of candidate travel directions in 3D.

    Formula for the i-th point (i = 0, ..., K-1):
        phi   = arccos(1 - 2*(i + 0.5) / K)
        theta = pi * (1 + sqrt(5)) * i
        direction = [sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi)]

    Args:
        num_directions: Number of candidate directions K to generate.

    Returns:
        (K, 3) array of unit vectors on the sphere.
    """
    if num_directions <= 0:
        return np.empty((0, 3))

    golden_ratio_factor = np.pi * (1.0 + np.sqrt(5.0))
    indices = np.arange(num_directions, dtype=float)

    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / num_directions)
    theta = golden_ratio_factor * indices

    directions = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])
    return directions


def ray_box_intersection(origin: np.ndarray, direction: np.ndarray, box,
                         max_range: float = 20.0) -> float:
    """Compute ray-AABB intersection distance using the 3D slab method.

    Finds the nearest intersection point of a ray (origin + t * direction)
    with an axis-aligned bounding box. Returns max_range if the ray does
    not intersect the box within [0, max_range].

    The slab method works by computing entry/exit t-values for each axis
    slab (pair of parallel planes) and finding their overlap.

    Args:
        origin: (3,) ray origin point.
        direction: (3,) ray direction (should be unit vector).
        box: Object with x_min, y_min, z_min, x_max, y_max, z_max attributes.
        max_range: Maximum ray distance to consider.

    Returns:
        Distance to nearest intersection, or max_range if no hit.
    """
    mins = np.array([box.x_min, box.y_min, box.z_min])
    maxs = np.array([box.x_max, box.y_max, box.z_max])

    t_near = 0.0
    t_far = max_range

    for i in range(3):
        if abs(direction[i]) < 1e-12:
            # Ray is parallel to this slab
            if origin[i] < mins[i] or origin[i] > maxs[i]:
                return max_range  # No intersection
        else:
            inv_d = 1.0 / direction[i]
            t1 = (mins[i] - origin[i]) * inv_d
            t2 = (maxs[i] - origin[i]) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            t_near = max(t_near, t1)
            t_far = min(t_far, t2)
            if t_near > t_far:
                return max_range  # No intersection

    return t_near


def compute_local_utilities(position: np.ndarray, obstacles: list,
                            directions: np.ndarray, max_range: float = 20.0,
                            goal: np.ndarray = None) -> np.ndarray:
    """Compute this robot's local utility for each candidate direction.

    For each candidate direction, the utility represents how far the robot
    can travel in that direction before hitting an obstacle, optionally
    weighted by alignment with the goal direction.

    Base utility for direction k = min distance to any obstacle along ray.
    If a goal is provided, utility is scaled by a weight that favors
    directions aligned with the goal:
        weight = 0.2 + 0.8 * max(0, dot(direction, goal_dir))

    Args:
        position: (3,) this robot's current position.
        obstacles: List of box obstacles with x/y/z min/max attributes.
        directions: (K, 3) array of candidate unit directions.
        max_range: Maximum sensing range for ray casting.
        goal: (3,) optional goal position. If provided, directions toward
              the goal are weighted more highly.

    Returns:
        (K,) array of utility values, one per candidate direction.
    """
    K = directions.shape[0]
    utilities = np.full(K, max_range)

    # For each direction, find minimum distance to any obstacle
    for k in range(K):
        d = directions[k]
        for obs in obstacles:
            dist = ray_box_intersection(position, d, obs, max_range)
            if dist < utilities[k]:
                utilities[k] = dist

    # Apply goal-alignment weighting if goal is provided
    if goal is not None:
        goal_vec = goal - position
        goal_dist = np.linalg.norm(goal_vec)
        if goal_dist > 1e-9:
            goal_dir = goal_vec / goal_dist
            # Weight: 0.2 base + 0.8 * alignment (clamped to non-negative)
            alignment = directions @ goal_dir  # (K,)
            weights = 0.2 + 0.8 * np.maximum(0.0, alignment)
            utilities *= weights

    return utilities


def merge_direction_utilities(my_utils: np.ndarray,
                              neighbor_utils: list) -> np.ndarray:
    """Merge direction utilities with neighbors via componentwise minimum.

    This implements the min-consensus step: the utility of a direction is
    the worst-case (minimum) utility across all robots that have contributed.
    After enough rounds (graph diameter), this converges to the global
    worst-case utility for each direction.

    Args:
        my_utils: (K,) this robot's current utility estimates.
        neighbor_utils: List of (K,) arrays received from neighbors.

    Returns:
        (K,) array of merged utilities (componentwise minimum).
    """
    result = my_utils.copy()
    for nu in neighbor_utils:
        if nu is not None and len(nu) > 0:
            result = np.minimum(result, nu)
    return result


def select_preferred_direction(utilities: np.ndarray,
                               directions: np.ndarray) -> np.ndarray:
    """Select the direction with the highest utility (max of min-consensus).

    After min-consensus has converged, the direction with the highest
    remaining utility is the one that is safest in the worst case across
    all robots. This is the max-min optimal direction.

    Args:
        utilities: (K,) array of converged utility values.
        directions: (K, 3) array of candidate unit directions.

    Returns:
        (3,) unit vector of the preferred direction.
    """
    best_idx = np.argmax(utilities)
    return directions[best_idx].copy()
