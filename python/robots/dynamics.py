"""Double-integrator dynamics for 2D robots (Phase 0 / Phase 7).

The continuous-time model is:

    dp/dt = v
    dv/dt = u

Discrete-time (forward Euler, timestep dt):

    p[k+1] = p[k] + dt * v[k]
    v[k+1] = v[k] + dt * u[k]

Paper reference: Section 2.1.1 (robot model) and Section 3.6 (real-time
control).  The paper uses a 3D model; we simplify to 2D here.
"""

from __future__ import annotations
import numpy as np
from geometry.rectangles import Rectangle


def step_double_integrator(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance a robot's position and velocity by one timestep.

    Uses the explicit (forward) Euler integration of the double-integrator:

        p_new = p + dt * v
        v_new = v + dt * u

    Args:
        position:     Current position [px, py], shape (2,).
        velocity:     Current velocity [vx, vy], shape (2,).
        acceleration: Control input [ax, ay], shape (2,).
        dt:           Timestep in seconds.  Must be positive.

    Returns:
        new_position: Updated position, shape (2,).
        new_velocity: Updated velocity, shape (2,).

    Notes:
        - All three input arrays must have shape (2,).
        - Forward Euler is first-order accurate; small dt keeps it stable.
        - For larger timesteps a symplectic integrator (update position with
          the NEW velocity) would be more energy-conserving, but Euler is
          sufficient for learning purposes.
    """
    if position.shape != (2,) or velocity.shape != (2,) or acceleration.shape != (2,):
        raise ValueError("position, velocity, and acceleration must all have shape (2,)")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    new_position = position + dt * velocity
    new_velocity = velocity + dt * acceleration
    return new_position, new_velocity


def compute_pd_acceleration(
    position: np.ndarray,
    velocity: np.ndarray,
    target_position: np.ndarray,
    kp: float,
    kd: float,
    target_velocity: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a PD control acceleration driving the robot toward a target.

    The PD law (paper Section 3.6):

        u = -Kp * (p - p*) - Kd * (v - v*)

    where p* is the target position and v* is the target velocity
    (zero by default, i.e., we want the robot to come to rest at the target).

    Args:
        position:        Current position [px, py], shape (2,).
        velocity:        Current velocity [vx, vy], shape (2,).
        target_position: Desired position [px*, py*], shape (2,).
        kp:              Proportional gain (scalar, positive).
        kd:              Derivative gain (scalar, positive).
        target_velocity: Desired velocity at target, shape (2,).
                         Defaults to zero vector if None.

    Returns:
        acceleration: Control output [ax, ay], shape (2,).

    Notes:
        - A good starting point for gains: kp=2.0, kd=1.5.
        - For stability of the closed-loop double-integrator you need
          kp > 0 and kd > 0.  Critical damping satisfies kd = 2*sqrt(kp).
        - Do not clamp the output here; if needed, apply acceleration limits
          in the simulation runner.
    """
    if target_velocity is None:
        target_velocity = np.zeros(2)
    pos_error = position - target_position
    vel_error = velocity - target_velocity
    return -kp * pos_error - kd * vel_error


def project_velocity(
    velocity: np.ndarray,
    position: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    margin: float = 0.05,
) -> np.ndarray:
    """Project velocity to prevent the robot from crossing a half-space boundary.

    For each constraint a · x ≤ b, if the robot is near the boundary AND its
    velocity points into the constraint (toward the forbidden side), strip out
    that velocity component so the robot slides along the boundary instead of
    crossing it.

    Args:
        velocity: Current velocity [vx, vy], shape (2,).
        position: Current position [px, py], shape (2,).
        A:        Shape (M, 2) — half-space normals of the agreed region.
        b:        Shape (M,)  — half-space bounds.
        margin:   How close to the boundary (in meters) before projection kicks in.

    Returns:
        v_proj: Projected velocity, shape (2,).

    Notes:
        - Start with v_proj = velocity.copy().
        - Loop over each row (a_row, b_val) in zip(A, b).
        - Check two conditions:
            1. Robot is near or outside the boundary: a_row @ position >= b_val - margin
            2. Velocity points into the constraint: a_row @ v_proj > 0
        - If both hold, remove the component of v_proj along a_row:
              v_proj -= (a_row @ v_proj) / (a_row @ a_row) * a_row
        - Return v_proj.
    """
    v_proj = velocity.copy()
    for a_row,b_row in zip(A,b):
        if a_row @ position >= b_row-margin and a_row @ v_proj > 0:
            v_proj -= (a_row @ v_proj) / (a_row @ a_row) * a_row
    return v_proj


def project_velocity_obstacles(
    velocity: np.ndarray,
    position: np.ndarray,
    obstacles: list[Rectangle],
    margin: float = 0.1,
) -> np.ndarray:
    """Project velocity to prevent the robot from entering obstacle bodies.

    For each face of each obstacle, checks:
        1. The robot is on the exterior side of that face.
        2. The robot is within the face's perpendicular extent (so we don't
           block free movement beside the obstacle).
        3. The velocity points into the face.

    If all three hold, the velocity component into that face is removed.

    This is important because the P-region half-spaces only describe the
    corridor (e.g. y ∈ [1, 3]) — they don't block the solid obstacle bodies
    themselves.  A robot starting above y=3 could cross through an obstacle
    corner on the way down without this extra check.

    Args:
        velocity:  Current velocity, shape (2,).
        position:  Current position, shape (2,).
        obstacles: List of Rectangle obstacles.
        margin:    Distance from face at which blocking kicks in.

    Returns:
        v_proj: Projected velocity, shape (2,).
    """
    v = velocity.copy()
    for obs in obstacles:
        # Left face (x = x_min): robot is to the left, y strictly inside obstacle.
        # Use strict inequalities for the lateral (y) range so that robots sitting
        # exactly on the corridor boundary (y = y_max or y = y_min) are NOT blocked
        # from passing east — they are already on the safe side.
        if (position[0] <= obs.x_min and
                obs.y_min < position[1] < obs.y_max):
            a = np.array([1.0, 0.0])
            if a @ position >= obs.x_min - margin and a @ v > 0:
                v -= (a @ v) / (a @ a) * a

        # Right face (x = x_max): robot is to the right, y strictly inside obstacle.
        if (position[0] >= obs.x_max and
                obs.y_min < position[1] < obs.y_max):
            a = np.array([-1.0, 0.0])
            if a @ position >= -obs.x_max - margin and a @ v > 0:
                v -= (a @ v) / (a @ a) * a

        # Bottom face (y = y_min): robot is below, x strictly inside obstacle.
        if (position[1] <= obs.y_min and
                obs.x_min < position[0] < obs.x_max):
            a = np.array([0.0, 1.0])
            if a @ position >= obs.y_min - margin and a @ v > 0:
                v -= (a @ v) / (a @ a) * a

        # Top face (y = y_max): robot is above, x strictly inside obstacle.
        if (position[1] >= obs.y_max and
                obs.x_min < position[0] < obs.x_max):
            a = np.array([0.0, -1.0])
            if a @ position >= -obs.y_max - margin and a @ v > 0:
                v -= (a @ v) / (a @ a) * a

    return v
