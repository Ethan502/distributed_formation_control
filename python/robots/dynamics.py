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
    # TODO: validate that position, velocity, acceleration each have shape (2,)
    # TODO: validate that dt > 0
    # TODO: compute new_position = position + dt * velocity
    # TODO: compute new_velocity = velocity + dt * acceleration
    # TODO: return (new_position, new_velocity)
    raise NotImplementedError


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
    # TODO: if target_velocity is None, set it to np.zeros(2)
    # TODO: compute position error: pos_error = position - target_position
    # TODO: compute velocity error: vel_error = velocity - target_velocity
    # TODO: return -kp * pos_error - kd * vel_error
    raise NotImplementedError
