"""Formation template library (Phase 5).

A formation template defines the relative positions of N robots around a
shared center.  The template vectors r_i are unit-scale — i.e. the centroid
of the template is at the origin and the points are normalized so the largest
distance from the origin is 1.0.

The actual target positions are obtained by applying a similarity transform:

    p_i* = c + s * R(φ) * r_i

where:
    c   — formation center (2D translation)
    s   — scale (controls how spread out the robots are)
    R(φ) — 2D rotation matrix for angle φ
    r_i — template offset for robot i, shape (2,)

Paper reference: Section 2.2, Eq. (4) — the isomorphic transformation.
"""

from __future__ import annotations
import numpy as np


def rotation_matrix(angle: float) -> np.ndarray:
    """Return the 2D rotation matrix for the given angle in radians.

    Args:
        angle: Rotation angle in radians.

    Returns:
        R: Array of shape (2, 2).
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s],
                     [s,  c]])


def apply_template(
    template: np.ndarray,
    center: np.ndarray,
    scale: float,
    angle: float,
) -> np.ndarray:
    """Apply a similarity transform to a template to get target slot positions.

    Computes p_i* = center + scale * R(angle) * r_i for each template row r_i.

    Args:
        template: Shape (N, 2) — unit-scale template offsets.
        center:   Shape (2,)  — formation center position.
        scale:    Non-negative float — formation size.
        angle:    Rotation angle in radians.

    Returns:
        slots: Shape (N, 2) — target position for each robot slot.
    """
    R = rotation_matrix(angle)
    # (N, 2) @ (2, 2).T  →  each row r_i is rotated by R
    return center + scale * (template @ R.T)


def pentagon(n: int = 5) -> np.ndarray:
    """Regular polygon template with n vertices, normalized to unit radius.

    Vertices are evenly spaced around the unit circle.  The first vertex
    sits at angle 0 (pointing right).

    Args:
        n: Number of vertices (default 5 for a pentagon).

    Returns:
        template: Shape (n, 2).
    """
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def line(n: int = 5) -> np.ndarray:
    """Horizontal line template with n robots evenly spaced, normalized to unit half-width.

    Robots are placed at x ∈ [-1, 1] with y = 0.

    Args:
        n: Number of robots.

    Returns:
        template: Shape (n, 2).
    """
    xs = np.linspace(-1.0, 1.0, n)
    return np.column_stack([xs, np.zeros(n)])


def vee(n: int = 5) -> np.ndarray:
    """V-shape template pointing right, normalized so the tips are at unit distance.

    The lead robot is at the front (rightmost point) and the two arms
    spread backward symmetrically.

    Args:
        n: Number of robots (must be odd so there is a clear lead robot).
           If even, the split is slightly asymmetric.

    Returns:
        template: Shape (n, 2).
    """
    # Split robots into left arm, lead, right arm
    half = n // 2
    lead = n - 2 * half  # 1 if n is odd, 0 if even

    # Arms: x goes from -1 back to 0 (negative x = behind the lead)
    arm_x = np.linspace(-1.0, 0.0, half, endpoint=False)

    # Upper arm: positive y spread
    upper = np.column_stack([arm_x, np.linspace(0.5, 0.0, half, endpoint=False)])
    # Lower arm: negative y spread (mirror)
    lower = np.column_stack([arm_x, np.linspace(-0.5, 0.0, half, endpoint=False)])

    parts = [upper, lower]
    if lead:
        parts.append(np.array([[0.0, 0.0]]))  # lead robot at front center

    template = np.vstack(parts)

    # Normalize so the furthest point from origin is at distance 1
    max_dist = np.max(np.linalg.norm(template, axis=1))
    if max_dist > 1e-9:
        template = template / max_dist

    return template


# Registry — maps name strings to template arrays for 5 robots
TEMPLATES: dict[str, np.ndarray] = {
    'pentagon': pentagon(5),
    'line':     line(5),
    'vee':      vee(5),
}
