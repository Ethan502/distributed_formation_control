"""3D formation templates for 4 robots.

Provides unit-scale formation shapes centered at the origin, plus
transformation utilities to place them in the world frame.
"""

import numpy as np


def tetrahedron() -> np.ndarray:
    """Regular tetrahedron centered at origin, unit scale.

    Returns:
        (4, 3) array of vertex positions.
    """
    verts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])  # normalize to unit radius
    return verts


def square_horizontal() -> np.ndarray:
    """Square in the horizontal (XY) plane, unit scale.

    Returns:
        (4, 3) array of vertex positions.
    """
    return np.array([
        [1, 1, 0],
        [1, -1, 0],
        [-1, -1, 0],
        [-1, 1, 0],
    ], dtype=float) / np.sqrt(2)


def line_formation() -> np.ndarray:
    """Line formation along the x-axis.

    Returns:
        (4, 3) array of vertex positions.
    """
    return np.array([
        [-1.5, 0, 0],
        [-0.5, 0, 0],
        [0.5, 0, 0],
        [1.5, 0, 0],
    ], dtype=float) / 1.5  # normalize so max extent is 1


def diamond() -> np.ndarray:
    """Diamond/rhombus shape using 3 axes.

    Returns:
        (4, 3) array of vertex positions.
    """
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
    ], dtype=float)


TEMPLATES = {
    'tetrahedron': tetrahedron,
    'square': square_horizontal,
    'line': line_formation,
    'diamond': diamond,
}


def rotation_matrix_z(angle: float) -> np.ndarray:
    """3D rotation matrix around the Z axis.

    Args:
        angle: Rotation angle in radians.

    Returns:
        (3, 3) rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


def apply_template(template: np.ndarray, center: np.ndarray,
                   scale: float, angle: float) -> np.ndarray:
    """Apply transformation to a formation template.

    Computes: slots = center + scale * R_z(angle) @ r_i  for each vertex r_i.

    Args:
        template: (N, 3) unit-scale template vertices.
        center: (3,) formation center in world frame.
        scale: Formation scale factor.
        angle: Yaw rotation angle in radians.

    Returns:
        (N, 3) array of slot positions in world frame.
    """
    R = rotation_matrix_z(angle)
    return center + scale * (template @ R.T)
