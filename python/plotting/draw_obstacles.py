"""Additional obstacle visualization helpers (Phase 3+).

Basic obstacle drawing is in draw_team.draw_rect_obstacles.
This module holds the free-space region polygon drawer used in Phase 3/4.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def polytope_vertices(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Find the vertices of a 2D convex polytope {x | A x <= b}.

    Intersects every pair of constraint lines and keeps only the points
    that satisfy all constraints (i.e. are inside the polytope).
    Then sorts them counter-clockwise by angle from their centroid.

    Args:
        A: Shape (M, 2) — constraint normals.
        b: Shape (M,)  — constraint bounds.

    Returns:
        vertices: Shape (K, 2), the polytope corner points in CCW order.
                  Returns empty array if fewer than 3 valid vertices found.
    """
    M = A.shape[0]
    candidates = []

    for i in range(M):
        for j in range(i + 1, M):
            # Solve the 2x2 system [A[i]; A[j]] x = [b[i]; b[j]]
            mat = np.array([A[i], A[j]])
            rhs = np.array([b[i], b[j]])
            if abs(np.linalg.det(mat)) < 1e-9:
                continue  # parallel lines — no unique intersection
            try:
                pt = np.linalg.solve(mat, rhs)
            except np.linalg.LinAlgError:
                continue
            # Keep only if the point satisfies all constraints (with tolerance)
            if np.all(A @ pt <= b + 1e-9):
                candidates.append(pt)

    if len(candidates) < 3:
        return np.empty((0, 2))

    verts = np.array(candidates)
    # Remove near-duplicate points
    unique = [verts[0]]
    for v in verts[1:]:
        if not any(np.linalg.norm(v - u) < 1e-6 for u in unique):
            unique.append(v)
    verts = np.array(unique)

    if len(verts) < 3:
        return np.empty((0, 2))

    # Sort CCW by angle from centroid
    c = verts.mean(axis=0)
    angles = np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0])
    return verts[np.argsort(angles)]


def draw_free_region(
    ax: Axes,
    A: np.ndarray,
    b: np.ndarray,
    color: str = 'steelblue',
    alpha: float = 0.25,
    label: str | None = None,
) -> None:
    """Draw a 2D convex polytope {x | Ax <= b} as a filled polygon.

    Args:
        ax:    Matplotlib Axes.
        A:     Shape (M, 2) constraint normals.
        b:     Shape (M,) constraint bounds.
        color: Fill and edge colour.
        alpha: Polygon transparency.
        label: Optional legend label.
    """
    verts = polytope_vertices(A, b)
    if len(verts) < 3:
        return

    poly = plt.Polygon(verts, closed=True,
                       facecolor=color, edgecolor=color,
                       alpha=alpha, linewidth=1.5,
                       label=label, zorder=2)
    ax.add_patch(poly)


def draw_halfspace_boundaries(
    ax: Axes,
    A: np.ndarray,
    b: np.ndarray,
    color: str = 'dimgrey',
) -> None:
    """Draw each half-space boundary line as a dotted line across the axes.

    For constraint a · x = b, the boundary is the line a[0]*x + a[1]*y = b.
    Each line is clipped to the current axes limits.

    Args:
        ax:    Matplotlib Axes (must already have xlim/ylim set).
        A:     Shape (M, 2) constraint normals.
        b:     Shape (M,)  constraint bounds.
        color: Line colour.
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    for a_row, b_val in zip(A, b):
        ax_coef, ay_coef = a_row

        if abs(ay_coef) > 1e-9:
            # Express as y = (b - ax*x) / ay, sweep across x range
            xs = np.array([x_min, x_max])
            ys = (b_val - ax_coef * xs) / ay_coef
        elif abs(ax_coef) > 1e-9:
            # Vertical line: x = b / ax
            x_val = b_val / ax_coef
            xs = np.array([x_val, x_val])
            ys = np.array([y_min, y_max])
        else:
            continue  # degenerate — skip

        # Clip to plot bounds
        points = np.column_stack([xs, ys])
        in_bounds = (
            (points[:, 0] >= x_min - 1e-6) & (points[:, 0] <= x_max + 1e-6) &
            (points[:, 1] >= y_min - 1e-6) & (points[:, 1] <= y_max + 1e-6)
        )
        if in_bounds.sum() < 2:
            continue

        ax.plot(xs, ys, color=color, linewidth=1.0,
                linestyle=':', zorder=3)
