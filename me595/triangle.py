"""Triangular obstacle for the me595 alternate scenario.

The base distributed-formation project uses axis-aligned rectangles in
``python/geometry/rectangles.py``.  For me595 we want a single 30-60-90
right-triangle obstacle, so we define a small dataclass that mirrors the
style of ``Rectangle`` (vertices, center, simple containment / distance).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Triangle:
    """A triangular obstacle defined by its three vertices.

    Attributes:
        vertices: Array of shape (3, 2) holding the triangle vertices in
                  counter-clockwise order.  CCW order keeps the polygon
                  patch oriented correctly when drawn.
    """

    vertices: np.ndarray

    def __post_init__(self) -> None:
        verts = np.asarray(self.vertices, dtype=float)
        if verts.shape != (3, 2):
            raise ValueError(
                f"Triangle vertices must have shape (3, 2), got {verts.shape}."
            )
        # Force CCW orientation using the signed area.  If signed area is
        # negative the vertices are CW, so we flip them.
        if self._signed_area(verts) < 0.0:
            verts = verts[::-1]
        self.vertices = verts

    @staticmethod
    def _signed_area(v: np.ndarray) -> float:
        """Signed area of a triangle (positive if CCW)."""
        return 0.5 * (
            (v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1])
            - (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1])
        )

    @property
    def center(self) -> np.ndarray:
        """Centroid of the triangle, shape (2,)."""
        return self.vertices.mean(axis=0)

    @property
    def edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return the three edges as (start, end) point pairs."""
        return [
            (self.vertices[i], self.vertices[(i + 1) % 3]) for i in range(3)
        ]

    def distance_to_point(self, point: np.ndarray) -> float:
        """Return the minimum distance from ``point`` to the triangle.

        Returns 0 if the point is inside the triangle, otherwise the
        smallest distance from the point to any of the three edges.
        Mirrors ``Rectangle.distance_to_point`` so the two shapes are
        interchangeable for visibility queries.
        """
        p = np.asarray(point, dtype=float)
        if self.contains_point(p):
            return 0.0
        return min(_point_to_segment_distance(p, a, b) for a, b in self.edges)

    def contains_point(self, point: np.ndarray) -> bool:
        """Return True if point is strictly inside the triangle.

        Uses the standard same-side / barycentric sign test.
        """
        p = np.asarray(point, dtype=float)
        v0, v1, v2 = self.vertices

        # Compute barycentric coordinates via the area method.
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(denom) < 1e-12:
            return False
        a = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / denom
        b = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / denom
        c = 1.0 - a - b
        return (a > 0.0) and (b > 0.0) and (c > 0.0)

    def __repr__(self) -> str:
        verts_str = ", ".join(f"({x:.2f}, {y:.2f})" for x, y in self.vertices)
        return f"Triangle[{verts_str}]"


def _point_to_segment_distance(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> float:
    """Shortest distance from point p to line segment a-b."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float((p - a) @ ab) / denom
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def make_30_60_90(
    right_angle_vertex: np.ndarray,
    short_leg: float,
    base_side: str = "left",
) -> Triangle:
    """Construct a 30-60-90 right triangle with vertical base.

    A 30-60-90 triangle has side lengths in ratio ``1 : sqrt(3) : 2``
    (short leg : long leg : hypotenuse).  We treat the long leg as the
    "base" and orient it vertical (perpendicular to the x-axis).  The
    other (short) leg is then horizontal.

    Args:
        right_angle_vertex: 2D coordinates of the 90-degree corner.
                            The base goes upward from this vertex; the
                            short leg goes either left or right depending
                            on ``base_side``.
        short_leg:          Length of the short leg (the side opposite the
                            30-degree angle).  Long leg = sqrt(3) * short_leg.
                            Hypotenuse = 2 * short_leg.
        base_side:          Either "left" or "right".  This says which side
                            of the vertical base the rest of the triangle
                            sits on, which is the same as which way the
                            hypotenuse faces.  Use "left" if you want the
                            hypotenuse to face the drones when they start
                            on the left side of the world.

    Returns:
        Triangle with three vertices in CCW order.
    """
    if base_side not in ("left", "right"):
        raise ValueError("base_side must be 'left' or 'right'.")

    rv = np.asarray(right_angle_vertex, dtype=float)
    long_leg = np.sqrt(3.0) * short_leg

    # Right-angle vertex (bottom of the vertical base) and the top of the base.
    p_right_angle = rv
    p_top_of_base = rv + np.array([0.0, long_leg])

    # The third vertex sits horizontally off the right-angle corner.
    dx = -short_leg if base_side == "left" else short_leg
    p_third = rv + np.array([dx, 0.0])

    return Triangle(vertices=np.array([p_right_angle, p_top_of_base, p_third]))
