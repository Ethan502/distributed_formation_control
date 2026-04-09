"""Generic N-vertex polygon obstacle for the me595 scenario.

Used when we need to represent a non-triangular, non-rectangular shape —
for example the seamless union of a triangle and a rectangle that share
an edge.  Vertices are stored CCW.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Polygon:
    """A simple polygon described by its vertices in CCW order.

    Attributes:
        vertices: Array of shape (N, 2) holding the polygon vertices.
                  Stored CCW so matplotlib's Polygon patch fills correctly.
    """

    vertices: np.ndarray

    def __post_init__(self) -> None:
        verts = np.asarray(self.vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
            raise ValueError(
                f"Polygon vertices must have shape (N>=3, 2), got {verts.shape}."
            )
        if self._signed_area(verts) < 0.0:
            verts = verts[::-1]
        self.vertices = verts

    @staticmethod
    def _signed_area(v: np.ndarray) -> float:
        """Shoelace formula — positive if CCW."""
        x = v[:, 0]
        y = v[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def __repr__(self) -> str:
        return f"Polygon(N={len(self.vertices)})"
