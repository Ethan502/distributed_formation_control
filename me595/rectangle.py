"""Axis-aligned rectangular obstacle for the me595 scenario.

Mirrors ``python/geometry/rectangles.py`` but kept local so the me595
folder stays self-contained.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Rectangle:
    """An axis-aligned rectangular obstacle.

    Attributes:
        x_min: Left edge x coordinate.
        y_min: Bottom edge y coordinate.
        x_max: Right edge x coordinate.
        y_max: Top edge y coordinate.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(
                f"x_min ({self.x_min}) must be < x_max ({self.x_max})."
            )
        if self.y_min >= self.y_max:
            raise ValueError(
                f"y_min ({self.y_min}) must be < y_max ({self.y_max})."
            )

    @property
    def corners(self) -> np.ndarray:
        """Return the four corners as an array of shape (4, 2), CCW order."""
        return np.array([
            [self.x_min, self.y_min],
            [self.x_max, self.y_min],
            [self.x_max, self.y_max],
            [self.x_min, self.y_max],
        ])

    @property
    def center(self) -> np.ndarray:
        return np.array([
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        ])

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def distance_to_point(self, point: np.ndarray) -> float:
        """Minimum distance from ``point`` to the rectangle (0 if inside).

        Mirrors ``python/geometry/rectangles.Rectangle.distance_to_point``.
        Clamps the query point onto [x_min, x_max] x [y_min, y_max] and
        measures the distance from the original point to that clamp.
        """
        p = np.asarray(point, dtype=float)
        nx = float(np.clip(p[0], self.x_min, self.x_max))
        ny = float(np.clip(p[1], self.y_min, self.y_max))
        return float(np.linalg.norm(p - np.array([nx, ny])))

    def contains_point(self, point: np.ndarray) -> bool:
        """Strict containment test (interior only)."""
        p = np.asarray(point, dtype=float)
        return bool(
            self.x_min < p[0] < self.x_max
            and self.y_min < p[1] < self.y_max
        )

    def __repr__(self) -> str:
        return (
            f"Rectangle(x=[{self.x_min}, {self.x_max}], "
            f"y=[{self.y_min}, {self.y_max}])"
        )
