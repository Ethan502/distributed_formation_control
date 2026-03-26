"""Axis-aligned rectangular obstacles used in the 2D simulation.

Each obstacle is fully described by its minimum and maximum x and y
coordinates.  This is a 2D simplification of the paper's 3D cylindrical
obstacle model (Section 2.1.4).

Used by:
    - plotting/draw_team.py   — to draw obstacles on the scene.
    - consensus/preferred_direction.py — to compute clearance utilities.
    - geometry/free_space.py  — to build local obstacle-free regions.
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
                f"x_min ({self.x_min}) must be strictly less than x_max ({self.x_max})."
            )
        if self.y_min >= self.y_max:
            raise ValueError(
                f"y_min ({self.y_min}) must be strictly less than y_max ({self.y_max})."
            )

    @property
    def corners(self) -> np.ndarray:
        """Return the four corners as an array of shape (4, 2), counter-clockwise.

        Order: bottom-left, bottom-right, top-right, top-left.
        """
        return np.array([
            [self.x_min, self.y_min],
            [self.x_max, self.y_min],
            [self.x_max, self.y_max],
            [self.x_min, self.y_max],
        ])

    @property
    def center(self) -> np.ndarray:
        """Return the center of the rectangle as a shape (2,) array."""
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
        """Return the minimum distance from point to the rectangle boundary.

        Uses the nearest-point-on-rectangle approach: clamp the query point
        to [x_min, x_max] × [y_min, y_max] and measure the distance.
        Returns 0 if the point is inside the rectangle.

        Args:
            point: 2D point, shape (2,).
        """
        nearest_x = np.clip(point[0], self.x_min, self.x_max)
        nearest_y = np.clip(point[1], self.y_min, self.y_max)
        return float(np.linalg.norm(point - np.array([nearest_x, nearest_y])))

    def contains_point(self, point: np.ndarray) -> bool:
        """Return True if point is strictly inside the rectangle.

        Args:
            point: 2D point, shape (2,).
        """
        return (
            self.x_min < point[0] < self.x_max
            and self.y_min < point[1] < self.y_max
        )

    def __repr__(self) -> str:
        return (
            f"Rectangle(x=[{self.x_min}, {self.x_max}], "
            f"y=[{self.y_min}, {self.y_max}])"
        )
