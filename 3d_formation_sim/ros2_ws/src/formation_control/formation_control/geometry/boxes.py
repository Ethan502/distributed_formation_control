"""3D axis-aligned box obstacle.

Provides the Box dataclass used throughout the formation control pipeline
for representing obstacles and workspace boundaries.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Box:
    """Axis-aligned 3D box obstacle.

    Attributes:
        x_min, y_min, z_min: Lower corner coordinates.
        x_max, y_max, z_max: Upper corner coordinates.
    """
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @property
    def center(self) -> np.ndarray:
        """Center point of the box as a (3,) array."""
        return np.array([
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2,
        ])

    @property
    def size(self) -> np.ndarray:
        """Size of the box along each axis as a (3,) array."""
        return np.array([
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        ])

    def contains_point(self, p: np.ndarray) -> bool:
        """Check whether a point lies inside (or on the boundary of) the box.

        Args:
            p: (3,) point coordinates.

        Returns:
            True if the point is inside the box.
        """
        return (self.x_min <= p[0] <= self.x_max and
                self.y_min <= p[1] <= self.y_max and
                self.z_min <= p[2] <= self.z_max)

    def distance_to_point(self, p: np.ndarray) -> float:
        """Minimum Euclidean distance from a point to the box surface.

        Returns 0 if the point is inside the box.

        Args:
            p: (3,) point coordinates.

        Returns:
            Non-negative distance.
        """
        dx = max(self.x_min - p[0], 0.0, p[0] - self.x_max)
        dy = max(self.y_min - p[1], 0.0, p[1] - self.y_max)
        dz = max(self.z_min - p[2], 0.0, p[2] - self.z_max)
        return np.sqrt(dx * dx + dy * dy + dz * dz)

    @classmethod
    def from_flat(cls, flat: list) -> 'Box':
        """Create a Box from a flat list [xmin, ymin, zmin, xmax, ymax, zmax].

        Args:
            flat: List or array of 6 floats.

        Returns:
            Box instance.
        """
        return cls(*flat[:6])
