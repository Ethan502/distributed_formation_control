"""Polygon-aware velocity projection for the me595 runner.

``python/robots/dynamics.project_velocity_obstacles`` blocks robots from
sliding into rectangle bodies by enumerating the four AABB faces.  For
me595 we need the same behaviour for arbitrary convex polygons (the
30-60-90 triangle has a sloped face).  This module provides a generic
``project_velocity_obstacles`` that loops over each face of each convex
obstacle.

Convention reminder (matches ``me595/geometry.py``):

    A face of a CCW convex polygon goes from v_i to v_{i+1}.
    The unit OUTWARD normal is  n = (edge.y, -edge.x) / ||edge||.
    "Inside the obstacle" means  n · x  <  n · v_i  for all faces.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from me595.geometry import ConvexObstacle, polygon_vertices


def project_velocity_obstacles(
    velocity: np.ndarray,
    position: np.ndarray,
    obstacles: Iterable[ConvexObstacle],
    margin: float = 0.1,
) -> np.ndarray:
    """Strip velocity components that would push the robot into a polygon face.

    For each face of each convex obstacle:

        1. Compute the inward unit normal  n_in = -n_out  and the face
           tangent t.
        2. Project the robot's offset from the face start onto t.  Skip
           the face unless the projection lies strictly inside the edge
           (so robots beside the obstacle are not blocked).
        3. The robot must be on the OUTSIDE of the face (gap >= -margin).
        4. The velocity must point INTO the face  (n_in · v > 0).
        5. If all of the above hold, remove the component of v along
           n_in:  v -= (n_in · v) * n_in   (n_in is unit length).

    Args:
        velocity:  Current velocity, shape (2,).
        position:  Current position, shape (2,).
        obstacles: Iterable of convex obstacles (Triangle / Rectangle).
        margin:    How close to the face (in the same units as positions)
                   the projection kicks in.

    Returns:
        v_proj: Projected velocity, shape (2,).
    """
    v = np.array(velocity, dtype=float, copy=True)
    p = np.asarray(position, dtype=float)

    for obs in obstacles:
        verts = polygon_vertices(obs)
        K = len(verts)
        for i in range(K):
            v_i = verts[i]
            v_next = verts[(i + 1) % K]
            edge = v_next - v_i
            edge_len = float(np.linalg.norm(edge))
            if edge_len < 1e-12:
                continue
            t_hat = edge / edge_len
            # Outward normal (right-hand of CCW edge).
            n_out = np.array([t_hat[1], -t_hat[0]], dtype=float)
            n_in = -n_out

            offset = p - v_i
            tangent_proj = float(offset @ t_hat)
            signed_dist = float(offset @ n_out)  # >0 outside the face, <0 inside

            # 2. Strict tangent extent (allow corner sliding past endpoints).
            if not (0.0 < tangent_proj < edge_len):
                continue
            # 3. Robot must be on the outside, within `margin` of the face.
            if signed_dist < 0.0 or signed_dist > margin:
                continue
            # 4. Velocity must have a component into the face.
            if (n_in @ v) <= 0.0:
                continue
            # 5. Strip the inward component (n_in is unit length).
            v = v - (n_in @ v) * n_in

    return v
