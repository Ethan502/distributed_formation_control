"""Scenario builder for the me595 alternate map.

Mirrors the style of ``python/simulation/scenario.py``: a single function
returns a dict bundling agents, adjacency, obstacles, and a goal.  The
twist for me595 is that the obstacle is one large 30-60-90 triangle whose
hypotenuse sits just to the right of the drone swarm.
"""

from __future__ import annotations
import os
import sys

import numpy as np

# Make python/ importable so we can reuse its Agent dataclass and other
# pipeline pieces from the me595 alternate runner.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_PATH = os.path.join(_REPO_ROOT, "python")
if _PYTHON_PATH not in sys.path:
    sys.path.insert(0, _PYTHON_PATH)

from robots.agent import Agent  # noqa: E402  (sys.path adjusted above)

from me595.triangle import Triangle, make_30_60_90
from me595.rectangle import Rectangle
from me595.polygon import Polygon


def create_triangle_scenario() -> dict:
    """Create the me595 5-robot scenario with a 30-60-90 triangle obstacle.

    Layout (world coordinates):
        - 5 robots in a loose cluster on the left side of the world,
          spread across roughly x in [0, 2], y in [2, 6].
        - One large 30-60-90 right triangle obstacle with:
              * the long leg ("base") vertical, perpendicular to the x-axis,
              * the right-angle corner on the right at (8, 0),
              * the hypotenuse facing left toward the drones,
              * short leg = 5  ->  long leg = 5*sqrt(3) ~ 8.66,
                                   hypotenuse  = 10.
        - Goal placed past the obstacle on the right side at (12, 5).

    Communication graph:
        Ring topology (each robot connected to its two neighbors), so the
        graph is connected with diameter 2.

    Returns:
        scenario: dict with keys
            'agents'         : list of 5 ``Agent`` dataclass instances
                               (uses ``python/robots/agent.py``) with zero
                               initial velocity.
            'adjacency'      : np.ndarray of shape (5, 5), binary, symmetric.
            'obstacles'      : list of convex obstacle pieces actually used
                               by the algorithm pipeline.  For me595 this is
                               ``[triangle, rectangle]`` — the merged shape
                               itself is non-convex (reflex corner where the
                               rectangle meets the hypotenuse), so the two
                               convex pieces are kept separate for free-space
                               construction.
            'merged_outline' : ``Polygon`` for the seamless union shape.
                               Visualization-only; the algorithm never sees
                               this directly.
            'goal'           : np.ndarray of shape (2,).
    """
    # Robot starting positions: a loose cluster just left of the hypotenuse.
    # The hypotenuse runs from (3, 0) to (8, 8.66).  Putting the cluster
    # around x in [0, 2] keeps it visibly close but clear of the triangle.
    positions = np.array([
        [0.0, 3.0],
        [0.5, 4.5],
        [1.0, 3.5],
        [1.5, 5.0],
        [2.0, 4.0],
    ])

    # Use python/'s Agent dataclass so the me595 runner can reuse the rest
    # of the pipeline (consensus, formation, dynamics) unchanged.  The
    # sensing range is bumped up so every robot can see the whole obstacle
    # in this small scene; the default of 3.0 would leave most of the
    # triangle invisible from the swarm's starting cluster.
    agents = [
        Agent(
            id=i,
            position=positions[i].copy(),
            velocity=np.zeros(2),
            sensing_range=20.0,
        )
        for i in range(5)
    ]

    # Ring graph: 0-1-2-3-4-0
    adjacency = np.zeros((5, 5), dtype=float)
    for i in range(5):
        adjacency[i, (i + 1) % 5] = 1.0
        adjacency[(i + 1) % 5, i] = 1.0

    # Build the triangle obstacle.  Started life as a strict 30-60-90 but
    # the bottom (short) leg has been stretched a bit further in the -x
    # direction so the hypotenuse is more shallow and gives the swarm a
    # bit more room to slide past.
    #   - Right-angle vertex (bottom-right): (8, -2)
    #   - Top of vertical base ("tip"):      (8, -2 + 5*sqrt(3)) ~ (8, 6.66)
    #   - Bottom-left vertex:                (1, -2)   (was (3, -2))
    # Vertical right edge (x=8) and bottom edge (y=-2) are preserved so
    # the rectangle merge logic still applies unchanged.
    tri_y_bot = -2.0
    tri_x_right = 8.0
    tri_tip_y_value = tri_y_bot + 5.0 * float(np.sqrt(3.0))  # ~6.66
    tri_x_bot_left = 1.0
    triangle = Triangle(
        vertices=np.array([
            [tri_x_right,    tri_y_bot],         # right-angle corner
            [tri_x_right,    tri_tip_y_value],   # tip (top of vertical base)
            [tri_x_bot_left, tri_y_bot],         # bottom-left
        ])
    )

    # Rectangle that extends up from the triangle, overlapped into the
    # triangle's body so the two shapes can be merged into a single
    # seamless polygon.  Geometry choices:
    #   * Right edge collinear with the triangle's vertical base (x = 8).
    #   * Width = 1.5  -> left edge at x = 6.5, which lies inside the
    #     triangle's x-range [3, 8].
    #   * Bottom y pushed BELOW the hypotenuse at the rectangle's left
    #     edge, so the rectangle's bottom edge is fully inside the
    #     triangle (the merge has no gap and no internal seam).
    #   * Top extends ~5 units above the triangle's tip so the visible
    #     "stem" above the triangle is comfortably > 4 units long.
    tri_tip_y = float(triangle.vertices[:, 1].max())  # ~6.66
    rect_x_min = 6.5
    rect_x_max = 8.0
    rect_y_min = 3.5                  # below hypotenuse y at x=6.5 (~4.06)
    rect_y_max = tri_tip_y + 9.0      # ~15.66
    rectangle = Rectangle(
        x_min=rect_x_min,
        y_min=rect_y_min,
        x_max=rect_x_max,
        y_max=rect_y_max,
    )

    # Build the union polygon of the triangle and rectangle so the scene
    # renders as one solid obstacle with no internal edges.  Trace the
    # outer boundary CCW; see notes below.
    merged_outline = _merge_triangle_with_rectangle(triangle, rectangle)

    # The algorithm uses the convex pieces directly (the merged shape is
    # non-convex so it can't be used as a single half-plane block).
    obstacles = [triangle, rectangle]

    # Goal sits past the obstacle on the right.
    goal = np.array([12.0, 3.0])

    return {
        "agents": agents,
        "adjacency": adjacency,
        "obstacles": obstacles,
        "merged_outline": merged_outline,
        "goal": goal,
    }


def _merge_triangle_with_rectangle(triangle: Triangle, rect: Rectangle) -> Polygon:
    """Compute the union of the 30-60-90 triangle and the upward rectangle.

    This is a special-case helper, not a general polygon-union routine.
    It assumes the exact configuration produced by ``create_triangle_scenario``:

        * The triangle is the 30-60-90 with vertical right edge at x = 8,
          horizontal bottom edge at y = triangle's bottom y, and the
          hypotenuse running from the bottom-left vertex up to the top
          vertex at (8, tip_y).
        * The rectangle has x_max == 8 (so its right edge is collinear
          with the triangle's right edge) and x_min strictly between the
          triangle's bottom-left x and 8.
        * The rectangle's y_min is strictly below the hypotenuse at
          x = rect.x_min, so the rectangle's bottom edge cuts into the
          triangle's interior.

    With those assumptions the union outline (CCW) is:

        bottom-left of triangle
         -> bottom-right of triangle              (along the bottom edge)
         -> top-right of rectangle                (straight up x = 8)
         -> top-left of rectangle                 (along the rectangle top)
         -> point where rect's left edge meets the hypotenuse
         -> back along the hypotenuse to the bottom-left of the triangle

    Args:
        triangle: The 30-60-90 triangle from ``make_30_60_90``.
        rect:     The rectangle satisfying the assumptions above.

    Returns:
        Polygon containing the merged outline.
    """
    # Identify the three named triangle vertices.  In CCW order produced
    # by Triangle.__post_init__ we expect:
    #   v[0] = right-angle corner (bottom-right of triangle, here (8, y_bot))
    #   v[1] = top of vertical base ("tip", here (8, tip_y))
    #   v[2] = third vertex (bottom-left, here (x_bot_left, y_bot))
    v_right_angle = triangle.vertices[0]
    v_tip = triangle.vertices[1]
    v_bot_left = triangle.vertices[2]

    y_bot = float(v_right_angle[1])
    x_right = float(v_right_angle[0])
    tip_y = float(v_tip[1])
    x_bot_left = float(v_bot_left[0])

    # Sanity-check the special-case assumptions so this function fails
    # loudly rather than silently producing a bad polygon.
    if not np.isclose(rect.x_max, x_right):
        raise ValueError(
            "Rectangle x_max must equal triangle's right edge x "
            f"({x_right}); got {rect.x_max}."
        )
    if not (x_bot_left < rect.x_min < x_right):
        raise ValueError(
            f"Rectangle x_min ({rect.x_min}) must lie strictly between "
            f"the triangle's bottom-left x ({x_bot_left}) and right x ({x_right})."
        )

    # Hypotenuse y at x = rect.x_min, by linear interpolation between
    # (x_bot_left, y_bot) and (x_right, tip_y).
    t = (rect.x_min - x_bot_left) / (x_right - x_bot_left)
    hypo_y_at_left = y_bot + t * (tip_y - y_bot)

    if rect.y_min >= hypo_y_at_left:
        raise ValueError(
            f"Rectangle y_min ({rect.y_min}) must be strictly less than the "
            f"hypotenuse y ({hypo_y_at_left:.3f}) at x = rect.x_min so the "
            "rectangle truly overlaps the triangle's interior."
        )

    # Trace the outer boundary CCW.
    union_vertices = np.array([
        [x_bot_left, y_bot],            # triangle bottom-left
        [x_right,    y_bot],            # triangle bottom-right
        [x_right,    rect.y_max],       # rectangle top-right (straight up x=8)
        [rect.x_min, rect.y_max],       # rectangle top-left
        [rect.x_min, hypo_y_at_left],   # where rect left edge meets the hypotenuse
    ])

    return Polygon(vertices=union_vertices)
