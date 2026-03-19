"""Formation optimization inside the agreed obstacle-free region (Phase 5).

Given the common convex region P and the preferred direction θ*, each robot
independently solves the same constrained optimization to find the best
configuration state z* = (t, s, α) for one of the template formations.

Objective (paper Eq. 18):
    J_f(z) = w_t ||t - g(t_1)||² + w_s ||s - s̃||² + w_α ||α - ᾱ||²

Subject to:
    - All formation vertices V(z, f) × t_1 ⊂ P  (collision-free)
    - s ≥ 2 * max(r, h) / d_f                   (minimum size)

Paper reference: Section 3.4, Eq. (17)–(18).

TODO: implement in Phase 5.
"""

# TODO: implement in Phase 5
