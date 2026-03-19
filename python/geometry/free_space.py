"""Local obstacle-free convex region construction (Phase 3) and
distributed consensus intersection (Phase 4).

Section 3.3 of the paper: each robot computes a large convex region in
obstacle-free position-time space P_i ⊂ R^3 × [0, τ], then robots run
distributed consensus (Algorithm 3) to agree on the intersection P = ∩ P_i.

In our 2D simplification we will work in position space R^2 × [0, τ].
The region is represented as a half-space system {x | A x ≤ b}, so that
intersection is simply stacking the (A, b) rows together.

Paper reference: Section 3.3, Algorithm 3, Proposition 2 (paper p. 1086–1087).

TODO: implement in Phase 3.
"""

# TODO: implement in Phase 3
