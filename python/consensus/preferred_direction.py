"""Distributed preferred direction of motion consensus — Algorithm 2 (Phase 2).

Section 3.2 of the paper: robots agree on the best direction θ* for the team
to move by running a min-consensus on per-robot utility vectors.

Key idea:
    - A discrete set Θ = {θ_1, ..., θ_κ} of candidate angles is shared by all.
    - Each robot i assigns a utility u_i(θ) ≥ 0 to each angle θ based on its
      local perception (e.g., clearance from obstacles in direction θ).
    - The global utility u(θ) = min_{i} u_i(θ).
    - The team selects θ* = argmax_θ u(θ)  (best worst-case direction).
    - Distributed update (Algorithm 2):
        u_i(k+1) = min_{j ∈ N_i} ( u_i(k), u_j(k) )  component-wise.
    - After d rounds, every robot knows the global utility vector.

Paper reference: Algorithm 2, Section 3.2 (paper p. 1085–1086).
"""

# TODO: implement in Phase 2
