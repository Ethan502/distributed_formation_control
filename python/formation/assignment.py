"""Robot-to-formation-slot assignment (Phase 6).

Given current robot positions p_i and target slot positions r_j*, solve the
assignment that minimizes total squared travel distance:

    min_X  Σ_i Σ_j  x_ij * ||p_i - r_j*||²

where X is a permutation matrix (each robot gets exactly one slot).

This is a linear assignment problem, solvable in polynomial time via the
Hungarian algorithm (scipy.optimize.linear_sum_assignment).

Paper reference: Section 3.5, Eq. (19).

TODO: implement in Phase 6.
"""

# TODO: implement in Phase 6
