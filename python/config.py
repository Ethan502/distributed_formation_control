"""Global configuration constants for the distributed formation control simulation.

Change values here to affect all modes at once.
"""

# --- Team ---
NUM_ROBOTS: int = 5

# --- Simulation ---
DT: float = 0.1          # discrete timestep, seconds
SIM_STEPS: int = 100     # default number of simulation steps

# --- PD controller gains (Phase 7) ---
KP: float = 2.0          # proportional gain
KD: float = 1.5          # derivative gain

# --- Preferred direction consensus (Phase 2) ---
NUM_CANDIDATE_DIRECTIONS: int = 36   # number of evenly-spaced candidate angles in [0, 2pi)

# --- Formation (Phase 5) ---
DEFAULT_TEMPLATE: str = "pentagon"   # which formation template to try first
