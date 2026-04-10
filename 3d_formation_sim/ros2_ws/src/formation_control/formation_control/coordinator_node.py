"""
coordinator_node.py -- Mission coordinator for distributed formation control.

This node orchestrates the formation control pipeline by sequencing phases:
  1. Wait for all drones to come online
  2. Command takeoff
  3. Run consensus phases (hull, direction, free region, region)
  4. Compute formation and solve assignment
  5. Command movement to targets
  6. Check goal proximity and either replan or land

The coordinator does NOT participate in consensus -- it only sequences phases
and handles the assignment step (which requires global position knowledge).
After region consensus converges, the coordinator reads the agreed-upon safe
region from drone 0, runs the formation optimizer, solves the assignment
problem using current drone positions, and publishes per-drone targets.

Parameters are injected via launch file / YAML config.
"""

from enum import IntEnum, auto

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from rclpy.callback_groups import ReentrantCallbackGroup

from formation_msgs.msg import (
    PhaseCommand,
    DroneStatus,
    RegionConstraints,
    FormationTarget,
    MissionConfig,
)

# Algorithm imports -- used for formation optimization and assignment
from formation_control.consensus.graph_utils import (
    validate_adjacency_matrix,
    graph_diameter,
)
from formation_control.geometry.boxes import Box
from formation_control.formation.templates import TEMPLATES
from formation_control.formation.optimizer import optimize_formation
from formation_control.formation.assignment import solve_assignment


# ---------------------------------------------------------------------------
# QoS helpers
# ---------------------------------------------------------------------------
def _reliable_qos(depth: int = 10) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _transient_local_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


# ---------------------------------------------------------------------------
# Coordinator states
# ---------------------------------------------------------------------------
class CoordState(IntEnum):
    WAITING_FOR_DRONES = auto()
    COMMANDING_TAKEOFF = auto()
    WAITING_TAKEOFF = auto()
    COMMANDING_HULL = auto()
    WAITING_HULL = auto()
    COMMANDING_DIRECTION = auto()
    WAITING_DIRECTION = auto()
    COMMANDING_FREE_REGION = auto()
    WAITING_FREE_REGION = auto()
    COMMANDING_REGION = auto()
    WAITING_REGION = auto()
    COMPUTING_ASSIGNMENT = auto()
    COMMANDING_MOVE = auto()
    WAITING_MOVE = auto()
    CHECK_GOAL = auto()
    COMMANDING_LAND = auto()
    WAITING_LAND = auto()
    DONE = auto()


# Phase constants (mirror PhaseCommand.msg)
PHASE_IDLE = 0
PHASE_TAKEOFF = 1
PHASE_HULL = 2
PHASE_DIRECTION = 3
PHASE_FREE_REGION = 4
PHASE_REGION = 5
PHASE_FORMATION = 6
PHASE_ASSIGNMENT = 7
PHASE_MOVE = 8
PHASE_LAND = 9

# Timing
STATE_MACHINE_HZ = 2.0
STATE_MACHINE_PERIOD = 1.0 / STATE_MACHINE_HZ
PHASE_TIMEOUT_S = 30.0
GOAL_RADIUS = 2.0  # metres -- how close the centroid must be to the goal


# ===========================================================================
class CoordinatorNode(Node):
    """Mission coordinator that sequences formation control phases."""

    def __init__(self):
        super().__init__("coordinator_node")

        # -- Parameters -----------------------------------------------------
        self.declare_parameter("formation_control.num_drones", 4)
        self.declare_parameter("formation_control.adjacency_flat", [])
        self.declare_parameter("formation_control.obstacles_flat", [])
        self.declare_parameter("formation_control.goal", [0.0, 0.0, 0.0])
        self.declare_parameter("formation_control.sensing_range", 10.0)
        self.declare_parameter("formation_control.workspace_bounds",
                               [-50.0, -50.0, 0.0, 50.0, 50.0, 10.0])
        self.declare_parameter("formation_control.template_name", "square")
        self.declare_parameter("formation_control.tau", 3.0)
        self.declare_parameter("formation_control.goal_radius", 2.0)

        self.num_drones: int = self.get_parameter("formation_control.num_drones").value

        adj_flat = list(self.get_parameter("formation_control.adjacency_flat").value)
        n = self.num_drones
        self.adjacency = np.array(adj_flat, dtype=np.float64).reshape(n, n)
        validate_adjacency_matrix(self.adjacency)
        self.diameter: int = graph_diameter(self.adjacency)

        obs_flat = list(self.get_parameter("formation_control.obstacles_flat").value)
        self.obstacles: list[Box] = []
        for i in range(0, len(obs_flat), 6):
            self.obstacles.append(
                Box(obs_flat[i], obs_flat[i + 1], obs_flat[i + 2],
                    obs_flat[i + 3], obs_flat[i + 4], obs_flat[i + 5])
            )

        goal_param = list(self.get_parameter("formation_control.goal").value)
        self.goal = np.array(goal_param, dtype=np.float64)
        self.sensing_range: float = self.get_parameter("formation_control.sensing_range").value
        wb = list(self.get_parameter("formation_control.workspace_bounds").value)
        self.workspace_bounds = np.array(wb, dtype=np.float64)
        self.template_name: str = self.get_parameter("formation_control.template_name").value
        self.tau: float = self.get_parameter("formation_control.tau").value

        global GOAL_RADIUS
        GOAL_RADIUS = self.get_parameter("formation_control.goal_radius").value

        # -- State -----------------------------------------------------------
        self.coord_state = CoordState.WAITING_FOR_DRONES
        self.drone_statuses: dict[int, DroneStatus] = {}
        self.phase_start_time = self.get_clock().now()
        self.replan_count = 0

        # Region data captured from drone 0 after region consensus
        self.region_A: np.ndarray | None = None
        self.region_b: np.ndarray | None = None

        # -- Callback group --------------------------------------------------
        self.cb_group = ReentrantCallbackGroup()

        # -- Publishers ------------------------------------------------------
        self.pub_phase = self.create_publisher(
            PhaseCommand,
            "/formation/phase_command",
            _reliable_qos(),
            callback_group=self.cb_group,
        )

        # Per-drone target publishers
        self.pub_targets: dict[int, rclpy.publisher.Publisher] = {}
        for i in range(self.num_drones):
            self.pub_targets[i] = self.create_publisher(
                FormationTarget,
                f"/formation/drone_{i}_target",
                _reliable_qos(),
                callback_group=self.cb_group,
            )

        # Mission config (transient-local so late joiners receive it)
        self.pub_config = self.create_publisher(
            MissionConfig,
            "/formation/mission_config",
            _transient_local_qos(),
            callback_group=self.cb_group,
        )

        # -- Subscribers -----------------------------------------------------
        self.create_subscription(
            DroneStatus,
            "/formation/drone_status",
            self._drone_status_cb,
            _reliable_qos(depth=50),
            callback_group=self.cb_group,
        )

        # Subscribe to drone 0's region constraints to capture the final
        # agreed safe region after region consensus completes.
        self.create_subscription(
            RegionConstraints,
            "/uav1/region_constraints",
            self._region_cb,
            _reliable_qos(),
            callback_group=self.cb_group,
        )

        # -- Publish mission config once on startup --------------------------
        self._publish_mission_config()

        # -- State machine timer (2 Hz) --------------------------------------
        self.create_timer(
            STATE_MACHINE_PERIOD,
            self._state_machine_tick,
            callback_group=self.cb_group,
        )

        self.get_logger().info(
            f"Coordinator started: {self.num_drones} drones, "
            f"graph diameter={self.diameter}, "
            f"goal={self.goal.tolist()}"
        )

    # ===================================================================
    # Config publisher
    # ===================================================================
    def _publish_mission_config(self):
        """Publish MissionConfig with TRANSIENT_LOCAL QoS."""
        msg = MissionConfig()
        msg.num_drones = self.num_drones
        msg.goal = self.goal.tolist()

        obs_flat = []
        for box in self.obstacles:
            obs_flat.extend([
                box.x_min, box.y_min, box.z_min,
                box.x_max, box.y_max, box.z_max,
            ])
        msg.obstacles_flat = obs_flat

        msg.adjacency_flat = self.adjacency.flatten().astype(np.uint8).tolist()
        msg.sensing_range = self.sensing_range
        msg.workspace_bounds = self.workspace_bounds.tolist()
        self.pub_config.publish(msg)
        self.get_logger().info("Published MissionConfig")

    # ===================================================================
    # Callbacks
    # ===================================================================
    def _drone_status_cb(self, msg: DroneStatus):
        """Store the latest status from each drone."""
        self.drone_statuses[msg.drone_id] = msg

    def _region_cb(self, msg: RegionConstraints):
        """Capture the latest region constraints from drone 0.

        After region consensus, all drones have the same region, so we
        only need to listen to one.  We keep the latest message.
        """
        if msg.drone_id != 0:
            return
        m = msg.num_constraints
        if m == 0:
            return
        self.region_A = np.array(msg.a_matrix_flat).reshape(m, 3)
        self.region_b = np.array(msg.b_vector)

    # ===================================================================
    # Helpers
    # ===================================================================
    def _all_drones_reported(self) -> bool:
        """True if we have a recent status from every drone."""
        return len(self.drone_statuses) >= self.num_drones

    def _all_phase_complete(self) -> bool:
        """True if every drone reports phase_complete == True."""
        if not self._all_drones_reported():
            return False
        return all(
            s.phase_complete for s in self.drone_statuses.values()
        )

    def _send_phase(self, phase: int, num_rounds: int = 0):
        """Publish a PhaseCommand."""
        msg = PhaseCommand()
        msg.phase = phase
        msg.num_rounds = num_rounds
        self.pub_phase.publish(msg)
        self.phase_start_time = self.get_clock().now()
        self.get_logger().info(
            f"Sent phase command: {phase} (rounds={num_rounds})"
        )

    def _phase_elapsed_s(self) -> float:
        """Seconds since the last phase command was sent."""
        now = self.get_clock().now()
        return (now - self.phase_start_time).nanoseconds * 1e-9

    def _check_phase_timeout(self) -> bool:
        """Warn and return True if the current phase has timed out."""
        if self._phase_elapsed_s() > PHASE_TIMEOUT_S:
            self.get_logger().warn(
                f"Phase timeout ({PHASE_TIMEOUT_S}s) in state "
                f"{self.coord_state.name}"
            )
            return True
        return False

    def _get_drone_positions(self) -> np.ndarray:
        """Return (N, 3) array of current drone positions from status msgs."""
        positions = np.zeros((self.num_drones, 3))
        for drone_id, status in self.drone_statuses.items():
            if drone_id < self.num_drones:
                positions[drone_id] = np.array(status.position[:3])
        return positions

    def _centroid_distance_to_goal(self) -> float:
        """Distance from the team centroid to the goal position."""
        positions = self._get_drone_positions()
        centroid = positions.mean(axis=0)
        return float(np.linalg.norm(centroid - self.goal))

    # ===================================================================
    # State machine
    # ===================================================================
    def _state_machine_tick(self):
        """Main coordinator state machine, called at 2 Hz."""
        state = self.coord_state

        # ------------------------------------------------------------------
        if state == CoordState.WAITING_FOR_DRONES:
            if self._all_drones_reported():
                self.get_logger().info(
                    f"All {self.num_drones} drones online"
                )
                self.coord_state = CoordState.COMMANDING_TAKEOFF
            else:
                reported = list(self.drone_statuses.keys())
                self.get_logger().info(
                    f"Waiting for drones... have {reported}",
                    throttle_duration_sec=5.0,
                )

        # ------------------------------------------------------------------
        elif state == CoordState.COMMANDING_TAKEOFF:
            self._send_phase(PHASE_TAKEOFF)
            self.coord_state = CoordState.WAITING_TAKEOFF

        elif state == CoordState.WAITING_TAKEOFF:
            if self._all_phase_complete():
                self.get_logger().info("All drones airborne")
                self.coord_state = CoordState.COMMANDING_HULL
            elif self._check_phase_timeout():
                self.coord_state = CoordState.COMMANDING_TAKEOFF  # retry

        # ------------------------------------------------------------------
        elif state == CoordState.COMMANDING_HULL:
            self.replan_count += 1
            self.get_logger().info(
                f"--- Planning iteration {self.replan_count} ---"
            )
            self._send_phase(PHASE_HULL, num_rounds=self.diameter)
            self.coord_state = CoordState.WAITING_HULL

        elif state == CoordState.WAITING_HULL:
            if self._all_phase_complete():
                self.get_logger().info("Hull consensus complete")
                self.coord_state = CoordState.COMMANDING_DIRECTION
            elif self._check_phase_timeout():
                self.coord_state = CoordState.COMMANDING_HULL

        # ------------------------------------------------------------------
        elif state == CoordState.COMMANDING_DIRECTION:
            self._send_phase(PHASE_DIRECTION, num_rounds=self.diameter)
            self.coord_state = CoordState.WAITING_DIRECTION

        elif state == CoordState.WAITING_DIRECTION:
            if self._all_phase_complete():
                self.get_logger().info("Direction consensus complete")
                self.coord_state = CoordState.COMMANDING_FREE_REGION
            elif self._check_phase_timeout():
                self.coord_state = CoordState.COMMANDING_DIRECTION

        # ------------------------------------------------------------------
        elif state == CoordState.COMMANDING_FREE_REGION:
            self._send_phase(PHASE_FREE_REGION)
            self.coord_state = CoordState.WAITING_FREE_REGION

        elif state == CoordState.WAITING_FREE_REGION:
            if self._all_phase_complete():
                self.get_logger().info("Free region computation complete")
                self.coord_state = CoordState.COMMANDING_REGION
            elif self._check_phase_timeout():
                self.coord_state = CoordState.COMMANDING_FREE_REGION

        # ------------------------------------------------------------------
        elif state == CoordState.COMMANDING_REGION:
            self._send_phase(PHASE_REGION, num_rounds=self.diameter)
            self.coord_state = CoordState.WAITING_REGION

        elif state == CoordState.WAITING_REGION:
            if self._all_phase_complete():
                self.get_logger().info("Region consensus complete")
                self.coord_state = CoordState.COMPUTING_ASSIGNMENT
            elif self._check_phase_timeout():
                self.coord_state = CoordState.COMMANDING_REGION

        # ------------------------------------------------------------------
        elif state == CoordState.COMPUTING_ASSIGNMENT:
            self._compute_formation_and_assign()
            # _compute_formation_and_assign transitions state on its own

        # ------------------------------------------------------------------
        elif state == CoordState.COMMANDING_MOVE:
            self._send_phase(PHASE_MOVE)
            self.coord_state = CoordState.WAITING_MOVE

        elif state == CoordState.WAITING_MOVE:
            if self._all_phase_complete():
                self.get_logger().info("All drones arrived at targets")
                self.coord_state = CoordState.CHECK_GOAL
            elif self._check_phase_timeout():
                # Drones may still be moving; do not retry, just warn
                pass

        # ------------------------------------------------------------------
        elif state == CoordState.CHECK_GOAL:
            dist = self._centroid_distance_to_goal()
            self.get_logger().info(
                f"Centroid distance to goal: {dist:.2f} m "
                f"(threshold: {GOAL_RADIUS} m)"
            )
            if dist < GOAL_RADIUS:
                self.get_logger().info("Goal reached! Commanding landing.")
                self.coord_state = CoordState.COMMANDING_LAND
            else:
                self.get_logger().info(
                    "Goal not reached -- replanning from hull consensus"
                )
                self.coord_state = CoordState.COMMANDING_HULL

        # ------------------------------------------------------------------
        elif state == CoordState.COMMANDING_LAND:
            self._send_phase(PHASE_LAND)
            self.coord_state = CoordState.WAITING_LAND

        elif state == CoordState.WAITING_LAND:
            if self._all_phase_complete():
                self.get_logger().info("All drones landed. Mission complete.")
                self.coord_state = CoordState.DONE
            elif self._check_phase_timeout():
                self.coord_state = CoordState.COMMANDING_LAND

        # ------------------------------------------------------------------
        elif state == CoordState.DONE:
            pass  # Nothing to do

    # ===================================================================
    # Formation optimization + assignment
    # ===================================================================
    def _compute_formation_and_assign(self):
        """Run formation optimizer on the agreed safe region, solve
        the assignment problem, and publish targets to each drone.

        This is called once after region consensus completes. The
        coordinator uses the safe region captured from drone 0's
        RegionConstraints topic.
        """
        # Verify we have the safe region
        if self.region_A is None or self.region_b is None:
            self.get_logger().error(
                "No region data from drone 0 -- cannot compute formation. "
                "Retrying region consensus."
            )
            self.coord_state = CoordState.COMMANDING_REGION
            return

        # Get template
        template_fn = TEMPLATES.get(self.template_name)
        if template_fn is None:
            self.get_logger().error(
                f"Unknown template '{self.template_name}'"
            )
            self.coord_state = CoordState.COMMANDING_LAND
            return
        template = template_fn()

        # Compute preferred direction from goal relative to centroid.
        # This is a fallback; drones already computed the preferred direction
        # via consensus. We approximate it here as the direction toward goal.
        positions = self._get_drone_positions()
        centroid = positions.mean(axis=0)
        goal_vec = self.goal - centroid
        goal_dist = np.linalg.norm(goal_vec)
        if goal_dist > 1e-6:
            direction = goal_vec / goal_dist
        else:
            direction = np.array([1.0, 0.0, 0.0])

        # Run formation optimizer
        result = optimize_formation(
            self.region_A,
            self.region_b,
            template,
            direction,
            n_rotations=16,
            centroid=centroid,
            tau=self.tau,
        )

        if result is None:
            self.get_logger().warn(
                "Formation optimizer found no feasible solution. "
                "Commanding drones to move toward goal directly."
            )
            # Fallback: assign each drone a point near the goal
            for i in range(self.num_drones):
                target = self.goal.copy()
                # Slight offset per drone to avoid collision
                offset = 1.0 * (positions[i] - centroid)
                target_i = self.goal + offset
                self._publish_target(i, target_i)
            self.coord_state = CoordState.COMMANDING_MOVE
            return

        center, scale, angle, slots = result
        self.get_logger().info(
            f"Formation optimized: center={center}, "
            f"scale={scale:.2f}, angle={angle:.2f} rad"
        )

        # Solve assignment: match drones to formation slots
        assignment, cost = solve_assignment(positions, slots)
        self.get_logger().info(
            f"Assignment cost: {cost:.2f}, mapping: {assignment.tolist()}"
        )

        # Publish targets
        for drone_id in range(self.num_drones):
            slot_idx = assignment[drone_id]
            target = slots[slot_idx]
            self._publish_target(drone_id, target)

        self.coord_state = CoordState.COMMANDING_MOVE

    def _publish_target(self, drone_id: int, target: np.ndarray):
        """Publish a FormationTarget to a specific drone."""
        msg = FormationTarget()
        msg.drone_id = drone_id
        msg.target = target.tolist()
        if drone_id in self.pub_targets:
            self.pub_targets[drone_id].publish(msg)
            self.get_logger().info(
                f"  Drone {drone_id} -> target "
                f"[{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]"
            )


# ===========================================================================
# Entry point
# ===========================================================================
def main(args=None):
    rclpy.init(args=args)
    node = CoordinatorNode()

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
