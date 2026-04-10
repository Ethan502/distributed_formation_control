"""
drone_node.py -- Per-drone ROS2 node for distributed formation control.

Each drone runs one instance of this node. It handles:
  1. PX4 offboard interface (arming, mode switching, velocity/position commands)
  2. Distributed consensus participation (hull, direction, region)
  3. Local free-region computation
  4. Formation optimization (all drones solve identically)
  5. Motion control toward an assigned target

The node is designed to be self-synchronizing: consensus rounds advance
only after all neighbors have published data for the current round, so
no central coordinator is needed for intra-phase synchronization. A
separate CoordinatorNode sequences the *phases* themselves.

Parameters are injected via launch file / YAML config.
"""

import math
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

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from formation_msgs.msg import (
    PhaseCommand,
    DroneStatus,
    HullEstimate,
    DirectionUtilities,
    RegionConstraints,
    FormationTarget,
    MissionConfig,
)

# ---------- Algorithm imports (pure Python) --------------------------------
from formation_control.consensus.graph_utils import (
    validate_adjacency_matrix,
    get_neighbors,
)
from formation_control.consensus.convex_hull import merge_hull_estimates
from formation_control.consensus.preferred_direction import (
    generate_candidate_directions,
    compute_local_utilities,
    merge_direction_utilities,
    select_preferred_direction,
)
from formation_control.consensus.region_consensus import merge_regions
from formation_control.geometry.boxes import Box
from formation_control.geometry.hulls import extract_hull_vertices_3d
from formation_control.geometry.free_space import compute_local_free_region
from formation_control.formation.optimizer import optimize_formation
from formation_control.formation.templates import TEMPLATES
from formation_control.utils.frames import enu_to_ned, ned_to_enu


# ---------------------------------------------------------------------------
# Phase enum -- mirrors PhaseCommand constants
# ---------------------------------------------------------------------------
class Phase(IntEnum):
    IDLE = 0
    TAKEOFF = 1
    HULL_CONSENSUS = 2
    DIRECTION_CONSENSUS = 3
    COMPUTE_FREE_REGION = 4
    REGION_CONSENSUS = 5
    COMPUTE_FORMATION = 6
    ASSIGNMENT = 7
    MOVE_TO_TARGET = 8
    LAND = 9


# ---------------------------------------------------------------------------
# Internal state-machine states (finer-grained than Phase)
# ---------------------------------------------------------------------------
class State(IntEnum):
    BOOT = auto()
    IDLE = auto()
    TAKING_OFF = auto()
    HOVERING = auto()
    HULL_CONSENSUS = auto()
    DIRECTION_CONSENSUS = auto()
    COMPUTE_FREE_REGION = auto()
    REGION_CONSENSUS = auto()
    COMPUTE_FORMATION = auto()
    WAITING_ASSIGNMENT = auto()
    MOVING = auto()
    LANDING = auto()


# ---------------------------------------------------------------------------
# QoS helpers
# ---------------------------------------------------------------------------
def _px4_qos() -> QoSProfile:
    """QoS profile compatible with PX4 uORB bridge (best-effort, keep-last 1)."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def _reliable_qos(depth: int = 10) -> QoSProfile:
    """Standard reliable QoS for inter-drone consensus topics."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _transient_local_qos(depth: int = 1) -> QoSProfile:
    """Transient-local QoS so late joiners receive the last published message."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NAN = float("nan")
CONTROL_RATE_HZ = 50
CONTROL_PERIOD_S = 1.0 / CONTROL_RATE_HZ

# Motion-control gains
KP_MOVE = 1.5          # proportional gain for velocity command
MAX_SPEED = 2.0        # m/s clamp
POSITION_TOL = 0.25    # metres -- "arrived" threshold
TAKEOFF_TOL = 0.30     # metres -- takeoff arrival threshold
LANDING_SPEED = 0.5    # m/s descent rate
LAND_ALT_TOL = 0.20    # metres above ground to consider landed

# Number of offboard setpoints required before engaging offboard mode
OFFBOARD_SETPOINT_COUNT = 15

# Candidate directions for preferred-direction consensus
NUM_CANDIDATE_DIRECTIONS = 26


# ===========================================================================
class DroneNode(Node):
    """ROS2 node that runs on each drone in the formation."""

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------
    def __init__(self):
        super().__init__("drone_node")

        # -- Parameters -----------------------------------------------------
        self.declare_parameter("drone_id", 0)
        self.declare_parameter("namespace", "/uav1")
        self.declare_parameter("instance_id", 1)
        self.declare_parameter("takeoff_altitude", 2.5)
        self.declare_parameter("formation_control.num_drones", 4)
        self.declare_parameter("formation_control.adjacency_flat", [])
        self.declare_parameter("formation_control.obstacles_flat", [])
        self.declare_parameter("formation_control.goal", [0.0, 0.0, 0.0])
        self.declare_parameter("formation_control.sensing_range", 10.0)
        self.declare_parameter("formation_control.workspace_bounds",
                               [-50.0, -50.0, 0.0, 50.0, 50.0, 10.0])
        self.declare_parameter("formation_control.template_name", "square")
        self.declare_parameter("formation_control.tau", 3.0)
        self.declare_parameter("formation_control.move_speed", 2.0)

        self.drone_id: int = self.get_parameter("drone_id").value
        self.ns: str = self.get_parameter("namespace").value
        self.instance_id: int = self.get_parameter("instance_id").value
        self.takeoff_alt: float = self.get_parameter("takeoff_altitude").value
        self.num_drones: int = self.get_parameter("formation_control.num_drones").value

        # Parse adjacency matrix and derive neighbour list
        adj_flat = list(self.get_parameter("formation_control.adjacency_flat").value)
        n = self.num_drones
        self.adjacency = np.array(adj_flat, dtype=np.float64).reshape(n, n)
        validate_adjacency_matrix(self.adjacency)
        self.neighbors: list[int] = get_neighbors(self.adjacency, self.drone_id)
        self.neighbor_set: set[int] = set(self.neighbors)

        # Parse obstacles into Box objects
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
        global MAX_SPEED
        MAX_SPEED = self.get_parameter("formation_control.move_speed").value

        # -- Internal state --------------------------------------------------
        self.state = State.BOOT
        self.phase = Phase.IDLE
        self.phase_complete = False

        # PX4 state
        self.position = np.zeros(3)       # ENU
        self.velocity = np.zeros(3)       # ENU
        self.armed = False
        self.nav_state = 0
        self.offboard_setpoint_counter = 0
        self.offboard_engaged = False

        # Target for motion control (ENU)
        self.target = np.zeros(3)

        # Consensus variables -- hull
        self.hull_estimate: np.ndarray | None = None  # (M, 3) vertices
        self.hull_round = 0
        self.max_hull_rounds = 0
        self.neighbor_hull_data: dict[int, np.ndarray | None] = {}

        # Consensus variables -- direction
        self.direction_utilities: np.ndarray | None = None  # (K,)
        self.candidate_directions: np.ndarray | None = None  # (K, 3)
        self.direction_round = 0
        self.max_direction_rounds = 0
        self.neighbor_direction_data: dict[int, np.ndarray | None] = {}
        self.preferred_direction: np.ndarray | None = None

        # Consensus variables -- region
        self.region_A: np.ndarray | None = None
        self.region_b: np.ndarray | None = None
        self.region_round = 0
        self.max_region_rounds = 0
        self.neighbor_region_data: dict[int, tuple | None] = {}

        # Formation result
        self.formation_slots: np.ndarray | None = None

        # -- Callback group (reentrant so control loop never blocks) ---------
        self.cb_group = ReentrantCallbackGroup()

        # -- PX4 publishers --------------------------------------------------
        self.pub_offboard = self.create_publisher(
            OffboardControlMode,
            f"{self.ns}/fmu/in/offboard_control_mode",
            _px4_qos(),
            callback_group=self.cb_group,
        )
        self.pub_setpoint = self.create_publisher(
            TrajectorySetpoint,
            f"{self.ns}/fmu/in/trajectory_setpoint",
            _px4_qos(),
            callback_group=self.cb_group,
        )
        self.pub_command = self.create_publisher(
            VehicleCommand,
            f"{self.ns}/fmu/in/vehicle_command",
            _px4_qos(),
            callback_group=self.cb_group,
        )

        # -- PX4 subscribers -------------------------------------------------
        self.create_subscription(
            VehicleLocalPosition,
            f"{self.ns}/fmu/out/vehicle_local_position_v1",
            self._vehicle_position_cb,
            _px4_qos(),
            callback_group=self.cb_group,
        )
        self.create_subscription(
            VehicleStatus,
            f"{self.ns}/fmu/out/vehicle_status_v1",
            self._vehicle_status_cb,
            _px4_qos(),
            callback_group=self.cb_group,
        )

        # -- Consensus publishers --------------------------------------------
        self.pub_hull = self.create_publisher(
            HullEstimate,
            f"{self.ns}/hull_estimate",
            _reliable_qos(),
            callback_group=self.cb_group,
        )
        self.pub_direction = self.create_publisher(
            DirectionUtilities,
            f"{self.ns}/direction_utilities",
            _reliable_qos(),
            callback_group=self.cb_group,
        )
        self.pub_region = self.create_publisher(
            RegionConstraints,
            f"{self.ns}/region_constraints",
            _reliable_qos(),
            callback_group=self.cb_group,
        )

        # -- Status publisher ------------------------------------------------
        self.pub_status = self.create_publisher(
            DroneStatus,
            "/formation/drone_status",
            _reliable_qos(depth=20),
            callback_group=self.cb_group,
        )

        # -- Consensus subscribers (one per neighbour) -----------------------
        for nid in self.neighbors:
            ns_neighbor = f"/uav{nid + 1}"
            self.create_subscription(
                HullEstimate,
                f"{ns_neighbor}/hull_estimate",
                self._hull_cb,
                _reliable_qos(),
                callback_group=self.cb_group,
            )
            self.create_subscription(
                DirectionUtilities,
                f"{ns_neighbor}/direction_utilities",
                self._direction_cb,
                _reliable_qos(),
                callback_group=self.cb_group,
            )
            self.create_subscription(
                RegionConstraints,
                f"{ns_neighbor}/region_constraints",
                self._region_cb,
                _reliable_qos(),
                callback_group=self.cb_group,
            )

        # -- Command / target subscribers ------------------------------------
        self.create_subscription(
            PhaseCommand,
            "/formation/phase_command",
            self._phase_command_cb,
            _reliable_qos(),
            callback_group=self.cb_group,
        )
        self.create_subscription(
            FormationTarget,
            f"/formation/drone_{self.drone_id}_target",
            self._target_cb,
            _reliable_qos(),
            callback_group=self.cb_group,
        )

        # -- Main control timer (50 Hz) -------------------------------------
        self.create_timer(
            CONTROL_PERIOD_S,
            self._control_loop,
            callback_group=self.cb_group,
        )

        self.get_logger().info(
            f"DroneNode started: id={self.drone_id}, ns={self.ns}, "
            f"instance={self.instance_id}, neighbors={self.neighbors}"
        )

        # Transition to IDLE once construction is done
        self.state = State.IDLE

    # ===================================================================
    # PX4 callbacks
    # ===================================================================
    def _vehicle_position_cb(self, msg: VehicleLocalPosition):
        """Convert NED position/velocity from PX4 into ENU and store."""
        if not math.isnan(msg.x):
            ex, ey, ez = ned_to_enu(msg.x, msg.y, msg.z)
            self.position = np.array([ex, ey, ez])
        if not math.isnan(msg.vx):
            vex, vey, vez = ned_to_enu(msg.vx, msg.vy, msg.vz)
            self.velocity = np.array([vex, vey, vez])

    def _vehicle_status_cb(self, msg: VehicleStatus):
        self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.nav_state = msg.nav_state

    # ===================================================================
    # Phase command callback
    # ===================================================================
    def _phase_command_cb(self, msg: PhaseCommand):
        """Handle a phase command from the coordinator."""
        incoming = Phase(msg.phase)
        self.get_logger().info(
            f"Phase command received: {incoming.name} (rounds={msg.num_rounds})"
        )
        self.phase = incoming
        self.phase_complete = False

        if incoming == Phase.TAKEOFF:
            self._start_takeoff()
        elif incoming == Phase.HULL_CONSENSUS:
            self._start_hull_consensus(msg.num_rounds)
        elif incoming == Phase.DIRECTION_CONSENSUS:
            self._start_direction_consensus(msg.num_rounds)
        elif incoming == Phase.COMPUTE_FREE_REGION:
            self._start_compute_free_region()
        elif incoming == Phase.REGION_CONSENSUS:
            self._start_region_consensus(msg.num_rounds)
        elif incoming == Phase.COMPUTE_FORMATION:
            self._start_compute_formation()
        elif incoming == Phase.ASSIGNMENT:
            # Coordinator handles assignment; drone waits for target
            self.state = State.WAITING_ASSIGNMENT
        elif incoming == Phase.MOVE_TO_TARGET:
            self._start_move_to_target()
        elif incoming == Phase.LAND:
            self._start_landing()
        elif incoming == Phase.IDLE:
            self.state = State.HOVERING
            self._mark_phase_complete()

    # ===================================================================
    # Target callback (from coordinator after assignment)
    # ===================================================================
    def _target_cb(self, msg: FormationTarget):
        """Receive an assigned formation target from the coordinator."""
        if msg.drone_id != self.drone_id:
            return
        self.target = np.array(msg.target[:3], dtype=np.float64)
        self.get_logger().info(
            f"Received target: [{self.target[0]:.2f}, "
            f"{self.target[1]:.2f}, {self.target[2]:.2f}]"
        )
        # If we were waiting for assignment, mark assignment complete
        if self.state == State.WAITING_ASSIGNMENT:
            self._mark_phase_complete()

    # ===================================================================
    # 50 Hz control loop
    # ===================================================================
    def _control_loop(self):
        """Main control loop running at 50 Hz.

        Responsibilities:
          - Publish offboard control mode and setpoints to PX4
          - Drive state-machine transitions (arrival checks, etc.)
          - Publish DroneStatus
        """
        if self.state == State.BOOT:
            return

        # -- Always publish status ------------------------------------------
        self._publish_status()

        # -- State-dependent control ----------------------------------------
        if self.state == State.IDLE:
            # Send idle setpoints to build up offboard counter
            self._send_idle_setpoint()

        elif self.state == State.TAKING_OFF:
            self._control_takeoff()

        elif self.state == State.HOVERING:
            self._send_hover_setpoint()

        elif self.state in (
            State.HULL_CONSENSUS,
            State.DIRECTION_CONSENSUS,
            State.COMPUTE_FREE_REGION,
            State.REGION_CONSENSUS,
            State.COMPUTE_FORMATION,
            State.WAITING_ASSIGNMENT,
        ):
            # During consensus / computation phases, hold position
            self._send_hover_setpoint()

        elif self.state == State.MOVING:
            self._control_move()

        elif self.state == State.LANDING:
            self._control_landing()

    # ===================================================================
    # PX4 command helpers
    # ===================================================================
    def _publish_vehicle_command(self, command: int, **kwargs):
        """Publish a VehicleCommand to PX4."""
        msg = VehicleCommand()
        msg.command = command
        msg.target_system = self.instance_id
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        for key, value in kwargs.items():
            setattr(msg, key, float(value))
        self.pub_command.publish(msg)

    def _arm(self):
        self.get_logger().info("Sending ARM command")
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )

    def _disarm(self):
        self.get_logger().info("Sending DISARM command")
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0,
        )

    def _engage_offboard_mode(self):
        self.get_logger().info("Engaging offboard mode")
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0,
        )

    def _publish_offboard_mode(self, *, position: bool = False,
                               velocity: bool = False):
        """Publish OffboardControlMode telling PX4 which axes we control."""
        msg = OffboardControlMode()
        msg.position = position
        msg.velocity = velocity
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_offboard.publish(msg)

    def _send_position_setpoint_ned(self, x_ned: float, y_ned: float,
                                    z_ned: float):
        """Send a position-hold setpoint in NED."""
        self._publish_offboard_mode(position=True)
        msg = TrajectorySetpoint()
        msg.position = [x_ned, y_ned, z_ned]
        msg.velocity = [NAN, NAN, NAN]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_setpoint.publish(msg)

    def _send_velocity_setpoint_ned(self, vx: float, vy: float, vz: float):
        """Send a velocity setpoint in NED."""
        self._publish_offboard_mode(velocity=True)
        msg = TrajectorySetpoint()
        msg.position = [NAN, NAN, NAN]
        msg.velocity = [vx, vy, vz]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_setpoint.publish(msg)

    # ===================================================================
    # Idle / Hover helpers
    # ===================================================================
    def _send_idle_setpoint(self):
        """Send zero-velocity setpoint; used to accumulate offboard count."""
        self._send_velocity_setpoint_ned(0.0, 0.0, 0.0)
        self.offboard_setpoint_counter += 1

    def _send_hover_setpoint(self):
        """Hold the current position by sending a position setpoint."""
        nx, ny, nz = enu_to_ned(
            self.position[0], self.position[1], self.position[2]
        )
        self._send_position_setpoint_ned(nx, ny, nz)

    # ===================================================================
    # TAKEOFF
    # ===================================================================
    def _start_takeoff(self):
        """Begin the takeoff sequence.

        Sets the target to (current_x, current_y, takeoff_altitude) in ENU,
        then uses position control to reach it.  Offboard mode and arming
        are handled inside the control loop once enough setpoints have been
        sent.
        """
        self.state = State.TAKING_OFF
        self.target = np.array([
            self.position[0],
            self.position[1],
            self.takeoff_alt,
        ])
        self.offboard_setpoint_counter = 0
        self.offboard_engaged = False
        self.get_logger().info(
            f"Takeoff target (ENU): [{self.target[0]:.2f}, "
            f"{self.target[1]:.2f}, {self.target[2]:.2f}]"
        )

    def _control_takeoff(self):
        """Called every control tick while TAKING_OFF.

        Strategy:
          1. Send position setpoints toward the takeoff target.
          2. After OFFBOARD_SETPOINT_COUNT setpoints, engage offboard + arm.
          3. Once within TAKEOFF_TOL of the target, transition to HOVERING.
        """
        # Convert target to NED and send position setpoint
        nx, ny, nz = enu_to_ned(
            self.target[0], self.target[1], self.target[2]
        )
        self._send_position_setpoint_ned(nx, ny, nz)
        self.offboard_setpoint_counter += 1

        # Engage offboard mode and arm after enough setpoints
        if (self.offboard_setpoint_counter == OFFBOARD_SETPOINT_COUNT
                and not self.offboard_engaged):
            self._engage_offboard_mode()
            self._arm()
            self.offboard_engaged = True

        # Check arrival
        error = np.linalg.norm(self.position - self.target)
        if self.offboard_engaged and error < TAKEOFF_TOL:
            self.get_logger().info("Takeoff complete -- hovering")
            self.state = State.HOVERING
            self._mark_phase_complete()

    # ===================================================================
    # HULL CONSENSUS
    # ===================================================================
    def _start_hull_consensus(self, num_rounds: int):
        """Initialize hull consensus with own position as the seed."""
        self.state = State.HULL_CONSENSUS
        self.max_hull_rounds = max(num_rounds, 1)
        self.hull_round = 0
        # Seed: own position as a single vertex
        self.hull_estimate = self.position.reshape(1, 3).copy()
        self.neighbor_hull_data = {nid: None for nid in self.neighbors}
        self._publish_hull()

    def _publish_hull(self):
        """Broadcast current hull estimate with the current round number."""
        msg = HullEstimate()
        msg.drone_id = self.drone_id
        msg.round_num = self.hull_round
        msg.vertices_flat = self.hull_estimate.flatten().tolist()
        self.pub_hull.publish(msg)

    def _hull_cb(self, msg: HullEstimate):
        """Receive a hull estimate from a neighbour."""
        sender = msg.drone_id
        if sender not in self.neighbor_set:
            return
        if self.state != State.HULL_CONSENSUS:
            return
        if msg.round_num != self.hull_round:
            return  # ignore out-of-sync messages
        vertices = np.array(msg.vertices_flat).reshape(-1, 3)
        self.neighbor_hull_data[sender] = vertices
        self._try_advance_hull()

    def _try_advance_hull(self):
        """Advance hull round if all neighbours have reported."""
        if not all(v is not None for v in self.neighbor_hull_data.values()):
            return
        # Merge own hull with all neighbour hulls
        neighbor_hulls = list(self.neighbor_hull_data.values())
        self.hull_estimate = merge_hull_estimates(
            self.hull_estimate, neighbor_hulls
        )
        self.hull_round += 1
        self.neighbor_hull_data = {nid: None for nid in self.neighbors}

        if self.hull_round >= self.max_hull_rounds:
            self.get_logger().info(
                f"Hull consensus complete after {self.hull_round} rounds, "
                f"{len(self.hull_estimate)} vertices"
            )
            self.state = State.HOVERING
            self._mark_phase_complete()
        else:
            self._publish_hull()

    # ===================================================================
    # DIRECTION CONSENSUS
    # ===================================================================
    def _start_direction_consensus(self, num_rounds: int):
        """Initialize preferred-direction consensus.

        Each drone evaluates a set of candidate directions against local
        obstacles/goal and then runs max-min consensus with neighbours.
        """
        self.state = State.DIRECTION_CONSENSUS
        self.max_direction_rounds = max(num_rounds, 1)
        self.direction_round = 0

        # Generate shared set of candidate directions
        self.candidate_directions = generate_candidate_directions(
            NUM_CANDIDATE_DIRECTIONS
        )

        # Compute local utilities using own position, obstacles, goal
        self.direction_utilities = compute_local_utilities(
            self.position,
            self.obstacles,
            self.candidate_directions,
            max_range=self.sensing_range,
            goal=self.goal,
        )

        self.neighbor_direction_data = {nid: None for nid in self.neighbors}
        self._publish_direction()

    def _publish_direction(self):
        msg = DirectionUtilities()
        msg.drone_id = self.drone_id
        msg.round_num = self.direction_round
        msg.utilities = self.direction_utilities.tolist()
        self.pub_direction.publish(msg)

    def _direction_cb(self, msg: DirectionUtilities):
        sender = msg.drone_id
        if sender not in self.neighbor_set:
            return
        if self.state != State.DIRECTION_CONSENSUS:
            return
        if msg.round_num != self.direction_round:
            return
        self.neighbor_direction_data[sender] = np.array(msg.utilities)
        self._try_advance_direction()

    def _try_advance_direction(self):
        if not all(v is not None for v in self.neighbor_direction_data.values()):
            return
        neighbor_utils = list(self.neighbor_direction_data.values())
        self.direction_utilities = merge_direction_utilities(
            self.direction_utilities, neighbor_utils
        )
        self.direction_round += 1
        self.neighbor_direction_data = {nid: None for nid in self.neighbors}

        if self.direction_round >= self.max_direction_rounds:
            self.preferred_direction = select_preferred_direction(
                self.direction_utilities, self.candidate_directions
            )
            self.get_logger().info(
                f"Direction consensus complete: "
                f"theta*=[{self.preferred_direction[0]:.3f}, "
                f"{self.preferred_direction[1]:.3f}, "
                f"{self.preferred_direction[2]:.3f}]"
            )
            self.state = State.HOVERING
            self._mark_phase_complete()
        else:
            self._publish_direction()

    # ===================================================================
    # COMPUTE FREE REGION (local, no networking)
    # ===================================================================
    def _start_compute_free_region(self):
        """Compute the local obstacle-free convex region.

        Uses the converged hull vertices, obstacles, workspace bounds,
        and preferred direction to build a polyhedral safe region
        {x : Ax <= b}.
        """
        self.state = State.COMPUTE_FREE_REGION

        if self.hull_estimate is None or self.preferred_direction is None:
            self.get_logger().error(
                "Cannot compute free region: missing hull or direction"
            )
            self._mark_phase_complete()
            return

        # Centroid of the converged hull
        centroid = self.hull_estimate.mean(axis=0)

        A, b = compute_local_free_region(
            centroid=centroid,
            hull_vertices=self.hull_estimate,
            obstacles=self.obstacles,
            workspace_bounds=self.workspace_bounds,
            theta_star=self.preferred_direction,
            tau=self.tau,
        )
        self.region_A = A
        self.region_b = b
        self.get_logger().info(
            f"Free region computed: {A.shape[0]} constraints"
        )
        self.state = State.HOVERING
        self._mark_phase_complete()

    # ===================================================================
    # REGION CONSENSUS
    # ===================================================================
    def _start_region_consensus(self, num_rounds: int):
        """Run distributed consensus on the polyhedral safe region.

        Each drone intersects its local region with neighbours' regions.
        """
        self.state = State.REGION_CONSENSUS
        self.max_region_rounds = max(num_rounds, 1)
        self.region_round = 0

        if self.region_A is None or self.region_b is None:
            self.get_logger().error(
                "Cannot start region consensus: no local region"
            )
            self._mark_phase_complete()
            return

        self.neighbor_region_data = {nid: None for nid in self.neighbors}
        self._publish_region()

    def _publish_region(self):
        msg = RegionConstraints()
        msg.drone_id = self.drone_id
        msg.round_num = self.region_round
        msg.num_constraints = self.region_A.shape[0]
        msg.a_matrix_flat = self.region_A.flatten().tolist()
        msg.b_vector = self.region_b.tolist()
        self.pub_region.publish(msg)

    def _region_cb(self, msg: RegionConstraints):
        sender = msg.drone_id
        if sender not in self.neighbor_set:
            return
        if self.state != State.REGION_CONSENSUS:
            return
        if msg.round_num != self.region_round:
            return
        m = msg.num_constraints
        A = np.array(msg.a_matrix_flat).reshape(m, 3)
        b = np.array(msg.b_vector)
        self.neighbor_region_data[sender] = (A, b)
        self._try_advance_region()

    def _try_advance_region(self):
        if not all(v is not None for v in self.neighbor_region_data.values()):
            return
        neighbor_As = [data[0] for data in self.neighbor_region_data.values()]
        neighbor_bs = [data[1] for data in self.neighbor_region_data.values()]
        self.region_A, self.region_b = merge_regions(
            self.region_A, self.region_b, neighbor_As, neighbor_bs
        )
        self.region_round += 1
        self.neighbor_region_data = {nid: None for nid in self.neighbors}

        if self.region_round >= self.max_region_rounds:
            self.get_logger().info(
                f"Region consensus complete: "
                f"{self.region_A.shape[0]} constraints"
            )
            self.state = State.HOVERING
            self._mark_phase_complete()
        else:
            self._publish_region()

    # ===================================================================
    # COMPUTE FORMATION
    # ===================================================================
    def _start_compute_formation(self):
        """Compute formation slots inside the agreed safe region.

        All drones that converged on the same region will compute the
        same formation result, so no additional communication is needed.
        """
        self.state = State.COMPUTE_FORMATION

        if self.region_A is None or self.preferred_direction is None:
            self.get_logger().error(
                "Cannot compute formation: missing region or direction"
            )
            self._mark_phase_complete()
            return

        template_fn = TEMPLATES.get(self.template_name)
        if template_fn is None:
            self.get_logger().error(
                f"Unknown template: {self.template_name}"
            )
            self._mark_phase_complete()
            return
        template = template_fn()

        centroid = (
            self.hull_estimate.mean(axis=0)
            if self.hull_estimate is not None
            else self.position
        )

        result = optimize_formation(
            self.region_A,
            self.region_b,
            template,
            self.preferred_direction,
            n_rotations=16,
            centroid=centroid,
            tau=self.tau,
        )

        if result is None:
            self.get_logger().warn("Formation optimizer returned no solution")
            self.formation_slots = None
        else:
            center, scale, angle, slots = result
            self.formation_slots = slots
            self.get_logger().info(
                f"Formation computed: center={center}, scale={scale:.2f}, "
                f"angle={angle:.2f} rad, {len(slots)} slots"
            )

        self.state = State.HOVERING
        self._mark_phase_complete()

    # ===================================================================
    # MOVE TO TARGET
    # ===================================================================
    def _start_move_to_target(self):
        """Begin moving toward the assigned target."""
        self.state = State.MOVING
        self.get_logger().info(
            f"Moving to target: [{self.target[0]:.2f}, "
            f"{self.target[1]:.2f}, {self.target[2]:.2f}]"
        )

    def _control_move(self):
        """PD velocity controller toward self.target.

        Computes a proportional velocity command in ENU, clamps its
        magnitude, converts to NED, and sends as a velocity setpoint.
        """
        error_vec = self.target - self.position
        distance = np.linalg.norm(error_vec)

        if distance < POSITION_TOL:
            self.get_logger().info("Arrived at target -- hovering")
            self.state = State.HOVERING
            self._mark_phase_complete()
            return

        # Proportional velocity command (ENU)
        v_cmd = KP_MOVE * error_vec
        speed = np.linalg.norm(v_cmd)
        if speed > MAX_SPEED:
            v_cmd = v_cmd * (MAX_SPEED / speed)

        # Convert to NED and send
        vx_ned, vy_ned, vz_ned = enu_to_ned(v_cmd[0], v_cmd[1], v_cmd[2])
        self._send_velocity_setpoint_ned(vx_ned, vy_ned, vz_ned)

    # ===================================================================
    # LANDING
    # ===================================================================
    def _start_landing(self):
        """Begin a controlled descent."""
        self.state = State.LANDING
        self.get_logger().info("Starting landing sequence")

    def _control_landing(self):
        """Descend at a fixed rate until near ground, then disarm.

        Uses velocity control to descend at LANDING_SPEED m/s in ENU
        (negative z in NED).
        """
        if self.position[2] < LAND_ALT_TOL:
            self.get_logger().info("Landed -- disarming")
            self._disarm()
            self.state = State.IDLE
            self._mark_phase_complete()
            return

        # Descend straight down (ENU: negative z velocity)
        vx_ned, vy_ned, vz_ned = enu_to_ned(0.0, 0.0, -LANDING_SPEED)
        self._send_velocity_setpoint_ned(vx_ned, vy_ned, vz_ned)

    # ===================================================================
    # Status / completion helpers
    # ===================================================================
    def _mark_phase_complete(self):
        """Flag the current phase as complete so the coordinator can proceed."""
        self.phase_complete = True
        self.get_logger().info(f"Phase {self.phase.name} complete")

    def _publish_status(self):
        """Publish current drone status at every control tick."""
        msg = DroneStatus()
        msg.drone_id = self.drone_id
        msg.current_phase = int(self.phase)
        msg.phase_complete = self.phase_complete
        msg.position = self.position.tolist()
        msg.velocity = self.velocity.tolist()
        self.pub_status.publish(msg)


# ===========================================================================
# Entry point
# ===========================================================================
def main(args=None):
    rclpy.init(args=args)
    node = DroneNode()

    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
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
