# 3D Distributed Formation Control with PX4 Quadrotors

A ROS2-based implementation of distributed formation control for a team of PX4 quadrotors navigating through environments with obstacles. The system uses consensus-based algorithms for convex hull estimation, preferred direction selection, free-region computation, and formation optimization, following the pipeline described in Alonso-Mora et al. Each drone runs its own ROS2 node and communicates with neighbors over topics, while a coordinator node manages phase transitions across the team.

## Prerequisites

- **ROS2** (Humble or Iron)
- **PX4-Autopilot** (v1.14+)
- **Gazebo** (Garden or Harmonic)
- **Python 3.10+**
- **MicroXRCE-DDS Agent**

### Python dependencies

```
numpy
scipy
cvxpy
```

## Quick Start

1. Clone `px4_msgs` next to the workspace source directory:
   ```bash
   cd ros2_ws
   git clone https://github.com/PX4/px4_msgs.git
   ```

2. Build the workspace (including px4_msgs):
   ```bash
   cd ros2_ws
   colcon build --paths src/* px4_msgs
   ```

3. Start PX4 SITL instances (one terminal per drone):
   ```bash
   bash scripts/start_sim.sh 0
   bash scripts/start_sim.sh 1
   bash scripts/start_sim.sh 2
   bash scripts/start_sim.sh 3
   ```

4. Launch the formation control nodes:
   ```bash
   source ros2_ws/install/setup.bash
   ros2 launch formation_control formation_demo.launch.py
   ```
