#!/bin/bash
# Start PX4 SITL instance for formation control demo.
# Usage: bash start_sim.sh <instance_id>  (0-3)

INSTANCE=${1:-0}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Path to PX4 — adjust if your PX4-Autopilot is elsewhere
PX4_DIR="${PX4_DIR:-$HOME/drone_sims/PX4-Autopilot}"

if [ ! -d "$PX4_DIR" ]; then
    echo "ERROR: PX4-Autopilot not found at $PX4_DIR"
    echo "Set PX4_DIR env var to your PX4-Autopilot path"
    exit 1
fi

# Point Gazebo at our custom worlds directory
export PX4_GZ_WORLDS="$PROJECT_DIR/ros2_ws/src/formation_control/worlds"
export PX4_GZ_MODEL=x500
export PX4_GZ_WORLD=obstacle_corridor

# Spawn positions (ENU-ish: x,y,z,roll,pitch,yaw)
case $INSTANCE in
    0) export PX4_GZ_MODEL_POSE="-12,-3,0,0,0,0" ;;
    1) export PX4_GZ_MODEL_POSE="-12,-1,0,0,0,0" ;;
    2) export PX4_GZ_MODEL_POSE="-12,1,0,0,0,0" ;;
    3) export PX4_GZ_MODEL_POSE="-12,3,0,0,0,0" ;;
    *) echo "ERROR: Instance must be 0-3"; exit 1 ;;
esac

export PX4_GZ_MODEL_NAME="uav$((INSTANCE+1))"
export PX4_SYS_AUTOSTART=4001

echo "====================================="
echo " Starting PX4 SITL instance $INSTANCE"
echo " Model: x500 as $PX4_GZ_MODEL_NAME"
echo " World: obstacle_corridor"
echo " Pose: $PX4_GZ_MODEL_POSE"
echo "====================================="

cd "$PX4_DIR"

# Build PX4 if not already built
if [ ! -f "build/px4_sitl_default/bin/px4" ]; then
    echo "PX4 not built. Building first (this may take a while)..."
    make px4_sitl_default
fi

# Run PX4 SITL
./build/px4_sitl_default/bin/px4 -i $INSTANCE
