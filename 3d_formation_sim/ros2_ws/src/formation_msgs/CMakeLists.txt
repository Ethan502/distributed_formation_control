cmake_minimum_required(VERSION 3.8)
project(formation_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PhaseCommand.msg"
  "msg/DroneStatus.msg"
  "msg/HullEstimate.msg"
  "msg/DirectionUtilities.msg"
  "msg/RegionConstraints.msg"
  "msg/FormationTarget.msg"
  "msg/MissionConfig.msg"
)

ament_package()
