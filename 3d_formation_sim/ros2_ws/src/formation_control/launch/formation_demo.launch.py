"""Launch file for the 3D distributed formation control demo.

Starts:
  - Four MicroXRCE-DDS agents (one per drone, UDP ports 8888/8890/8892/8894)
  - Four drone_node instances (namespaced /uav1 through /uav4)
  - One coordinator_node
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('formation_control'), 'config', 'params.yaml'
    )

    # Determine MicroXRCE-DDS agent executable path
    custom_agent = '/home/ethan/drone_sims/MicroXRCEAgent'
    if os.path.isfile(custom_agent) and os.access(custom_agent, os.X_OK):
        agent_exec = custom_agent
    else:
        agent_exec = 'MicroXRCEAgent'

    # UDP ports for each drone instance
    agent_ports = [8888, 8890, 8892, 8894]

    actions = []

    # --- MicroXRCE-DDS agents ---
    for i, port in enumerate(agent_ports):
        actions.append(
            ExecuteProcess(
                cmd=[agent_exec, 'udp4', '-p', str(port)],
                name=f'uxrce_agent_{i}',
                output='screen',
            )
        )

    # --- Drone nodes ---
    for i in range(4):
        actions.append(
            Node(
                package='formation_control',
                executable='drone_node',
                name=f'drone_node_{i}',
                parameters=[
                    config,
                    {
                        'drone_id': i,
                        'namespace': f'/uav{i + 1}',
                        'instance_id': i + 1,
                        'takeoff_altitude': 3.0,
                    },
                ],
                output='screen',
            )
        )

    # --- Coordinator node ---
    actions.append(
        Node(
            package='formation_control',
            executable='coordinator_node',
            name='coordinator_node',
            parameters=[config],
            output='screen',
        )
    )

    return LaunchDescription(actions)
