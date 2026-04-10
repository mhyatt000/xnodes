from __future__ import annotations

from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

from launch import LaunchDescription
from xnodes.launching.robot_command import build_legacy_robot_launch_source


def generate_launch_description() -> LaunchDescription:
    input_mode = LaunchConfiguration("input")
    ctrl_mode = LaunchConfiguration("ctrl")
    backend = LaunchConfiguration("backend")
    robot_ip = LaunchConfiguration("robot_ip")
    service_ns = LaunchConfiguration("service_ns")

    robot = ExecuteProcess(
        cmd=[
            "python3",
            "-c",
            build_legacy_robot_launch_source(
                input_mode=input_mode,
                ctrl_mode=ctrl_mode,
                backend=backend,
                robot_ip=robot_ip,
                service_ns=service_ns,
            ),
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("input", default_value="gello", choices=["gello", "spacemouse", "model"]),
            DeclareLaunchArgument("ctrl", default_value="joint", choices=["joint", "cartesian"]),
            DeclareLaunchArgument("backend", default_value="auto", choices=["auto", "service", "fake", "sdk"]),
            DeclareLaunchArgument("robot_ip", default_value="192.168.1.231"),
            DeclareLaunchArgument("service_ns", default_value="/xarm"),
            robot,
        ]
    )
