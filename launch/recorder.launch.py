from __future__ import annotations

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")

    recorder = Node(
        package="xnodes",
        executable="recorder",
        output="screen",
        emulate_tty=True,
        parameters=[params_file],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=[FindPackageShare("xnodes"), "/config/record.yaml"],
                description="Path to recorder params YAML",
            ),
            recorder,
        ]
    )
