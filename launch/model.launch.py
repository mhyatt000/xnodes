from __future__ import annotations

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    model_host = LaunchConfiguration("model_host")
    model_port = LaunchConfiguration("model_port")
    model_rep = LaunchConfiguration("model_rep")
    model_task = LaunchConfiguration("model_task")
    model_ensemble = LaunchConfiguration("model_ensemble")

    model = Node(
        package="xnodes",
        executable="model",
        output="screen",
        arguments=[
            "--host",
            model_host,
            "--port",
            model_port,
            "--rep",
            model_rep,
            "--task",
            model_task,
            "--ensemble",
            model_ensemble,
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("model_host", default_value="none"),
            DeclareLaunchArgument("model_port", default_value="8000"),
            DeclareLaunchArgument("model_rep", default_value="relative", choices=["relative", "absolute"]),
            DeclareLaunchArgument("model_task", default_value="none"),
            DeclareLaunchArgument("model_ensemble", default_value="false", choices=["true", "false"]),
            model,
        ]
    )
