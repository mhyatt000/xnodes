from __future__ import annotations

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    recorder_params_file = LaunchConfiguration("recorder_params_file")
    footpedal_params_file = LaunchConfiguration("footpedal_params_file")

    recorder = Node(
        package="xnodes",
        executable="recorder",
        name="recorder",
        output="screen",
        emulate_tty=True,
        parameters=[recorder_params_file],
    )

    footpedal = Node(
        package="xnodes",
        executable="footpedal",
        output="screen",
        emulate_tty=True,
        parameters=[footpedal_params_file],
    )

    rqt = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="image_view",
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "recorder_params_file",
                default_value=PathJoinSubstitution([FindPackageShare("xnodes"), "config", "hrec.yaml"]),
                description="Path to camera-only recorder params YAML",
            ),
            DeclareLaunchArgument(
                "footpedal_params_file",
                default_value=PathJoinSubstitution([FindPackageShare("xnodes"), "config", "pedal.yaml"]),
                description="Path to foot pedal params YAML",
            ),
            recorder,
            footpedal,
            rqt,
        ]
    )
