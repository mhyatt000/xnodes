from __future__ import annotations

from pathlib import Path

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    default_params = str(Path(__file__).resolve().parents[1] / "config" / "fg_record.yaml")

    node = Node(
        package="xnodes",
        executable="fg_record_flex",
        name="fg_record_flex",
        output="screen",
        emulate_tty=True,
        parameters=[params_file],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description=(
                    "Path to fg_record_flex params YAML; recordings are named "
                    "YYMMDD-hhmmss_<episode index>_<note>.mcap"
                ),
            ),
            node,
        ]
    )
