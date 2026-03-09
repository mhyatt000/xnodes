from __future__ import annotations

from launch.actions import ExecuteProcess

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    box = ExecuteProcess(
        cmd=["python3", "-m", "xnodes.nodes.box"],
        output="screen",
    )

    return LaunchDescription([box])
