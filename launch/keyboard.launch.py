from __future__ import annotations

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    topic = LaunchConfiguration("topic")
    step = LaunchConfiguration("step")
    poll_hz = LaunchConfiguration("poll_hz")

    keyboard = Node(
        package="xnodes",
        executable="keyboard",
        output="screen",
        emulate_tty=True,
        parameters=[{"topic": topic, "step": step, "poll_hz": poll_hz}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("topic", default_value="/ctrl/twist", description="Output Twist topic"),
            DeclareLaunchArgument("step", default_value="0.25", description="Linear velocity per key press"),
            DeclareLaunchArgument("poll_hz", default_value="60.0", description="Keyboard polling rate"),
            keyboard,
        ]
    )
