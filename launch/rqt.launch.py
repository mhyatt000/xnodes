from __future__ import annotations

from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="rqt_image_view",
                executable="rqt_image_view",
                name="image_view",
                output="screen",
            )
        ]
    )
