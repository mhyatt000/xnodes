from launch import LaunchDescription
from launch_ros.actions import Node, PushRosNamespace
from launch.actions import GroupAction

from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

from pathlib import Path
from launch import LaunchDescription
from launch_ros.actions import Node
from rich import print
import os

def generate_launch_description():

    # declare fps
    # fps = DeclareLaunchArgument(
        # "fps",
        # default_value="30",
        # description="Frames per second for the camera",
    # )

    cam_links = sorted(Path("/dev").glob("cam*"))
    if not cam_links:
        raise RuntimeError("No /dev/cam* symlinks found")

    print(cam_links)
    fps, width, height = 30.0, 640, 480
    nodes = []

    root = Path(os.environ["PIXI_PROJECT_ROOT"])
    params_file = root / "config" / "cam.yaml"

    for i, dev in enumerate(cam_links):
        name = dev.name.replace("cam_", "")
        dev = dev.resolve()
        vname = dev.name
        nodes.append(
            Node(
                package="usb_cam",
                executable="usb_cam_node_exe",
                name=f"{name}_camera",
                namespace=name,
                parameters=[
                params_file,
                {
                "output_encoding": "rgb8",
                "video_device": str(dev),
                "image_size": [640, 480],
                "frame_id": f"{name}_optical_frame",
                "camera_info_url": f"file://{root}/config/camera_info/{name}.yaml",
                "camera_name": name,
                "framerate": fps,
                }],
                arguments=["--ros-args", "--log-level", "info"],
            )
        )

    # global namespace grouping
    group = GroupAction([
        PushRosNamespace("cam"),
        *nodes,
        ])

    return LaunchDescription([group])

