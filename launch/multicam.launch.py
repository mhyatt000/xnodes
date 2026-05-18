from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from rich import print
import yaml

from launch import LaunchDescription


def _launch_setup(context):
    params_file = Path(LaunchConfiguration("params_file").perform(context))
    cfg = yaml.safe_load(params_file.read_text()) if params_file.exists() else {}
    matches = cfg.get("launch", {}).get("ros__parameters", {}).get("matches", "cam_*")

    _links = Path().home() / ".xnodes" / "dev"
    if isinstance(matches, list):
        cam_links = sorted(p for name in matches for p in _links.glob(f"cam_{name}"))
    else:
        cam_links = sorted(_links.glob(matches))
    print(_links, cam_links)
    if not cam_links:
        raise RuntimeError("No /dev/cam* symlinks found")

    fps, width, height = 30.0, 640, 480
    nodes = []

    config = Path().home().resolve() / ".xnodes"

    for i, dev in enumerate(cam_links):
        name = dev.name.replace("cam_", "")
        resolved = dev.resolve()
        nodes.append(
            Node(
                package="usb_cam",
                executable="usb_cam_node_exe",
                name=f"{name}_camera",
                namespace=name,
                parameters=[
                    {
                        "output_encoding": "rgb8",
                        "video_device": str(resolved),
                        "image_size": [640, 480],
                        "frame_id": f"{name}_optical_frame",
                        "camera_info_url": f"file://{config}/camera_info/{name}.yaml",
                        "camera_name": name,
                        # "framerate": fps,
                    },
                ],
                arguments=["--ros-args", "--log-level", "info"],
            )
        )

    group = GroupAction([PushRosNamespace("cam"), *nodes])
    return [group]


def generate_launch_description():
    pkg_share = Path(get_package_share_directory("xnodes"))
    default_params_file = str(pkg_share / "config" / "multicam.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params_file),
            OpaqueFunction(function=_launch_setup),
        ]
    )
