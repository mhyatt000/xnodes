from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    # Parameters for each camera
    cam_ids = ["left", "right"]
    device_paths = ["/dev/video0", "/dev/video1"]

    # Common parameters
    fps = 30
    width = 1280
    height = 720

    nodes = []
    for name, dev in zip(cam_ids, device_paths):
        nodes.append(
            Node(
                package="usb_camera_driver",
                executable="usb_camera_driver_node",
                name=f"{name}_camera",
                namespace=name,
                parameters=[{
                    "camera_name": name,
                    "camera_frame_id": f"{name}_optical_frame",
                    "video_device": dev,
                    "framerate": fps,
                    "image_width": width,
                    "image_height": height,
                }],
                output="screen",
            )
        )

    return LaunchDescription(nodes)

