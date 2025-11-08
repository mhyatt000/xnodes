from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node, ComposableNodeContainer, LoadComposableNodes, PushRosNamespace
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ns = LaunchConfiguration("ns")
    use_rviz = LaunchConfiguration("use_rviz")
    log_level = LaunchConfiguration("log_level")

    declare_ns = DeclareLaunchArgument("ns", default_value="", description="ROS namespace")
    declare_use_rviz = DeclareLaunchArgument("use_rviz", default_value="false", choices=["true","false"])
    declare_log = DeclareLaunchArgument("log_level", default_value="info")

    params_file = PathJoinSubstitution(
        [get_package_share_directory("your_pkg"), "config", "params.yaml"]
    )

    # regular node
    cam_node = Node(
        package="your_pkg",
        executable="camera_node",
        name="camera",
        namespace=ns,
        output="screen",
        emulate_tty=True,
        parameters=[params_file, {"frame_id": "cam_frame"}],
        remappings=[("image_raw", "image"), ("camera_info", "info")],
        arguments=["--ros-args", "--log-level", log_level],
        respawn=True,
        respawn_delay=2.0,
    )

    # composable graph
    container = ComposableNodeContainer(
        name="your_container",
        namespace=ns,
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        emulate_tty=True,
        arguments=["--ros-args", "--log-level", log_level],
    )

    load_nodes = LoadComposableNodes(
        target_container=container,
        composable_node_descriptions=[
            ComposableNode(
                package="your_pkg",
                plugin="your_pkg::FilterComponent",
                name="filter",
                parameters=[{"alpha": 0.7}],
                remappings=[("in/image", "image"), ("out/image", "image_filtered")],
            ),
            ComposableNode(
                package="your_pkg",
                plugin="your_pkg::DetectorComponent",
                name="detector",
                parameters=[params_file],
                remappings=[("image", "image_filtered"), ("detections", "detections")],
            ),
        ],
    )

    # optional RViz include
    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [get_package_share_directory("your_pkg"), "/launch/rviz.launch.py"]
        ),
        condition=IfCondition(use_rviz),
    )

    # namespace group (keeps TF and topics clean)
    in_namespace = GroupAction([
        PushRosNamespace(ns),
        cam_node,
        container,
        load_nodes,
        rviz_launch,
    ])

    # example env pin
    domain_env = SetEnvironmentVariable("ROS_DOMAIN_ID", "42")

    return LaunchDescription([
        domain_env,
        declare_ns,
        declare_use_rviz,
        declare_log,
        in_namespace,
    ])

