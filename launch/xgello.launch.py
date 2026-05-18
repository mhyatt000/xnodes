from __future__ import annotations

from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _append_option(cmd: list[str], name: str, value: str) -> None:
    if value:
        cmd.extend([name, value])


def _append_flag(cmd: list[str], name: str, enabled: bool) -> None:
    if enabled:
        cmd.append(name)


def _build_xgello(_context, *_, **__):
    hz = LaunchConfiguration("hz").perform(_context)
    robot_port = LaunchConfiguration("robot_port").perform(_context)
    hostname = LaunchConfiguration("hostname").perform(_context)
    gello_port = LaunchConfiguration("gello_port").perform(_context)
    gello_config = LaunchConfiguration("gello_config").perform(_context)
    robot_type = LaunchConfiguration("robot_type").perform(_context)
    start_joints = LaunchConfiguration("start_joints").perform(_context)

    torque = _truthy(LaunchConfiguration("torque").perform(_context))
    mock = _truthy(LaunchConfiguration("mock").perform(_context))
    use_save_interface = _truthy(LaunchConfiguration("use_save_interface").perform(_context))
    bimanual = _truthy(LaunchConfiguration("bimanual").perform(_context))
    verbose = _truthy(LaunchConfiguration("verbose").perform(_context))
    safety = _truthy(LaunchConfiguration("safety").perform(_context))

    cmd = [
        "python3",
        "-m",
        "xnodes.nodes.legacy.xgello",
        "--hz",
        hz,
        "--robot-port",
        robot_port,
        "--hostname",
        hostname,
    ]

    _append_option(cmd, "--gello-port", gello_port)
    _append_option(cmd, "--gello-config", gello_config)
    _append_option(cmd, "--robot-type", robot_type)
    _append_option(cmd, "--start-joints", start_joints)
    _append_flag(cmd, "--torque", torque)
    _append_flag(cmd, "--mock", mock)
    _append_flag(cmd, "--use-save-interface", use_save_interface)
    _append_flag(cmd, "--bimanual", bimanual)
    _append_flag(cmd, "--verbose", verbose)
    _append_flag(cmd, "--no-safety", not safety)

    return [ExecuteProcess(cmd=cmd, output="screen", emulate_tty=True)]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("hz", default_value="100", description="Robot control rate used by the CLI args"),
            DeclareLaunchArgument("robot_port", default_value="6001", description="ZMQ robot port"),
            DeclareLaunchArgument("hostname", default_value="127.0.0.1", description="Robot host"),
            DeclareLaunchArgument("gello_port", default_value="", description="Serial port for the Gello device"),
            DeclareLaunchArgument(
                "gello_config",
                default_value=PathJoinSubstitution([FindPackageShare("xnodes"), "config", "gello.yaml"]),
                description="Local GELLO Dynamixel config YAML",
            ),
            DeclareLaunchArgument(
                "robot_type", default_value="", description="Optional robot type passed through to the CLI"
            ),
            DeclareLaunchArgument(
                "start_joints",
                default_value="",
                description="Optional start joints string passed through to tyro",
            ),
            DeclareLaunchArgument("torque", default_value="false", choices=["true", "false"]),
            DeclareLaunchArgument("mock", default_value="false", choices=["true", "false"]),
            DeclareLaunchArgument("use_save_interface", default_value="false", choices=["true", "false"]),
            DeclareLaunchArgument("bimanual", default_value="false", choices=["true", "false"]),
            DeclareLaunchArgument("verbose", default_value="false", choices=["true", "false"]),
            DeclareLaunchArgument("safety", default_value="true", choices=["true", "false"]),
            OpaqueFunction(function=_build_xgello),
        ]
    )
