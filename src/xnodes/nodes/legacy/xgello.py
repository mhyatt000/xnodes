from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from gello.agents.gello_agent import DynamixelRobotConfig, GelloAgent
from gello.env import RobotEnv
from gello.zmq_core.robot_node import ZMQClientRobot
import numpy as np
import rclpy
from sensor_msgs.msg import JointState
import tyro
import yaml

from xnodes.nodes.legacy.base import Base


@dataclass
class GelloArgs:
    agent: str = "gello"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_type: str | None = None
    hz: int = 100
    start_joints: tuple[float, ...] | None = None
    torque: bool = False

    gello_port: str | None = None
    gello_config: Path | None = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False
    safety: bool = True


class Gello(Base):
    def __init__(self, args: GelloArgs | None = None, env: RobotEnv | None = None) -> None:
        super().__init__("gello")
        args = args or GelloArgs()
        logger = self.get_logger()
        self.env = env

        self.port = args.gello_port or self._find_port()
        logger.info(f"Using port {self.port}")

        config_path = resolve_gello_config_path(args.gello_config)
        dynamixel_config = load_dynamixel_config(config_path, port=self.port)
        if dynamixel_config is None:
            msg = f"{config_path} must define a GELLO dynamixel_config"
            raise ValueError(msg)
        logger.info(f"Using local GELLO config {config_path}")

        self.agent = GelloAgent(
            port=self.port,
            dynamixel_config=dynamixel_config,
            start_joints=args.start_joints,
        )
        self.agent._robot.set_torque_mode(args.torque)

        logger.info("Gello Agent Created.")

        self.hz = 50
        self.pub = self.create_publisher(JointState, "/gello/state", 10)
        self.timer = self.create_timer(1 / self.hz, self.run)

        logger.info("Gello Node Initialized.")
        self.set_period()

    def _find_port(self) -> str:
        usb_ports = sorted(Path("/dev/serial/by-id").glob("*"))
        self.get_logger().info(f"Found {len(usb_ports)} ports")
        if usb_ports:
            return str(usb_ports[0])
        msg = "No gello port found, please specify one or plug in gello"
        raise ValueError(msg)

    def run(self) -> None:
        action = self.agent.act()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint{i + 1}" for i in range(len(action))]
        msg.position = action.tolist()
        self.pub.publish(msg)

        # self.get_logger().info(f"{msg.name}")
        # rad to deg
        # no scientific notation, round to 2 decimals
        np.set_printoptions(precision=1, suppress=True)
        self.get_logger().info(f"{(np.rad2deg(action[:-1]))} deg - {action[-1]:.2f} gripper")

        if self.env is not None:
            self.env.step(action)


def build_env(args: GelloArgs) -> RobotEnv:
    reset_joints = None if args.start_joints is None else np.array(args.start_joints)
    if reset_joints is None:
        reset_joints = np.deg2rad([0, -90, 90, -90, -90, 0, 0])

    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict={})
    obs = env.get_obs()
    print(obs)

    curr_joints = obs["joint_positions"]
    if reset_joints.shape == curr_joints.shape:
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
            time.sleep(0.001)

    return env


def resolve_gello_config_path(path: Path | None) -> Path:
    if path is not None:
        return path.expanduser()

    cwd_config = Path("config/gello.yaml")
    if cwd_config.exists():
        return cwd_config

    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("xnodes")) / "config" / "gello.yaml"
    except Exception:
        return cwd_config


def load_dynamixel_config(path: Path, port: str | None = None) -> DynamixelRobotConfig | None:
    if not path.exists():
        return None

    data = yaml.safe_load(path.read_text()) or {}
    valid_ports = data.get("valid_ports", [])
    if port is not None and valid_ports and port not in valid_ports:
        ports = "\n  ".join(str(p) for p in valid_ports)
        msg = f"{path} does not define GELLO port {port}. Valid ports:\n  {ports}"
        raise ValueError(msg)

    cfg = data.get("dynamixel_config") or data.get("dynamixel") or data.get("agent", {}).get("dynamixel_config")
    if cfg is None:
        return None

    offsets = cfg.get("joint_offsets")
    if offsets is None and "joint_offsets_pi" in cfg:
        offsets = [float(x) * np.pi for x in cfg["joint_offsets_pi"]]
    if offsets is None and "joint_offsets_pi_over_2" in cfg:
        offsets = [float(x) * np.pi / 2 for x in cfg["joint_offsets_pi_over_2"]]
    if offsets is None:
        msg = f"{path} must define joint_offsets, joint_offsets_pi, or joint_offsets_pi_over_2"
        raise ValueError(msg)

    return DynamixelRobotConfig(
        joint_ids=tuple(int(x) for x in cfg["joint_ids"]),
        joint_offsets=tuple(float(x) for x in offsets),
        joint_signs=tuple(int(x) for x in cfg["joint_signs"]),
        gripper_config=tuple(cfg["gripper_config"]),
    )


def startup(agent: GelloAgent, env: RobotEnv, args: GelloArgs) -> None:
    print("Going to start position")
    start_pos = agent.act()
    print("start_pos", start_pos)

    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}")

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(joints), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act()
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joint_positions"]
    action = agent.act()
    if (action - joints > 0.5).any():
        print("Action is too big")
        joint_indices = np.where(action - joints > 0.8)[0]
        for joint_index in joint_indices:
            print(
                f"Joint [{joint_index}], leader: {action[joint_index]}, "
                f"follower: {joints[joint_index]}, diff: {action[joint_index] - joints[joint_index]}"
            )
            print(f"leader: {action}")
            print(f"follower: {joints}")
        if args.safety:
            raise SystemExit(1)


def main(args: GelloArgs) -> None:
    rclpy.init(args=None)
    node = Gello(args, env=None)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Environment Node shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(tyro.cli(GelloArgs))
