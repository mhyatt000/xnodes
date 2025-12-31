from __future__ import annotations

from dataclasses import dataclass
import glob
import time

import draccus # TODO replace draccus with tyro
from gello.agents.gello_agent import GelloAgent
from gello.env import RobotEnv
from gello.zmq_core.robot_node import ZMQClientRobot
import numpy as np
import rclpy
from std_msgs.msg import Float32MultiArray

from xgym.nodes.base import Base

# from gello.agents.agent import BimanualAgent, DummyAgent
# from gello.data_utils.format_obs import save_frame
# from gello.robots.robot import PrintRobot


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class GelloArgs:
    agent: str = "gello"  # "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: tuple[float, ...] | None = None

    gello_port: str | None = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False
    safety: bool = True


class Gello(Base):
    def __init__(self, args: GelloArgs = GelloArgs(), env=None):
        super().__init__("gello")
        logger = self.get_logger()
        self.env = env

        self.port = args.gello_port
        if self.port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            logger.info(f"Found {len(usb_ports)} ports")
            if len(usb_ports) > 0:
                self.port = usb_ports[0]
                logger.info(f"using port {self.port}")
            else:
                msg = "No gello port found, please specify one or plug in gello"
                raise ValueError(msg)

        self.agent = GelloAgent(
            port=self.port,
            start_joints=args.start_joints,
            # dynamixel_config= DynamixelRobotConfig(),
        )

        logger.info("Gello Agent Created.")

        self.hz = 50
        self.loghz = 5
        self.pub = self.create_publisher(Float32MultiArray, "/gello/state", 10)
        self.timer = self.create_timer(1 / self.hz, self.run)
        self.i = 0

        self._home = {
            "angle": [
                -0.010571539402008057,
                -0.5,
                # 0.03834952041506767,
                -0.09203694015741348,
                1.5278464555740356,
                # 1.26,
                0.06442747265100479,
                1.428409457206726,
                -0.20091573894023895,
                # 3.42691308
                3.4,
                # 0.0 # open
            ],
            "pose": [
                502.73455810546875,
                -34.92372131347656,
                343.8944396972656,
                -3.094181776046753,
                -0.07189014554023743,
                0.08958626538515091,
            ],
        }

        # self.agent._robot.set_torque_mode(True)
        # self.agent._robot.command_joint_state(np.array(self._home["angle"]))
        # time.sleep(3)
        # self.agent._robot.set_torque_mode(False)
        # time.sleep(2)

        # self.set_active(Bool(data=False))
        logger.info("Gello Node Initialized.")
        self.set_period()

    def run(self):
        # raw = self._driver.get_joints()
        action = self.agent.act()

        logger = self.get_logger()
        # if self.i % (self.hz/self.loghz) == 0:
        # self.logp(f"Action: {action.round(2)}")

        msg = Float32MultiArray(data=action)
        self.pub.publish(msg)

        if self.env is not None:
            obs = self.env.step(action)

        self.i += 1


def build_env(args):
    # Change this to your own reset joints
    reset_joints = np.deg2rad([0, -90, 90, -90, -90, 0, 0]) if args.start_joints is None else args.start_joints

    camera_clients = {
        # you can optionally add camera nodes here for imitation learning purposes
        # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
        # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
    }

    # leader
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)

    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    print(env.get_obs())

    curr_joints = env.get_obs()["joint_positions"]
    if reset_joints.shape == curr_joints.shape:
        max_delta = (np.abs(curr_joints - reset_joints)).max()
        steps = min(int(max_delta / 0.01), 100)

        for jnt in np.linspace(curr_joints, reset_joints, steps):
            env.step(jnt)
            time.sleep(0.001)

    return env


def startup(agent, env, args):
    robo = agent._robot
    raw = robo._driver.get_joints()
    # pprint({"raw": raw})
    pos = (raw - robo._joint_offsets) * robo._joint_signs
    # pprint( pos[-1])
    # quit()

    # going to start position
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

        # print which joints are too big
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}")
            print(f"leader: {action}")
            print(f"follower: {joints}")
        if args.safety:
            exit()


@draccus.wrap() # TODO replace draccus with tyro
def main(args: GelloArgs):
    rclpy.init(args=None)

    # env = build_env(args)
    node = Gello(args, env=None)
    # startup(node.agent, env, args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Environment Node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main() # TODO replace draccus with tyro
