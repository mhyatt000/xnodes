from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import rclpy
from rich.pretty import pprint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
import tyro
from webpolicy.client import Client
from xarm_msgs.msg import RobotMsg

from xnodes.core.model_client import (
    ActionRepresentation,
    build_idle_target,
    build_policy_payload,
    extract_action_targets,
    extract_camera_images,
    missing_camera_images,
    ModelClientConfig,
    NOMODEL,
)

from .base import Base

np.set_printoptions(suppress=True)  # no scientific notation

__all__ = [  # bwd compatibility
    "NOMODEL",
    "ActionRepresentation",
    "Model",
    "ModelClientConfig",
    "MyClient",
    "main",
    "run",
]


class MyClient(Client):
    def reset(self):
        self.step({"reset": True})


class Model(Base):
    """Recieves action from model server"""

    def __init__(self, cfg: ModelClientConfig):
        super().__init__("model")
        self.cfg = cfg

        self.joints: np.ndarray | None = None
        self.pose: np.ndarray | None = None
        self.gripper: np.ndarray | None = None

        self.req_hz = 20  # request from server frequency
        self.cmd_hz = 100  # command frequency
        self.resolution = 1  # every 2 predicted steps

        self.data: dict[str, np.ndarray] = {}
        self.build_cam_subs()

        self.policy = MyClient(host=cfg.host, port=cfg.port)
        self.policy.reset()
        self._reset = True

        self.targets = None

        self.get_logger().info("Model Client Initialized.")

        self.moveit_sub = self.create_subscription(JointState, "/xarm/joint_states", self.set_joints, 10)
        self.moveit_pose_sub = self.create_subscription(RobotMsg, "/xarm/robot_states", self.set_pose, 10)
        self.gripper_sub = self.create_subscription(Float32MultiArray, "/xgym/gripper", self.set_gripper, 10)
        # self.timer = self.create_timer(1, self.step)
        self.publisher = self.create_publisher(Float32MultiArray, "/gello/state", 10)
        self.timer = self.create_timer(1 / self.cmd_hz, self.command)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._command_loop, daemon=True)
        self._thread.start()
        # self.stepper = self.create_timer(1 / self.req_hz, self.step)

        # self.ds = tfds.load("xgym_duck_single", split="train")
        # self.episode = self.ds.take(1)

    def set_active(self, msg):
        super().set_active(msg)
        time.sleep(0.25)
        self.reset()
        time.sleep(0.25)

        self.joints = None
        self.pose = None
        self.gripper = None
        self.targets = None

    def set_gripper(self, msg: Float32MultiArray):
        self.gripper = np.array(msg.data).astype(np.float32)

    def set_pose(self, msg: RobotMsg):
        """
        RobotMsg message has:
            header: std_msgs/Header header
            pose: List[float64]
            ...others...
        """
        self.pose = np.array(msg.pose).astype(np.float32)

    def set_joints(self, msg: JointState):
        """
        JointState message has:
            header: std_msgs/Header header
            name: List[string]
            position: List[float64]
            velocity: List[float64]
            effort: List[float64]
        """

        if len(msg.position) == 6:
            return
        self.joints = np.array(msg.position).astype(np.float32)
        # self.joint_names = msg.name

    def _command_loop(self):
        rate = 1.0 / self.req_hz
        while not self._stop_event.is_set():
            tic = time.time()
            self.step()
            toc = time.time()
            time.sleep(max(0, rate - (toc - tic)))

    def destroy_node(self):
        self._stop_event.set()
        self._thread.join()
        super().destroy_node()

    def step(self):
        if any(x is None for x in [self.joints, self.pose, self.gripper]):
            return
        if self.active is False:
            if not self._reset:
                self.policy.reset()
                self._reset = True
            self.targets = [build_idle_target(self.joints, self.gripper)]
            return

        imgs = extract_camera_images(self.data)
        missing = missing_camera_images(imgs)
        if missing:
            self.get_logger().info(f"Missing images: {missing}")
            return

        try:
            payload = build_policy_payload(
                joints=self.joints,
                pose=self.pose,
                gripper=self.gripper,
                images={k: v for k, v in imgs.items() if v is not None},
                ensemble=self.cfg.ensemble,
            )
        except ValueError as exc:
            self.get_logger().info(f"Payload error: {exc}")
            return

        actions: dict[str, Any] = self.policy.step(payload)
        try:
            self.targets = extract_action_targets(actions, resolution=self.resolution)
        except (KeyError, TypeError, ValueError) as exc:
            self.get_logger().info(f"Action parse error: {exc}")

    def reset(self):
        self._reset = False  # send reset signal to thread

    def command(self):
        """Publishes model action"""

        if self.targets is None or len(self.targets) == 0:
            if self.joints is None or self.gripper is None:
                # print("No joints or gripper")
                return
            target = build_idle_target(self.joints, self.gripper)
            # self.step()
            # return
        else:
            target, self.targets = self.targets[0], self.targets[1:]
            # print(len(self.targets))

        msg = Float32MultiArray()
        msg.data = target.tolist()
        self.publisher.publish(msg)


def main(cfg: ModelClientConfig):
    pprint(cfg)

    args = None
    rclpy.init(args=args)

    node = Model(cfg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Controller Node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run(cfg: ModelClientConfig | None = None):
    if cfg is None:
        cfg = tyro.cli(ModelClientConfig)
    main(cfg)


if __name__ == "__main__":
    run()
