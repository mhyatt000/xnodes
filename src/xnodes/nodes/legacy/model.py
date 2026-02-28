from __future__ import annotations

import threading
from typing import Any

import numpy as np
import rclpy
from rich import print
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
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
from xnodes.nodes.legacy.base import Base

np.set_printoptions(suppress=True)  # no scientific notation

__all__ = [  # bwd compatibility
    "NOMODEL",
    "ActionRepresentation",
    "Model",
    "ModelClientConfig",
    "main",
    "run",
]


class Model(Base):
    """Recieves action from model server"""

    def __init__(self, cfg: ModelClientConfig | None = None):
        super().__init__("model")
        self._lock = threading.Lock()

        self.joints: np.ndarray | None = None
        self.pose: np.ndarray | None = None
        self.gripper: np.ndarray | None = None

        default_cfg = NOMODEL if cfg is None else cfg
        host = str(self.declare_parameter("host", default_cfg.host).value)
        port = int(self.declare_parameter("port", int(default_cfg.port)).value)
        rep_raw = str(self.declare_parameter("rep", default_cfg.rep.name).value).strip().upper()
        task = str(self.declare_parameter("task", default_cfg.task).value)
        ensemble = bool(self.declare_parameter("ensemble", bool(default_cfg.ensemble)).value)

        try:
            rep = ActionRepresentation[rep_raw]
        except KeyError as exc:
            msg = f"Invalid rep={rep_raw}. Expected one of {[x.name for x in ActionRepresentation]}"
            raise ValueError(msg) from exc

        self.cfg = ModelClientConfig(host=host, port=port, rep=rep, task=task, ensemble=ensemble)

        self.req_hz = float(self.declare_parameter("req_hz", 20.0).value)  # request from server frequency
        self.cmd_hz = float(self.declare_parameter("cmd_hz", 100.0).value)  # command frequency
        self.resolution = int(self.declare_parameter("resolution", 1).value)  # every N predicted steps
        self.joint_topic = str(self.declare_parameter("joint_topic", "/xarm/joint_states").value)
        self.pose_topic = str(self.declare_parameter("pose_topic", "/xarm/robot_states").value)
        self.gripper_topic = str(self.declare_parameter("gripper_topic", "/xgym/gripper").value)
        self.state_topic = str(self.declare_parameter("state_topic", "/gello/state").value)

        if self.req_hz <= 0 or self.cmd_hz <= 0:
            raise ValueError("req_hz and cmd_hz must be > 0")
        if self.resolution <= 0:
            raise ValueError("resolution must be > 0")

        self.data: dict[str, np.ndarray] = {}
        self.build_cam_subs()

        self.policy = Client(host=self.cfg.host, port=self.cfg.port)
        # self.policy.reset()
        self._reset = True

        self.targets = None

        self.logger.info("Model Client Initialized.")

        self.moveit_sub = self.create_subscription(JointState, self.joint_topic, self.set_joints, 10)
        self.moveit_pose_sub = self.create_subscription(RobotMsg, self.pose_topic, self.set_pose, 10)
        self.gripper_sub = self.create_subscription(Float32MultiArray, self.gripper_topic, self.set_gripper, 10)
        self.step_timer = self.create_timer(1 / self.req_hz, self.step)
        self.publisher = self.create_publisher(Float32MultiArray, self.state_topic, 10)
        self.command_timer = self.create_timer(1 / self.cmd_hz, self.command)

    @property
    def logger(self):
        return self.get_logger()

    def set_active(self, msg):
        super().set_active(msg)
        with self._lock:
            self._reset = False  # send reset signal to command thread
            self.joints = None
            self.pose = None
            self.gripper = None
            self.targets = None

    def set_image(self, msg, key):
        frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        with self._lock:
            self.data[key] = frame

    def set_gripper(self, msg: Float32MultiArray):
        with self._lock:
            self.gripper = np.array(msg.data).astype(np.float32)

    def set_pose(self, msg: RobotMsg):
        """
        RobotMsg message has:
            header: std_msgs/Header header
            pose: List[float64]
            ...others...
        """
        with self._lock:
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
            self.logger.error("Received JointState with 6 joints; shutting down node.")
            rclpy.shutdown()
            return
        with self._lock:
            self.joints = np.array(msg.position).astype(np.float32)
        # self.joint_names = msg.name

    def step(self):
        with self._lock:
            joints = None if self.joints is None else self.joints.copy()
            pose = None if self.pose is None else self.pose.copy()
            gripper = None if self.gripper is None else self.gripper.copy()
            active = self.active
            reset = self._reset
            imgs = extract_camera_images(self.data)

        if any(x is None for x in [joints, pose, gripper]):
            return

        if active is False:
            if not reset:
                self.policy.reset()
                with self._lock:
                    self._reset = True
            with self._lock:
                self.targets = [build_idle_target(joints, gripper)]
            return

        missing = missing_camera_images(imgs)
        if missing:
            self.logger.info(f"Missing images: {missing}")
            return

        try:
            payload = build_policy_payload(
                joints=joints,
                pose=pose,
                gripper=gripper,
                images={k: v for k, v in imgs.items() if v is not None},
                ensemble=self.cfg.ensemble,
            )
        except ValueError as exc:
            self.logger.info(f"Payload error: {exc}")
            return

        actions: dict[str, Any] = self.policy.step(payload)
        try:
            targets = extract_action_targets(actions, resolution=self.resolution)
            with self._lock:
                self.targets = targets
        except (KeyError, TypeError, ValueError) as exc:
            self.logger.info(f"Action parse error: {exc}")

    def reset(self):
        with self._lock:
            self._reset = False  # send reset signal to thread

    def command(self):
        """Publishes model action"""
        with self._lock:
            targets = self.targets
            joints = self.joints
            gripper = self.gripper
            if targets is not None and len(targets) > 0:
                target, self.targets = targets[0], targets[1:]
            else:
                if joints is None or gripper is None:
                    return
                target = build_idle_target(joints, gripper)

        msg = Float32MultiArray()
        msg.data = target.tolist()
        self.publisher.publish(msg)


def main(cfg: ModelClientConfig | None = None):
    rclpy.init(args=None)
    node = Model(cfg)
    print(node.cfg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.logger.info("Controller Node shutting down...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def run(cfg: ModelClientConfig | None = None):
    main(cfg)


if __name__ == "__main__":
    run()
