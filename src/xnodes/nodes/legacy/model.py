from __future__ import annotations

from functools import partial
import threading
import typing
from typing import Any

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import jax
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from rich import print
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray
from tf2_ros import Buffer, TransformException, TransformListener
from webpolicy.client import Client
from xarm_msgs.msg import RobotMsg

from xnodes.core.model_client import (
    ActionRepresentation,
    build_idle_target,
    build_policy_payload,
    CAMERA_KEYS,
    CAMERA_POSTFIX,
    CAMERA_PREFIX,
    extract_action_targets,
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


PyTree: typing.TypeAlias = dict[str, np.ndarray] | list[np.ndarray]


def spec(x: PyTree):
    return jax.tree.map(lambda y: y.shape if isinstance(y, np.ndarray) else type(y), x)


class TfLookup:
    def __init__(self, node: rclpy.node.Node):
        self._node = node
        self._buf = Buffer()
        self._listener = TransformListener(self._buf, node)

    def lookup(self, target: str, source: str, timeout_s: float = 0.1) -> TransformStamped | None:
        try:
            return self._buf.lookup_transform(
                target_frame=target,
                source_frame=source,
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_s),
            )
        except TransformException as err:
            self._node.get_logger().warn(f"TF lookup failed ({source} -> {target}): {err}")
            return None

    def xyz(self, target: str, source: str, timeout_s: float = 0.1) -> tuple[float, float, float] | None:
        tf = self.lookup(target, source, timeout_s)
        if tf is None:
            return None
        t = tf.transform.translation
        return t.x, t.y, t.z


class CameraStreams:
    def __init__(self, node: rclpy.node.Node, topics: dict[str, str]):
        self._node = node
        self._topics = topics
        self._lock = threading.Lock()
        self._bridge = CvBridge()
        self._frames: dict[str, np.ndarray | None] = dict.fromkeys(topics)
        self._seen: set[str] = set()
        self._subs = {
            key: node.create_subscription(
                Image,
                topic,
                partial(self._set_image, key=key),
                qos_profile_sensor_data,
            )
            for key, topic in topics.items()
        }
        node.get_logger().info(f"Camera Streams: {self._topics}")

    @property
    def topics(self) -> dict[str, str]:
        return self._topics

    def snapshot(self) -> dict[str, np.ndarray | None]:
        with self._lock:
            return {key: None if frame is None else frame.copy() for key, frame in self._frames.items()}

    def _set_image(self, msg: Image, key: str) -> None:
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self._frames[key] = frame
        if key not in self._seen:
            self._node.get_logger().info(f"Accepted image stream {key} from {self._topics[key]}")
            self._seen.add(key)


class Model(Base):
    """Recieves action from model server"""

    def __init__(self, cfg: ModelClientConfig | None = None, cameras: CameraStreams | None = None):
        super().__init__("model")
        self._lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        self._step_group = MutuallyExclusiveCallbackGroup()
        self._command_group = MutuallyExclusiveCallbackGroup()
        # A reentrant group could make sense later if policy requests become
        # concurrency-safe and overlapping step() calls are explicitly desired.

        self.joints: np.ndarray | None = None
        self.pose: np.ndarray | None = None
        self.gripper: np.ndarray | None = None
        self._last_joint_len: int | None = None
        self._last_joint_warning: str | None = None
        self._accepted_joint_stream = False

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

        self.req_hz = float(self.declare_parameter("req_hz", 5.0).value)  # request from server frequency
        self.cmd_hz = float(self.declare_parameter("cmd_hz", 10.0).value)  # command frequency
        self.resolution = int(self.declare_parameter("resolution", 1).value)  # every N predicted steps
        self.arm_dof = int(self.declare_parameter("arm_dof", 7).value)
        self.joint_topic = str(self.declare_parameter("joint_topic", "/joint_states").value)
        self.pose_topic = str(self.declare_parameter("pose_topic", "/xarm/robot_states").value)
        self.gripper_topic = str(self.declare_parameter("gripper_topic", "/xgym/gripper").value)
        self.state_topic = str(self.declare_parameter("state_topic", "/gello/state").value)

        self.camera_topics = {
            key: str(
                self.declare_parameter(
                    f"{key}_camera_topic",
                    f"{CAMERA_PREFIX}/{key}/{CAMERA_POSTFIX}",
                ).value
            )
            for key in CAMERA_KEYS
        }
        self.camera_topics |= {
            "wrist": str(
                self.declare_parameter(
                    "wrist_camera_topic",
                    "/camera/camera/color/image_raw",
                ).value
            )
        }

        if self.req_hz <= 0 or self.cmd_hz <= 0:
            raise ValueError("req_hz and cmd_hz must be > 0")
        if self.resolution <= 0:
            raise ValueError("resolution must be > 0")
        if self.arm_dof <= 0:
            raise ValueError("arm_dof must be > 0")

        self.get_logger().info(f"client->server at {self.cfg.host}:{self.cfg.port} ...")
        self.policy = Client(host=self.cfg.host, port=self.cfg.port)
        # self.policy.reset()
        self._reset = True

        self.targets = None
        self.cameras = cameras if cameras is not None else CameraStreams(self, self.camera_topics)

        self.logger.info("Model Client Initialized.")
        self.logger.info(
            f"Subscribing: joints={self.joint_topic}, pose={self.pose_topic}, gripper={self.gripper_topic}, cameras={self.cameras.topics}"
        )
        self.tf = TfLookup(self)

        self.moveit_sub = self.create_subscription(JointState, self.joint_topic, self.set_joints, 10)
        # self.moveit_pose_sub = self.create_subscription(RobotMsg, self.pose_topic, self.set_pose, 10)
        self.moveit_pose_t = self.create_timer(0.1, self.request_pose)  # polling tf for pose
        self.gripper_sub = self.create_subscription(Float32MultiArray, self.gripper_topic, self.set_gripper, 10)
        self.step_timer = self.create_timer(1 / self.req_hz, self.step, callback_group=self._step_group)
        self.publisher = self.create_publisher(JointState, self.state_topic, 10)
        self.command_timer = self.create_timer(1 / self.cmd_hz, self.command, callback_group=self._command_group)

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
        with self._cmd_lock:
            self.targets = None

    def request_pose(self):
        pose = self.tf.xyz("world", "link_tcp")
        if pose is not None:
            msg = RobotMsg()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose = list(pose) + [0.0] * 3  # pad to 6dof
            self.set_pose(msg)

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
        joints = np.asarray(msg.position, dtype=np.float32)
        names = list(msg.name)

        if len(joints) != self.arm_dof and names:
            joint_map = dict(zip(names, joints, strict=False))
            arm_names = [f"joint{i + 1}" for i in range(self.arm_dof)]
            if all(name in joint_map for name in arm_names):
                joints = np.asarray([joint_map[name] for name in arm_names], dtype=np.float32)

        n_joints = len(joints)
        if n_joints != self.arm_dof:
            warning = (
                f"Ignoring JointState on {self.joint_topic}: expected {self.arm_dof} arm joints, "
                f"got {n_joints} names={names}."
            )
            if self._last_joint_warning != warning:
                self.logger.warn(warning)
                self._last_joint_warning = warning
                self._last_joint_len = n_joints
            return

        with self._lock:
            self.joints = joints
            self._last_joint_len = n_joints
            self._last_joint_warning = None
        if not self._accepted_joint_stream:
            self.logger.info(f"Accepted JointState from {self.joint_topic} with {n_joints} arm joints.")
            self._accepted_joint_stream = True
        # self.joint_names = msg.name

    def step(self):
        with self._lock:
            joints = None if self.joints is None else self.joints.copy()
            pose = None if self.pose is None else self.pose.copy()
            gripper = None if self.gripper is None else self.gripper.copy()
            active = self.active
            reset = self._reset
            self.logger.info("enter step")

        imgs = self.cameras.snapshot()
        self.logger.info(f"snapshot got: {spec(imgs)}")

        if any(x is None for x in [joints, pose, gripper]):
            self.logger.info(
                f"Waiting for data... pose={pose is not None}, joints={joints is not None}, gripper={gripper is not None}"
            )
            return

        """
        if active is False:
            if not reset:
                self.policy.reset()
                with self._lock:
                    self._reset = True
            with self._lock:
                self.targets = [build_idle_target(joints, gripper)]
            return
        """

        missing = missing_camera_images(imgs)
        if missing:
            missing_topics = {key: self.cameras.topics[key] for key in missing}
            self.logger.info(f"Missing images: {missing_topics}")
            return
        imgs = jax.tree.map(lambda img: cv2.resize(img, (64, 64)) if img is not None else None, imgs)  # resize 64,64
        self.logger.info(f"snapshot got: {spec(imgs)}")

        try:
            payload = build_policy_payload(
                joints=joints,
                pose=pose,
                gripper=gripper,
                images={k: v for k, v in imgs.items() if v is not None},
                ensemble=self.cfg.ensemble,
            )
            payload["observation"]["proprio_single"] = payload["observation"].pop("proprio_single_arm")[
                ..., -(self.arm_dof + 1) :
            ]  # trim to expected arm dof
        except ValueError as exc:
            self.logger.info(f"Payload error: {exc}")
            return

        self.logger.info(f"Policy payload: joints={spec(payload)}")
        actions: dict[str, Any] = self.policy.step(payload)
        try:
            targets = extract_action_targets(actions, resolution=self.resolution)
        except (KeyError, TypeError, ValueError) as exc:
            self.logger.info(f"Action parse error: {exc}")
            return
        with self._cmd_lock:
            self.targets = targets

    def reset(self):
        with self._lock:
            self._reset = False  # send reset signal to thread

    def command(self):
        """Publishes model action"""
        with self._cmd_lock:
            targets = self.targets
            if targets is not None and len(targets) > 0:
                target, self.targets = targets[0], targets[1:]
            else:
                target = None

        if target is None:
            with self._lock:
                joints = None if self.joints is None else self.joints.copy()
                gripper = None if self.gripper is None else self.gripper.copy()
            if joints is None or gripper is None:
                return
            target = build_idle_target(joints, gripper)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint{i + 1}" for i in range(len(target))]
        msg.position = target.tolist()
        self.publisher.publish(msg)


def main(cfg: ModelClientConfig | None = None):
    rclpy.init(args=None)
    node = Model(cfg)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    print(node.cfg)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.logger.info("Controller Node shutting down...")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def run(cfg: ModelClientConfig | None = None):
    main(cfg)


if __name__ == "__main__":
    run()
