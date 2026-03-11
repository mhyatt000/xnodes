from __future__ import annotations

from pathlib import Path
import threading
import time
import unittest

import pytest

launch = pytest.importorskip("launch")
launch_testing = pytest.importorskip("launch_testing")
rclpy = pytest.importorskip("rclpy")

from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray
from xarm_msgs.msg import RobotMsg

OK = 0
SIGINT = -2
SIGTERM = -15
TEST_JOINT_TOPIC = "/test/joint_states"
TEST_POSE_TOPIC = "/test/xarm/robot_states"
TEST_GRIPPER_TOPIC = "/test/xgym/gripper"
TEST_STATE_TOPIC = "/test/gello/state"
TEST_LOW_CAMERA_TOPIC = "/test/xgym/camera/low"
TEST_SIDE_CAMERA_TOPIC = "/test/xgym/camera/side"
TEST_WRIST_CAMERA_TOPIC = "/test/xgym/camera/wrist"
TEST_PARAMS_FILE = Path("/tmp/xnodes_model_test_params.yaml")


@pytest.mark.launch_test
def generate_test_description():
    stub_path = Path(__file__).resolve().parent / "policy_stub_server.py"
    TEST_PARAMS_FILE.write_text(
        "\n".join(
            [
                "model:",
                "  ros__parameters:",
                f"    joint_topic: {TEST_JOINT_TOPIC}",
                f"    pose_topic: {TEST_POSE_TOPIC}",
                f"    gripper_topic: {TEST_GRIPPER_TOPIC}",
                f"    state_topic: {TEST_STATE_TOPIC}",
                f"    low_camera_topic: {TEST_LOW_CAMERA_TOPIC}",
                f"    side_camera_topic: {TEST_SIDE_CAMERA_TOPIC}",
                f"    wrist_camera_topic: {TEST_WRIST_CAMERA_TOPIC}",
            ]
        )
        + "\n"
    )
    policy_stub = launch.actions.ExecuteProcess(
        cmd=["python3", str(stub_path), "--host", "127.0.0.1", "--port", "8000"],
        name="policy_stub",
        output="screen",
    )
    launch_path = Path(__file__).resolve().parents[2] / "launch" / "model"
    include_model = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.AnyLaunchDescriptionSource(str(launch_path)),
        launch_arguments={
            "host": "127.0.0.1",
            "port": "8000",
            "rep": "REL",
            "task": "none",
            "ensemble": "False",
            "log_level": "info",
            "params_file": str(TEST_PARAMS_FILE),
        }.items(),
    )

    return (
        launch.LaunchDescription(
            [
                policy_stub,
                include_model,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {},
    )


class _ModelIoNode(Node):
    def __init__(self):
        super().__init__("test_model_io")
        self.pub_joint = self.create_publisher(JointState, TEST_JOINT_TOPIC, 10)
        self.pub_pose = self.create_publisher(RobotMsg, TEST_POSE_TOPIC, 10)
        self.pub_gripper = self.create_publisher(Float32MultiArray, TEST_GRIPPER_TOPIC, 10)
        self.pub_active = self.create_publisher(Bool, "/xgym/active", 10)

        self._lock = threading.Lock()
        self._gello_msgs: list[JointState] = []
        self.create_subscription(JointState, TEST_STATE_TOPIC, self._on_gello, 10)

    def _on_gello(self, msg: JointState) -> None:
        with self._lock:
            self._gello_msgs.append(msg)

    def publish_inputs(
        self,
        joint_positions: list[float] | None = None,
        joint_names: list[str] | None = None,
    ) -> None:
        joint = JointState()
        joint.position = [0.0] * 7 if joint_positions is None else joint_positions
        if joint_names is not None:
            joint.name = joint_names

        pose = RobotMsg()
        pose.pose = [1000.0, 0.0, 500.0, 0.0, 0.0, 0.0]

        grip = Float32MultiArray()
        grip.data = [0.5]

        self.pub_joint.publish(joint)
        self.pub_pose.publish(pose)
        self.pub_gripper.publish(grip)

    def has_gello(self) -> bool:
        with self._lock:
            return len(self._gello_msgs) > 0

    def clear_gello(self) -> None:
        with self._lock:
            self._gello_msgs.clear()

    def reset_model(self) -> None:
        self.clear_gello()
        for active in (False, True):
            self.pub_active.publish(Bool(data=active))
            deadline = time.time() + 0.2
            while time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)

    def first_gello(self) -> JointState:
        with self._lock:
            return self._gello_msgs[0]


class TestModelLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rclpy.init()
        cls.node = _ModelIoNode()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_model_ignores_six_dof_joint_state(self) -> None:
        self.node.reset_model()
        deadline = time.time() + 2.0
        while time.time() < deadline:
            self.node.publish_inputs(joint_positions=[0.0] * 6)
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertFalse(self.node.has_gello(), "6-DOF JointState should be ignored")

        deadline = time.time() + 6.0
        while time.time() < deadline:
            self.node.publish_inputs()
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.node.has_gello():
                break
        self.assertTrue(self.node.has_gello(), "Expected /gello/state after valid 7-DOF JointState")

    def test_model_publishes_gello_state(self) -> None:
        self.node.reset_model()
        deadline = time.time() + 6.0
        while time.time() < deadline:
            self.node.publish_inputs()
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.node.has_gello():
                break
        self.assertTrue(self.node.has_gello(), "Expected /gello/state within timeout")

        msg = self.node.first_gello()
        self.assertGreater(len(msg.position), 0)

    def test_model_accepts_joint_states_with_drive_joint(self) -> None:
        self.node.reset_model()
        deadline = time.time() + 6.0
        joint_names = [
            "drive_joint",
            "joint2",
            "joint3",
            "joint5",
            "joint6",
            "joint1",
            "joint4",
            "joint7",
        ]
        joint_positions = [0.5, -0.2, 0.1, 0.4, -0.3, 0.6, -0.1, 0.2]
        while time.time() < deadline:
            self.node.publish_inputs(
                joint_positions=joint_positions,
                joint_names=joint_names,
            )
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.node.has_gello():
                break
        self.assertTrue(self.node.has_gello(), "Expected /gello/state from /joint_states input")


@launch_testing.post_shutdown_test()
class TestModelExit(unittest.TestCase):
    def test_exit_codes(self, proc_info) -> None:
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[OK, SIGINT, SIGTERM])
