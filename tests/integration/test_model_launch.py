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
from std_msgs.msg import Float32MultiArray
from xarm_msgs.msg import RobotMsg

OK = 0
SIGINT = -2
SIGTERM = -15


@pytest.mark.launch_test
def generate_test_description():
    stub_path = Path(__file__).resolve().parent / "policy_stub_server.py"
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
        self.pub_joint = self.create_publisher(JointState, "/xarm/joint_states", 10)
        self.pub_pose = self.create_publisher(RobotMsg, "/xarm/robot_states", 10)
        self.pub_gripper = self.create_publisher(Float32MultiArray, "/xgym/gripper", 10)

        self._lock = threading.Lock()
        self._gello_msgs: list[Float32MultiArray] = []
        self.create_subscription(Float32MultiArray, "/gello/state", self._on_gello, 10)

    def _on_gello(self, msg: Float32MultiArray) -> None:
        with self._lock:
            self._gello_msgs.append(msg)

    def publish_inputs(self) -> None:
        joint = JointState()
        joint.position = [0.0] * 7

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

    def first_gello(self) -> Float32MultiArray:
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

    def test_model_publishes_gello_state(self) -> None:
        deadline = time.time() + 6.0
        while time.time() < deadline:
            self.node.publish_inputs()
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.node.has_gello():
                break
        self.assertTrue(self.node.has_gello(), "Expected /gello/state within timeout")

        msg = self.node.first_gello()
        self.assertGreater(len(msg.data), 0)


@launch_testing.post_shutdown_test()
class TestModelExit(unittest.TestCase):
    def test_exit_codes(self, proc_info) -> None:
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[OK, SIGINT, SIGTERM])
