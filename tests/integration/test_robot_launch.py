from __future__ import annotations

import threading
import time
import unittest

import pytest

launch = pytest.importorskip("launch")
launch_testing = pytest.importorskip("launch_testing")
rclpy = pytest.importorskip("rclpy")

from control_msgs.msg import JointJog
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from xarm_msgs.msg import RobotMsg

OK = 0
SIGINT = -2
SIGTERM = -15

JOINT_NAMES = [f"joint{i + 1}" for i in range(7)]


@pytest.mark.launch_test
def generate_test_description():
    xarm_node = launch.actions.ExecuteProcess(
        cmd=[
            "python3",
            "-c",
            "from xnodes.components.robot import RobotConfig; "
            "from xnodes.nodes.legacy.robot import run; "
            "run(RobotConfig(ip=None))",
        ],
        name="xarm_node",
        output="screen",
    )
    return (
        launch.LaunchDescription([xarm_node, launch_testing.actions.ReadyToTest()]),
        {},
    )


class _RobotIoNode(Node):
    def __init__(self) -> None:
        super().__init__("test_robot_io")
        self.pub_joint = self.create_publisher(JointState, "/xarm/joint_states", 10)
        self.pub_pose = self.create_publisher(RobotMsg, "/xarm/robot_states", 10)
        self.pub_gello = self.create_publisher(JointState, "/gello/state", 10)
        self.pub_active = self.create_publisher(Bool, "/xgym/active", 10)

        self._lock = threading.Lock()
        self._jog_msgs: list[JointJog] = []
        self.create_subscription(JointJog, "/servo_server/delta_joint_cmds", self._on_jog, 10)

    def _on_jog(self, msg: JointJog) -> None:
        with self._lock:
            self._jog_msgs.append(msg)

    def publish_inputs(self) -> None:
        joint = JointState()
        joint.name = JOINT_NAMES
        joint.position = [0.0] * 7

        pose = RobotMsg()
        pose.pose = [502.73, -34.92, 343.89, -3.09, -0.07, 0.09]

        gello = JointState()
        gello.name = [f"joint{i + 1}" for i in range(8)]
        gello.position = [0.0] * 8

        active = Bool()
        active.data = True

        self.pub_active.publish(active)
        self.pub_joint.publish(joint)
        self.pub_pose.publish(pose)
        self.pub_gello.publish(gello)

    def has_jog(self) -> bool:
        with self._lock:
            return len(self._jog_msgs) > 0

    def first_jog(self) -> JointJog:
        with self._lock:
            return self._jog_msgs[0]


class TestRobotLaunch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rclpy.init()
        cls.node = _RobotIoNode()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_joint_jog_published(self) -> None:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            self.node.publish_inputs()
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.node.has_jog():
                break

        self.assertTrue(self.node.has_jog(), "Expected /servo_server/delta_joint_cmds within 10s")
        msg = self.node.first_jog()
        self.assertEqual(len(msg.joint_names), 7)
        self.assertEqual(len(msg.displacements), 7)


@launch_testing.post_shutdown_test()
class TestRobotExit(unittest.TestCase):
    def test_exit_codes(self, proc_info) -> None:
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[OK, SIGINT, SIGTERM])
