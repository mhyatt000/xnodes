from __future__ import annotations

from dataclasses import dataclass
import time

import launch_ros
import launch_testing
import launch_testing.actions
import pytest
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import launch


# --------------------------
# Launch description
# --------------------------
@pytest.mark.launch_test
def generate_test_description():
    sut = launch_ros.actions.Node(
        package="my_pkg",
        executable="my_node",  # must match your entrypoint/executable name
        name="my_node",
        output="screen",
        parameters=[{"some_param": 123}],
    )

    # ReadyToTest makes sure launch_testing doesn't start tests until launch is stable
    return (
        launch.LaunchDescription(
            [
                sut,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {
            "sut": sut,  # pass the Node action object into tests
        },
    )


# --------------------------
# Helpers
# --------------------------
@dataclass
class MsgWaiter:
    msg: object | None = None


def wait_for_msg(node: Node, topic: str, msg_type, timeout_s: float):
    waiter = MsgWaiter()

    def cb(m):
        waiter.msg = m

    sub = node.create_subscription(msg_type, topic, cb, 10)

    t0 = time.time()
    while rclpy.ok() and waiter.msg is None and (time.time() - t0) < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_subscription(sub)
    return waiter.msg


# --------------------------
# Tests while system is running
# --------------------------
class TestWhileRunning:
    def test_process_starts(self, proc_info, sut):
        # Most common: ensure node actually spawned
        proc_info.assertWaitForStartup(process=sut, timeout=10)

    def test_emits_ready_log(self, proc_output):
        # If your node prints/logs something like "Ready"
        proc_output.assertWaitFor("Ready", timeout=10)

    def test_publishes_status_topic(self):
        # Example: verify node publishes /status once within 5s
        rclpy.init()
        node = rclpy.create_node("test_listener")

        msg = wait_for_msg(node, "/status", String, timeout_s=5.0)
        assert msg is not None, "Did not receive /status"
        # Example content check
        # assert msg.data == "ok"

        node.destroy_node()
        rclpy.shutdown()


# --------------------------
# Tests after shutdown (exit code, clean shutdown, etc.)
# --------------------------
@launch_testing.post_shutdown_test()
class TestAfterShutdown:
    def test_clean_shutdown(self, proc_info, sut):
        proc_info.assertWaitForShutdown(process=sut, timeout=10)

    def test_exit_code_zero(self, proc_info, sut):
        # Verifies it didn't crash (non-zero exit)
        proc_info.assertExitCodes(process=sut)
