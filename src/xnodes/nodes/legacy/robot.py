from __future__ import annotations

import time

from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray
import tyro
from xarm.wrapper import XArmAPI
from xarm_msgs.msg import RobotMsg

from .base import Base
from .robot_policy import ControlMode, HOME, RobotConfig, RobotPolicy


class Xarm(Base):
    """Thin xArm ROS node: wires subscriptions, publishers, and timers to RobotPolicy."""

    def __init__(self, cfg: RobotConfig):
        super().__init__("xarm_robot")
        self.cfg = cfg
        self.hz = cfg.hz
        self.griphz = cfg.grip_hz
        self.t0 = time.time()
        self.set_period()

        self.robot = XArmAPI(cfg.ip, is_radian=True)
        self.get_logger().info("Initializing robot.")
        self.robot.connect()
        self.mode = 1

        if cfg.use_gripper:
            self.robot.set_gripper_enable(True)
            self.robot.set_gripper_mode(0)
            self.robot.set_gripper_speed(5000)
        self.get_logger().info("Robot initialized.")

        self.policy = RobotPolicy(cfg, HOME)

        # Subscriptions
        self.create_subscription(Float32MultiArray, "/robot_commands", self._on_command, 10)
        self.create_subscription(Float32MultiArray, "/gello/state", self._on_leader, 10)
        self.create_subscription(JointState, "/xarm/joint_states", self._on_joints, 10)
        self.create_subscription(RobotMsg, "/xarm/robot_states", self._on_pose, 10)

        # Publishers
        self.twist_pub = self.create_publisher(TwistStamped, "/servo_server/delta_twist_cmds", 10)
        self.jog_pub = self.create_publisher(JointJog, "/servo_server/delta_joint_cmds", 10)

        self.timer = self.create_timer(1 / cfg.hz, self._tick)

        if cfg.use_gripper:
            self.gripper_pub = self.create_publisher(Float32MultiArray, "/xgym/gripper", 10)
            self.gtimer = self.create_timer(1 / cfg.grip_hz, self._grip_tick)

        self.get_logger().info("Robot Node Initialized.")

    def set_active(self, msg: Bool) -> None:
        super().set_active(msg)
        self.policy.on_active(msg.data)
        self._clear_error_states()

    # --- Subscription callbacks ---

    def _on_joints(self, msg: JointState) -> None:
        self.policy.update_joints(np.array(msg.position), list(msg.name))

    def _on_pose(self, msg: RobotMsg) -> None:
        self.policy.update_pose(np.array(msg.pose, dtype=np.float32))

    def _on_command(self, msg: Float32MultiArray) -> None:
        self.policy.update_command(np.array(msg.data))

    def _on_leader(self, msg: Float32MultiArray) -> None:
        self.policy.update_leader(np.array(msg.data))

    # --- Timer callbacks ---

    def _tick(self) -> None:
        """Main control loop: delegate to policy and publish result."""
        self.policy.tick()

        match self.cfg.ctrl:
            case ControlMode.JOINT:
                result = self.policy.step_joints()
                if result is None:
                    return
                displacements, joint_names, velocities = result
                msg = JointJog()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "link_base"
                msg.displacements = displacements
                msg.joint_names = joint_names
                msg.velocities = velocities
                self.jog_pub.publish(msg)

            case ControlMode.CARTESIAN:
                result = self.policy.step_cartesian()
                if result is None:
                    return
                msg = TwistStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "link_base"
                msg.twist.linear.x = result[0]
                msg.twist.linear.y = result[1]
                msg.twist.linear.z = result[2]
                msg.twist.angular.x = result[3]
                msg.twist.angular.y = result[4]
                msg.twist.angular.z = result[5]
                self.twist_pub.publish(msg)

    def _grip_tick(self) -> None:
        """Gripper control loop: read hardware position and apply policy command."""
        code, grip_raw = self.robot.get_gripper_position()
        if code or grip_raw is None:
            return
        self.gripper_pub.publish(Float32MultiArray(data=[grip_raw / self.cfg.grip_max]))
        cmd = self.policy.step_gripper(grip_raw)
        if cmd is not None:
            self.robot.set_gripper_position(cmd, wait=False)

    def _clear_error_states(self) -> None:
        if self.robot is None:
            return
        time.sleep(0.1)
        print(self.robot.set_state(state=0))
        print(self.robot.set_mode(1))
        time.sleep(0.1)
        self.robot.clean_error()
        self.robot.clean_warn()
        self.robot.motion_enable(True)
        time.sleep(0.1)
        self.robot.set_mode(1)
        time.sleep(0.1)
        self.robot.set_state(state=0)
        time.sleep(0.1)
        if self.cfg.use_gripper:
            self.robot.set_gripper_enable(True)
            time.sleep(0.1)
            self.robot.set_gripper_mode(0)
            time.sleep(0.1)
            self.robot.set_gripper_speed(5000)
            time.sleep(0.1)


def run(cfg: RobotConfig | None = None) -> None:
    rclpy.init(args=None)
    node = Xarm(cfg=cfg or RobotConfig())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Robot Node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(cfg: RobotConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main(tyro.cli(RobotConfig))
