from __future__ import annotations

from builtin_interfaces.msg import Duration
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tyro
from xarm_msgs.msg import RobotMsg

from xnodes.components.robot import ControlMode, HOME, RobotConfig, RobotPolicy
from xnodes.nodes.legacy.active import ActiveFlag
from xnodes.nodes.legacy.gripper import GripperController, RobotDriverPlugin


class Xarm(Node):
    """Thin xArm ROS node: wires subscriptions, publishers, and timers to RobotPolicy."""

    def __init__(
        self,
        cfg: RobotConfig,
        driver_plugin: RobotDriverPlugin | None = None,
        # activator: ActiveFlag | None = None,
    ):
        super().__init__("xarm_robot")
        self.cfg = cfg
        self.hz = cfg.hz

        self.policy = RobotPolicy(cfg, HOME)
        # self.activator = activator if activator is not None else ActiveFlag(self)
        self.activators = {
            "active": ActiveFlag(self, toggle=True),
            "estop": ActiveFlag(self, toggle=True, topic="/xgym/estop"),
        }
        self.activator = self.activators["active"]
        self.activator.add_hook(self.policy.on_active)
        self.gripper = GripperController(self, cfg, self.policy, self.activator, driver_plugin=driver_plugin)
        self.get_logger().info(f"Initialized robot backend={cfg.backend}.")

        # Subscriptions
        self.create_subscription(Float32MultiArray, "/robot_commands", self._on_command, 10)
        self.create_subscription(JointState, "/gello/state", self._on_leader, 10)
        # Keep both joint-state topics during the legacy migration.
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        self.create_subscription(JointState, "/xarm/joint_states", self._on_joints, 10)
        self.create_subscription(RobotMsg, "/xarm/robot_states", self._on_pose, 10)

        # Publishers
        self.twist_pub = self.create_publisher(TwistStamped, "/servo_server/delta_twist_cmds", 10)
        self.jog_pub = self.create_publisher(JointJog, "/servo_server/delta_joint_cmds", 10)

        self.timer = self.create_timer(1 / cfg.hz, self._tick)
        self.timer.cancel()
        self.activators["estop"].add_hook(lambda active: self.timer.cancel() if not active else self.timer.reset())
        self.activators["estop"].add_hook(self.gripper._on_active)

        # publisher
        self.traj_pub = self.create_publisher(JointTrajectory, "/xarm7_traj_controller/joint_trajectory", 10)

        self.get_logger().info("Robot Node Initialized.")

    @property
    def logger(self):
        return self.get_logger()

    # --- Subscription callbacks ---

    def _on_joints(self, msg: JointState) -> None:
        # self.logger.info(f"on joints: {np.array(msg.position).round(2)} {np.array(msg.position).shape}")
        # self.logger.info(f"on joints: {list(msg.name)} {len(list(msg.name))}")
        self.policy.update_joints(np.array(msg.position), list(msg.name))

    def _on_pose(self, msg: RobotMsg) -> None:
        self.policy.update_pose(np.array(msg.pose, dtype=np.float32))

    def _on_command(self, msg: Float32MultiArray) -> None:
        self.policy.update_command(np.array(msg.data))
        self.logger.info(f"{self.policy._act}")

    def _on_leader(self, msg: JointState) -> None:
        self.policy.update_leader(np.array(msg.position))
        self.logger.debug(f"on leader: {np.array(msg.position).round(2)} {np.array(msg.position).shape}")

    # --- Timer callbacks ---

    def _tick(self) -> None:
        """Main control loop: delegate to policy and publish result."""

        self.policy.tick()

        match self.cfg.ctrl:
            case ControlMode.JOINT:
                result = self.policy.step_joints()
                if result is None:
                    # self.logger.info(f"{result}")
                    return

                if result is False:
                    joint_names = self.policy._joint_names
                    if not joint_names:
                        return
                    displacements = [0.0] * len(joint_names)
                    velocities = [0.0] * len(joint_names)
                else:
                    displacements, joint_names, velocities = result
                    # Keep a zero-motion heartbeat flowing once the streams are synchronized.
                    displacements = np.array(self.policy._leader[:-1] - self.policy._joints).round(2)
                    # Per-joint clip: wrist joints (5-7) need more travel per tick than shoulder/elbow.
                    t1, t2, t3 = 0.08, 0.1, 0.15  # teleop
                    #t1, t2, t3 = 0.04, 0.05, 0.15  # mid
                    #t1, t2, t3 = 0.01, 0.01, 0.10  # slow/safe
                    eps = np.array([t1, t1, t1, t2, t2, t3, t3])
                    displacements = np.clip(displacements, -eps, eps).tolist()
                    # self.logger.info("after")
                    # self.logger.info(f"displ: {(self.policy._leader[:-1]-self.policy._joints).round(2)}")
                    # self.logger.info(f"leader{(self.policy._leader).round(2)}")
                    # self.logger.info(f"{self.policy._joint_names}")
                    # self.logger.info("")
                    # self.logger.info(f"{(self.policy._joints).round(2)}")

                # in _tick, replace JointJog publish:
                target = (self.policy._joints + np.array(displacements)).tolist()
                msg = JointTrajectory()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "link_base"
                msg.joint_names = joint_names
                pt = JointTrajectoryPoint()
                pt.positions = target
                pt.time_from_start = Duration(sec=0, nanosec=20_000_000)  # 20ms
                msg.points = [pt]
                self.traj_pub.publish(msg)

                msg = JointJog()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "link_base"
                # msg.displacements = displacements
                # servo_server can use velocities without displacement
                msg.joint_names = joint_names
                msg.velocities = velocities
                # self.jog_pub.publish(msg)

            case ControlMode.CARTESIAN:
                result = self.policy.step_cartesian()
                self.logger.info(f"{result}")
                if result is None:
                    return
                result = result.tolist()
                self.logger.info(f"{type(result)}")
                self.logger.info(f"{type(result[0])}")

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


def run(cfg: RobotConfig | None = None) -> None:
    rclpy.init(args=None)
    node = Xarm(cfg=cfg or RobotConfig())
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Robot Node shutting down...")
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(cfg: RobotConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main(tyro.cli(RobotConfig))
