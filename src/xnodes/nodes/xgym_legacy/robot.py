from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time

from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from xarm.wrapper import XArmAPI
from xarm_msgs.msg import RobotMsg

from .base import Base


class Accelerator:
    def __init__(
        self,
        dt=0.02,
        Kp=1.0,
        Kd=0.5,
        A_max=2.0,
        initial_position=0.0,
        initial_velocity=0.0,
    ):
        """
        Initializes the Accelerator for multi-dimensional input.

        :param dt: Time step in seconds.
        :param Kp: Proportional gain (scalar or vector).
        :param Kd: Derivative gain (scalar or vector).
        :param A_max: Maximum allowable acceleration (scalar or vector, saturation limit).
        :param initial_position: Starting position (scalar or vector).
        :param initial_velocity: Starting velocity (scalar or vector).
        """
        self.dt = dt
        self.Kp = Kp
        self.Kd = Kd
        self.A_max = A_max

        # Convert initial values to numpy arrays (if not already)
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float)
        self.i = 0

    def step(self, goal):
        """
        Advances the state by one time step based on the given goal.

        :param goal: The desired goal (scalar or vector).
        :return: A dictionary containing the goal, updated position, velocity, error, and acceleration.
        """

        tick = time.time()
        goal = np.array(goal, dtype=float)

        error = goal - self.position
        acceleration = self.Kp * error - self.Kd * self.velocity
        acceleration = np.clip(acceleration, -self.A_max, self.A_max)

        # Update velocity and position using Euler integration.
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        self.i += 1

        tock = time.time()
        t = tock - tick

        return {
            "goal": goal,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "error": error,
            "acceleration": acceleration,
            "time": t,
        }


class InputMode(str, Enum):
    SPACEMOUSE = "spacemouse"
    KEYBOARD = "keyboard"
    GELLO = "gello"
    MODEL = "model"
    HELEO = "heleo"


class ControlMode(Enum):
    JOINT = "joint"
    CARTESIAN = "cartesian"


@dataclass
class AccelConfigFactory:
    """Factory for creating AccelConfig"""

    A_max: float = 30.0  # max acceleration

    # proportional gain. controls the stiffness of the system.
    # -- determines how strongly the system responds to the position error.
    # -- A higher `Kp` value means the system will respond more aggressively to position errors, making it stiffer.
    Kp: float = 800.0

    # derivative gain. controls the damping of the system.
    # -- determines how strongly the system responds to change in position error
    # -- A higher `Kd` value means the system will become less oscillatory / more stable.
    Kd: float = 40.0

    def create(self, hz: float) -> Accelerator:
        dt: float = 1 / self.hz
        return Accelerator(dt=dt, A_max=self.A_max, Kp=self.Kp, Kd=self.Kd)


@dataclass
class RobotConfig:
    ip: str = "192.168.1.231"  # robot details
    dof: int = 7

    input: InputMode = InputMode.GELLO
    ctrl: ControlMode = ControlMode.JOINT  # controller details

    hz: int = 200  # command frequency
    grip_hz: int = 50  # gripper frequency
    acc: AccelConfigFactory = field(default_factory=AccelConfigFactory)

    def __post_init__(self):
        if self.input == InputMode.SPACEMOUSE:
            msg = "spacemouse only works with cartesian control"
            assert self.ctrl == ControlMode.CARTESIAN, msg
        if self.input == InputMode.GELLO:
            msg = "gello only works with joint control"
            assert self.ctrl == ControlMode.JOINT, msg

        # TODO is this a bad idea
        # self.acc = self.acc.create(self.hz)


@dataclass
class RobotState:
    angle: list[float]
    pose: list[float]


deg2rad = lambda x: x * np.pi / 180
HOME = RobotState(
    angle=[deg2rad(x) for x in [0, -45, 0, 35, 0, 65, 90]],
    # # angle=[-0.01057, 0.03834, -0.09203, 1.52784, 0.06442, 1.42840, -0.20091],
    pose=[502.73455, -34.92372, 343.89443, -3.09418, -0.07189, 0.08958],
)


class Xarm(Base):
    """Publishes xArm state and camera images."""

    def __init__(self, cfg: RobotConfig):
        super().__init__("xarm_robot")

        self.cfg = cfg
        self.ip = cfg.ip
        self.joint_ctrl = cfg.ctrl == ControlMode.JOINT
        self.dof = 7
        self.t0 = time.time()
        self.tic = time.time()

        self.home = HOME
        self.use_gripper = cfg.use_gripper

        self.robot = XArmAPI(self.ip, is_radian=True)
        self.get_logger().info("Initializing robot.")

        self.robot.connect()
        # self.robot.motion_enable(enable=True)
        # self.robot.set_teach_sensitivity(1, wait=True)

        # 0 is posiition control, 1 is servo control
        # self.robot.set_mode(1)
        self.mode = 1

        # self.robot.set_state(0)
        if self.use_gripper:
            self.robot.set_gripper_enable(True)
            self.robot.set_gripper_mode(0)
            # self.robot.set_gripper_speed(5000)
            self.robot.set_gripper_speed(5000)

        self.get_logger().info("Robot initialized.")

        self.hz = 200
        self.set_period()
        self.griphz = 50

        self.pos = None
        self.joints = None
        self.act = None
        self.leader = None
        self.grip = None
        self.displace = None

        self.acc = Accelerator(
            dt=1 / self.hz,
            A_max=30.0,
            Kp=800.0,
            Kd=40.0,
        )

        match cfg.input:
            case InputMode.GELLO | InputMode.SPACEMOUSE:
                self.ema = 5 / self.hz
                self.gema = 1.0  # gripper ema
            case InputMode.MODEL | InputMode.HELEO:
                self.ema = 0.9
                self.ema = 8 / self.hz
                self.gema = self.ema
                # self.gema = 0.01

        #
        # ROS things
        #

        # self.pub = self.create_publisher(Float32MultiArray, "/robot_state", 10)
        # self.state_timer = self.create_timer(1 / self.hz, self.publish_xarm_state)

        self.cmd_sub = self.create_subscription(Float32MultiArray, "/robot_commands", self.set_act, 10)
        self.gello_sub = self.create_subscription(Float32MultiArray, "/gello/state", self.set_gello, 10)
        self.moveit_sub = self.create_subscription(JointState, "/xarm/joint_states", self.set_joints, 10)
        self.moveit_pose_sub = self.create_subscription(RobotMsg, "/xarm/robot_states", self.set_pose, 10)

        # self.act_timer = self.create_timer(1 / self.hz, self.servo)

        self.get_logger().info("Robot Node Initialized.")

        self.twist_pub = self.create_publisher(TwistStamped, "/servo_server/delta_twist_cmds", 10)
        self.jog_pub = self.create_publisher(JointJog, "/servo_server/delta_joint_cmds", 10)
        self.get_logger().info("Keyboard servo publisher started.")

        # self._stop_event = threading.Event()
        # self._thread = threading.Thread(target=self._command_loop, daemon=True)
        # self._thread.start()
        self.timer = self.create_timer(1 / self.hz, self.run)

        #
        # gripper
        #

        # self.gset = self.create_client(GripperMove, "/xarm/set_gripper_position")
        # self.gget = self.create_client(GetFloat32, "/xarm/get_gripper_position")

        # ros2 service call /xarm/get_gripper_position xarm_msgs/srv/GetFloat32
        # Wait for the service to become available.
        # Create a request object that will be reused.
        # Create a timer to call the service at 50Hz (0.02 sec period).

        if self.use_gripper:
            self.gripper_pub = self.create_publisher(Float32MultiArray, "/xgym/gripper", 10)
            self.gtimer = self.create_timer(1 / self.griphz, self.gripper_callback)

    def _command_loop(self):
        rate = 1.0 / self.hz
        while not self._stop_event.is_set():
            tic = time.time()
            # self.step()
            self.run()
            toc = time.time()
            time.sleep(max(0, rate - (toc - tic)))

    def destroy_node(self):
        self._stop_event.set()
        self._thread.join()
        super().destroy_node()

    def set_active(self, msg):
        super().set_active(msg)

        self.leader = None
        self._clear_error_states()
        self.grip = 1
        noise = np.random.normal(0, 0.01, size=len(self.home.angle))
        self.home.angle = np.array(self.home.angle)  # +noise
        self.acc.velocity = np.zeros_like(self.acc.velocity)

    def gripper_callback(self):
        # self._clear_error_states()
        if not self.active:
            # self.get_logger().info("Inactive.")
            self.robot.set_gripper_position(850, wait=False)
            return

        code, grip = self.robot.get_gripper_position()
        logger = self.get_logger()

        if code or (grip is None):
            return
        grip = grip / 850

        self.gripper_pub.publish(Float32MultiArray(data=[grip]))
        self.grip = grip if self.grip is None else self.grip

        clip = lambda a, b: lambda x: max(min(b, x), a)

        match self.cfg.input:
            case InputMode.GELLO | InputMode.MODEL:
                if self.leader is None:
                    return
                new = self.leader[-1]
            case InputMode.SPACEMOUSE | InputMode.HELEO:
                if self.act is None:
                    return
                new = self.grip + self.act[-1]
            case _:
                raise NotImplementedError("not implemented for this input mode")

        match self.cfg.input:
            case InputMode.GELLO | InputMode.SPACEMOUSE:
                pass  # no binning for teleop
            case _:
                new = clip(0.0, 1.0)(new)
                new = new * (self.gema) + self.grip * (1 - self.gema)
                self.grip = new

        # print(new)
        out = new * 850

        match self.cfg.input:
            case InputMode.MODEL:
                binsize = 850 // 30
                discretize = lambda x: int(x / binsize) * binsize
                out = discretize(out)
            case _:
                pass

        self.robot.set_gripper_position(out, wait=False)

    def response_callback(self, future, func=None):
        try:
            response = future.result()
            if func is not None:
                func(response)
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        # else:
        # Log the response (optional, as 50Hz logging can be overwhelming).
        # self.get_logger().info(f"Service response: {response}")

    def run(self):
        warn = "Only one of act or leader should be set."
        assert self.act is None or self.leader is None, warn
        if self.joint_ctrl:
            assert self.act is None, "no cartesian control in joint control mode"
        if not self.joint_ctrl:
            assert self.leader is None, "no joint control in cartesian control mode"

        match self.cfg.input:
            case InputMode.GELLO | InputMode.MODEL:
                self.move_joints()
            case InputMode.SPACEMOUSE | InputMode.HELEO:
                self.move_cartesian()

    def move_cartesian(self):
        msg = TwistStamped()

        if self.act is None or self.pose is None:
            return

        if self.cfg.input == InputMode.HELEO:
            if not self.active:
                return

            if self.acc.i == 0:
                self.acc.position = self.act
                self.acc.velocity = np.zeros_like(self.act)
            self.acc.position = self.act

            # step = self.act if self.active else None
            # out = self.acc.step(step)

            # if self.p < self.hz:  # velocity ramp during start
            # out["velocity"] = out["velocity"] * (self.p / self.hz)

            # act = self.act[:-1] - self.pose
            # act = out['position'][:-1] - self.pose

            act = np.array(self.act[:-1]) - np.array(self.pose)
            # act = np.array(self.act).copy()
            # self.act = np.array([0]*6 + [self.grip])
            scale = 0.05
            act = np.clip(act, -scale, scale).tolist()
            self.pose = np.array(self.pose) + np.array(act)

            # if np.random.rand() < 0.9:
            # act = np.zeros_like(act).tolist()
            # act = np.zeros_like(act).tolist()
            msg.twist.linear.x = act[0]
            msg.twist.linear.y = act[1]
            msg.twist.linear.z = act[2]
            msg.twist.angular.x = 0.0
            msg.twist.angular.y = 0.0
            msg.twist.angular.z = 0.0

            # msg.velocities = out["velocity"].tolist()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "link_base"
            self.twist_pub.publish(msg)

        if self.cfg.input == InputMode.SPACEMOUSE:
            msg.twist.linear.x = self.act[0]
            msg.twist.linear.y = self.act[1]
            msg.twist.linear.z = self.act[2]
            msg.twist.angular.x = self.act[3]
            msg.twist.angular.y = self.act[4]
            msg.twist.angular.z = self.act[5]

            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "link_base"
            self.twist_pub.publish(msg)

    def move_joints(self):
        if self.joints is None or self.leader is None:
            return

        # if self.acc.i % self.hz == 0:
        # self.get_logger().info(f"Leader: {self.leader.round(2)}")
        # self.get_logger().info(f"Joints: {self.joints.round(2)}")

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "link_base"

        # ros2 topic echo /joint_states to find moveit robot joint names

        if self.acc.i == 0:
            self.acc.position = self.joints
            self.acc.velocity = np.zeros_like(self.joints)
        self.acc.position = self.joints

        step = self.leader[:-1] if self.active else self.home.angle
        out = self.acc.step(step)

        if self.p < self.hz:  # velocity ramp during start
            out["velocity"] = out["velocity"] * (self.p / self.hz)

        msg.displacements = out["position"].tolist()
        msg.joint_names = [self.joint_names[i] for i in range(len(self.leader[:-1]))]
        msg.velocities = out["velocity"].tolist()
        self.jog_pub.publish(msg)

    def publish_xarm_state(self):
        """Publishes xArm state."""

        pos = self.robot.position
        if self.use_gripper:
            code, gripper = self.robot.get_gripper_position()
        else:
            code, gripper = 0, 0
        if code != 0:
            return
        else:
            gripper = [gripper]

        joints = self.robot.angles

        self.pos = pos
        self.joints = joints
        self.gripper = gripper

        state = {
            "cartesian": {
                "cartesian": pos[:3],  # / 1000,  # m to mm
                "aa": pos[3:],  # double check is this axangle or euler... really matrix is best
            },
            "joint": {
                "position": joints,
                # "velocity": self.robot.realtime_joint_speeds,  # deg/s or rad/s
                # "torque": self.robot.joints_torque,
            },
            # "speed": self.robot.last_used_joint_speed,
            # "acceleration": self.robot.last_used_joint_acc,
            # "temperature": self.robot.temperatures,
            # "voltage": self.robot.voltages,
            # "current": self.robot.currents,
            # "is_moving": self.robot.is_moving,
            "gripper": {
                "position": gripper,
            },
        }

        msg = Float32MultiArray()
        data = pos + state["gripper"]["position"] + state["joint"]["position"]
        try:
            data = [float(d) for d in data]
        except TypeError:  # still gets none from some sensor read
            return

        msg.data = data
        self.pub.publish(msg)  # vector len 14

        # self.get_logger().info(f"Published xArm state: {state}")

    def handle_reset(self, msg):
        """Handles the reset request from ros_sandbox.py."""

        self.get_logger().info("Resetting the environment...")
        if msg.data == "Stop":
            self.robot.reset()
        # return response

    def set_act(self, msg):
        self.act = msg.data

    def set_gello(self, msg: Float32MultiArray):
        leader = np.array(msg.data)
        if self.joints is None or self.grip is None:
            return
        if self.leader is None:
            g = self.grip if self.grip is not None else 1
            self.leader = np.array([*self.joints.tolist(), g])  # smooth start
        else:
            # self.leader = leader
            # diff = np.log(np.linalg.norm(leader - self.leader))
            # response = 50/diff # faster moves update quicker

            self.leader[:-1] = leader[:-1] * (self.ema) + self.leader[:-1] * (1 - self.ema)
            self.leader[-1] = leader[-1]  # no smoothing on gripper because gema

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
        self.joints = np.array(msg.position)
        self.joint_names = msg.name

    def servo(self):
        """Servos the robot to the given joint positions."""

        if self.act is None:
            return

        # act = msg.data
        # self.robot.set_servo_angle_j(joints, is_radian=True, wait=False)
        pos = np.array(self.pos)
        pos += np.array(self.act[:6]) / 10
        # self.robot.set_servo_cartesian(pos, is_radian=True)

        joints = [j - 0.01 for j in self.joints]
        self.robot.set_servo_angle_j(joints, is_radian=True)
        # self.get_logger().info(f"Servoing to: {self.act}")
        # self.get_logger().info(f"Servoing to: {self.pos}")

    def _clear_error_states(self):
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
        # self.robot.set_collision_sensitivity(0)
        time.sleep(0.1)
        self.robot.set_state(state=0)
        time.sleep(0.1)
        if self.use_gripper:
            self.robot.set_gripper_enable(True)
            time.sleep(0.1)
            self.robot.set_gripper_mode(0)
            time.sleep(0.1)
            self.robot.set_gripper_speed(5000)
            time.sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    node = RobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Environment Node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
