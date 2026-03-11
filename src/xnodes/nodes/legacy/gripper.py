from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from xarm_msgs.srv import Call, GetFloat32, GripperMove, SetFloat32, SetInt16

from xnodes.components.robot import RobotBackend, RobotConfig, RobotPolicy
from xnodes.nodes.legacy.active import ActiveFlag


class RobotDriver(Protocol):
    def connect(self) -> None: ...

    def set_gripper_enable(self, enable: bool) -> None: ...

    def set_gripper_mode(self, mode: int) -> None: ...

    def set_gripper_speed(self, speed: int) -> None: ...

    def get_gripper_position(self) -> tuple[int, float]: ...

    def set_gripper_position(self, pos: float, wait: bool = True) -> None: ...

    def set_state(self, state: int = 0) -> int: ...

    def set_mode(self, mode: int) -> int: ...

    def clean_error(self) -> None: ...

    def clean_gripper_error(self) -> None: ...

    def clean_warn(self) -> None: ...

    def motion_enable(self, enable: bool) -> None: ...


class FakeXArm:
    """In-memory stand-in for XArmAPI; constructed when cfg.ip is None."""

    def __init__(self) -> None:
        self._grip: float = 425.0

    def connect(self) -> None:
        pass

    def set_gripper_enable(self, enable: bool) -> None:
        pass

    def set_gripper_mode(self, mode: int) -> None:
        pass

    def set_gripper_speed(self, speed: int) -> None:
        pass

    def get_gripper_position(self) -> tuple[int, float]:
        return 0, self._grip

    def set_gripper_position(self, pos: float, wait: bool = True) -> None:  # noqa: ARG002
        self._grip = float(pos)

    def set_state(self, state: int = 0) -> int:  # noqa: ARG002
        return 0

    def set_mode(self, mode: int) -> int:  # noqa: ARG002
        return 0

    def clean_error(self) -> None:
        pass

    def clean_gripper_error(self) -> None:
        pass

    def clean_warn(self) -> None:
        pass

    def motion_enable(self, enable: bool) -> None:
        pass


class ServiceXArm:
    """ROS service-backed xArm adapter.

    Matches the subset of the XArmAPI surface used by this node so the backend
    can be swapped via dependency injection without changing the control code.

    ros2 service call /xarm/set_gripper_enable xarm_msgs/srv/SetInt16 "{data: 1}"
    ros2 service call /xarm/set_gripper_mode xarm_msgs/srv/SetInt16 "{data: 0}"
    ros2 service call /xarm/set_gripper_speed xarm_msgs/srv/SetFloat32 "{data: 1500.0}"
    ros2 service call /xarm/set_gripper_position xarm_msgs/srv/GripperMove "{pos: 500.0, wait: true, timeout: 10.0}"
    ros2 service call /xarm/get_gripper_position xarm_msgs/srv/GetFloat32 "{}"
    ros2 service call /xarm/get_gripper_err_code xarm_msgs/srv/GetInt16 "{}"
    ros2 service call /xarm/clean_gripper_error xarm_msgs/srv/Call "{}"

    ros2 action send_goal /xarm_gripper/gripper_action control_msgs/action/GripperCommand "{command: {position: 0.5, max_effort: 0.0}}"
    """

    def __init__(self, node: Node, cfg: RobotConfig) -> None:
        self._node = node
        self._timeout_s = cfg.service_wait_s
        self._move_timeout_s = 10.0
        service_ns = cfg.service_ns.strip()
        if not service_ns.startswith("/"):
            service_ns = f"/{service_ns}"
        self._ns = service_ns.rstrip("/")

        self._set_gripper_enable = self._client(SetInt16, "set_gripper_enable")
        self._set_gripper_mode = self._client(SetInt16, "set_gripper_mode")
        self._set_gripper_speed = self._client(SetFloat32, "set_gripper_speed")
        self._get_gripper_position = self._client(GetFloat32, "get_gripper_position")
        self._set_gripper_position = self._client(GripperMove, "set_gripper_position")
        self._clean_gripper_error = self._client(Call, "clean_gripper_error")
        self._set_state = self._client(SetInt16, "set_state")
        self._set_mode = self._client(SetInt16, "set_mode")
        self._clean_error = self._client(Call, "clean_error")
        self._clean_warn = self._client(Call, "clean_warn")
        self._motion_enable = self._client(SetInt16, "motion_enable")
        self._gripper_action = ActionClient(node, GripperCommand, "/xarm_gripper/gripper_action")
        if not self._gripper_action.wait_for_server(timeout_sec=self._timeout_s):
            msg = f"Action /xarm_gripper/gripper_action unavailable after {self._timeout_s:.1f}s"
            raise RuntimeError(msg)

    def connect(self) -> None:
        self._node.get_logger().info(f"Using ROS service backend in namespace {self._ns}")

    def set_gripper_enable(self, enable: bool) -> None:
        req = SetInt16.Request()
        req.data = int(enable)
        self._call(self._set_gripper_enable, req, "set_gripper_enable")

    def set_gripper_mode(self, mode: int) -> None:
        req = SetInt16.Request()
        req.data = int(mode)
        self._call(self._set_gripper_mode, req, "set_gripper_mode")

    def set_gripper_speed(self, speed: int) -> None:
        req = SetFloat32.Request()
        req.data = float(speed)
        self._call(self._set_gripper_speed, req, "set_gripper_speed")

    def get_gripper_position(self) -> tuple[int, float]:
        resp = self._call(self._get_gripper_position, GetFloat32.Request(), "get_gripper_position")
        return int(resp.ret), float(resp.data)

    def set_gripper_position(self, pos: float, wait: bool = True) -> None:
        goal = GripperCommand.Goal()
        goal.command.position = self._hardware_to_action_position(float(pos))
        goal.command.max_effort = 0.0
        if wait:
            self._gripper_action.send_goal(goal)
        else:
            self._gripper_action.send_goal_async(goal)

    def set_state(self, state: int = 0) -> int:
        req = SetInt16.Request()
        req.data = int(state)
        return int(self._call(self._set_state, req, "set_state").ret)

    def set_mode(self, mode: int) -> int:
        req = SetInt16.Request()
        req.data = int(mode)
        return int(self._call(self._set_mode, req, "set_mode").ret)

    def clean_robot_error(self) -> None:
        self._call(self._clean_error, Call.Request(), "clean_error")

    def clean_gripper_error(self) -> None:
        self._call(self._clean_gripper_error, Call.Request(), "clean_gripper_error")

    def clean_error(self) -> None:
        self.clean_robot_error()
        self.clean_gripper_error()

    def clean_warn(self) -> None:
        self._call(self._clean_warn, Call.Request(), "clean_warn")

    def motion_enable(self, enable: bool) -> None:
        req = SetInt16.Request()
        req.data = int(enable)
        self._call(self._motion_enable, req, "motion_enable")

    def _service_name(self, name: str) -> str:
        return f"{self._ns}/{name}"

    def _hardware_to_action_position(self, pos: float) -> float:
        # xArm gripper service positions use 0..850 with 0 closed, 850 open.
        # The action uses the inverse virtual joint convention: ~0.0 open, ~0.86 closed.
        grip_max = 850.0
        closed_rad = 0.86
        norm_open = max(0.0, min(pos / grip_max, 1.0))
        return (1.0 - norm_open) * closed_rad

    def _client(self, srv_type, name: str):
        service_name = self._service_name(name)
        client = self._node.create_client(srv_type, service_name)
        if not client.wait_for_service(timeout_sec=self._timeout_s):
            msg = f"Service {service_name} unavailable after {self._timeout_s:.1f}s"
            raise RuntimeError(msg)
        return client

    def _call(self, client, request, name: str):
        resp = client.call(request)
        ret = int(resp.ret)
        if ret != 0:
            self._node.get_logger().warn(f"{self._service_name(name)} returned ret={ret}: {resp.message}")
        return resp


RobotDriverPlugin = Callable[[Node, RobotConfig], RobotDriver]


def create_robot_driver(node: Node, cfg: RobotConfig) -> RobotDriver:
    match cfg.backend:
        case RobotBackend.FAKE:
            return FakeXArm()
        case RobotBackend.SERVICE:
            return ServiceXArm(node, cfg)
        case RobotBackend.SDK | RobotBackend.AUTO:
            from xarm.wrapper import XArmAPI

            return XArmAPI(cfg.ip, is_radian=True)
        case _:
            raise ValueError(f"Unsupported backend: {cfg.backend}")


class GripperController:
    """Own gripper backend wiring, setup, recovery, and periodic command flow."""

    def __init__(
        self,
        node: Node,
        cfg: RobotConfig,
        policy: RobotPolicy,
        activator: ActiveFlag,
        driver_plugin: RobotDriverPlugin | None = None,
    ) -> None:
        self._node = node
        self._cfg = cfg
        self._policy = policy
        self._activator = activator
        self._driver = (driver_plugin or create_robot_driver)(node, cfg)
        self._driver.connect()

        self._pub = None
        self._timer = None
        if cfg.use_gripper:
            self._pub = node.create_publisher(Float32MultiArray, "/xgym/gripper", 10)
            self._timer = node.create_timer(1 / cfg.grip_hz, self._grip_tick)
            self.setup()
            activator.add_hook(lambda _active: self._driver.clean_gripper_error())

    @property
    def logger(self):
        return self._node.get_logger()

    @property
    def driver(self) -> RobotDriver:
        return self._driver

    def setup(self) -> None:
        if not self._cfg.use_gripper:
            return
        self._driver.set_gripper_enable(True)
        self._driver.set_gripper_mode(0)
        self._driver.set_gripper_speed(self._cfg.grip_speed)

    def _grip_tick(self) -> None:
        code, grip_raw = self._driver.get_gripper_position()
        if code or grip_raw is None or self._pub is None:
            return
        self._pub.publish(Float32MultiArray(data=[grip_raw / self._cfg.grip_max]))
        cmd = self._policy.step_gripper(grip_raw)
        # cmd = grip_raw +( random.random() -0.5)*50
        self.logger.info(f"grip_tick: raw={grip_raw:.1f} cmd={cmd}")
        if cmd is not None:
            self._driver.set_gripper_position(cmd, wait=False)
