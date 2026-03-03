from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import time

import numpy as np


class Accelerator:
    """PD controller with Euler integration for smooth velocity ramping."""

    def __init__(
        self,
        dt: float = 0.02,
        kp: float = 1.0,
        kd: float = 0.5,
        a_max: float = 2.0,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
    ):
        self.dt = dt
        self.kp = kp
        self.kd = kd
        self.a_max = a_max
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(initial_velocity, dtype=float)
        self.i = 0

    def step(self, goal: np.ndarray) -> dict:
        """Advance one time step toward goal; return state dict."""
        tick = time.time()
        goal = np.array(goal, dtype=float)
        error = goal - self.position
        acceleration = np.clip(self.kp * error - self.kd * self.velocity, -self.a_max, self.a_max)
        self.velocity += acceleration * self.dt
        self.position += self.velocity * self.dt
        self.i += 1
        return {
            "goal": goal,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "error": error,
            "acceleration": acceleration,
            "time": time.time() - tick,
        }


class InputMode(StrEnum):
    SPACEMOUSE = "spacemouse"
    KEYBOARD = "keyboard"
    GELLO = "gello"
    MODEL = "model"
    HELEO = "heleo"


class ControlMode(StrEnum):
    JOINT = "joint"
    CARTESIAN = "cartesian"


@dataclass
class AccelConfigFactory:
    """Factory for Accelerator instances."""

    a_max: float = 30.0  # max acceleration
    kp: float = 800.0  # proportional gain: stiffness of response to position error
    kd: float = 40.0  # derivative gain: damping against oscillation

    def create(self, hz: float) -> Accelerator:
        return Accelerator(dt=1 / hz, a_max=self.a_max, kp=self.kp, kd=self.kd)


@dataclass
class RobotConfig:
    ip: str | None = "192.168.1.231"  # robot IP address; None → FakeXArm
    dof: int = 7  # degrees of freedom
    input: InputMode = InputMode.GELLO  # input device
    ctrl: ControlMode = ControlMode.JOINT  # control mode
    use_gripper: bool = True
    hz: int = 200  # command frequency (Hz)
    grip_hz: int = 50  # gripper poll frequency (Hz)
    acc: AccelConfigFactory = field(default_factory=AccelConfigFactory)
    cart_scale: float = 0.05  # Cartesian velocity clipping bound per tick
    grip_bins: int = 30  # gripper discretization bins (MODEL mode only)
    grip_max: int = 850  # maximum gripper position in hardware units

    def __post_init__(self) -> None:
        if self.input == InputMode.SPACEMOUSE:
            assert self.ctrl == ControlMode.CARTESIAN, "spacemouse only works with cartesian control"
        if self.input == InputMode.GELLO:
            assert self.ctrl == ControlMode.JOINT, "gello only works with joint control"


@dataclass
class RobotState:
    """Home configuration for the robot."""

    angle: list[float]  # home joint angles (radians)
    pose: list[float]  # home Cartesian pose [x, y, z, rx, ry, rz] in mm


HOME = RobotState(
    angle=[np.deg2rad(x) for x in [0, -45, 0, 35, 0, 65, 90]],
    pose=[502.73455, -34.92372, 343.89443, -3.09418, -0.07189, 0.08958],
)


class RobotPolicy:
    """Pure-Python control policy for the xArm; no ROS dependencies.

    Owns EMA smoothing, PD acceleration, and input-mode dispatch.
    The ROS node calls update_* methods with plain numpy arrays, then
    calls step_* methods and publishes the returned commands.
    """

    def __init__(self, cfg: RobotConfig, home: RobotState = HOME):
        self.cfg = cfg
        self.home = home
        self.acc = cfg.acc.create(cfg.hz)

        # mutable robot state
        self._joints: np.ndarray | None = None
        self._joint_names: list[str] = []
        self._pose: np.ndarray | None = None
        self._leader: np.ndarray | None = None  # EMA-smoothed leader joints+grip
        self._act: np.ndarray | None = None  # model/heleo action
        self._grip: float | None = None  # EMA-smoothed normalized grip [0, 1]
        self._active: bool = False
        self._p: int = 0  # ticks since last activation (for velocity ramp)

        # EMA constants depend on input mode
        match cfg.input:
            case InputMode.GELLO | InputMode.SPACEMOUSE:
                self._ema = 5.0 / cfg.hz
                self._gema = 1.0  # no gripper smoothing for teleop
            case InputMode.MODEL | InputMode.HELEO:
                self._ema = 8.0 / cfg.hz
                self._gema = self._ema

    @property
    def active(self) -> bool:
        return self._active

    # --- state update methods (called from ROS callbacks) ---

    def update_joints(self, joints: np.ndarray, joint_names: list[str]) -> None:
        if len(joints) == 6:
            return  # 6-DOF variant from wrong topic; skip
        self._joints = joints
        self._joint_names = joint_names

    def update_pose(self, pose: np.ndarray) -> None:
        self._pose = pose

    def update_command(self, act: np.ndarray) -> None:
        """Store a model or heleo action command."""
        self._act = act

    def update_leader(self, raw: np.ndarray) -> None:
        """EMA-smooth the incoming leader command; initialize from robot state."""
        if self._joints is None or self._grip is None:
            return
        if self._leader is None:
            g = self._grip
            self._leader = np.array([*self._joints.tolist(), g])  # smooth start
        else:
            self._leader[:-1] = raw[:-1] * self._ema + self._leader[:-1] * (1 - self._ema)
            self._leader[-1] = raw[-1]  # gripper passed through; gema applied in step_gripper

    def on_active(self, active: bool) -> None:
        """Reset transient state when the active flag changes."""
        self._active = active
        self._p = 0
        self._leader = None
        self._grip = 1.0
        self.home.angle = np.array(self.home.angle)
        self.acc.velocity = np.zeros_like(self.acc.velocity)

    def tick(self) -> None:
        """Advance the period counter; call once per control cycle."""
        self._p += 1

    # --- output (step) methods ---

    def step_joints(self) -> tuple[list[float], list[str], list[float]] | None:
        """Compute joint jog command.

        Returns (displacements, joint_names, velocities) or None if not ready.
        """
        if self._joints is None or self._leader is None:
            return None

        if self.acc.i == 0:
            self.acc.position = self._joints.copy()
            self.acc.velocity = np.zeros_like(self._joints)
        self.acc.position = self._joints.copy()

        goal = self._leader[:-1] if self._active else np.array(self.home.angle)
        out = self.acc.step(goal)

        vel = out["velocity"]
        if self._p < self.cfg.hz:  # ramp velocity during first second
            vel = vel * (self._p / self.cfg.hz)

        n = len(self._leader) - 1
        names = [self._joint_names[i] for i in range(n)]
        return out["position"].tolist(), names, vel.tolist()

    def step_cartesian(self) -> list[float] | None:
        """Compute Cartesian twist command [lx, ly, lz, ax, ay, az].

        Returns 6-element list or None if not ready.
        """
        if self._act is None or self._pose is None:
            return None

        if self.cfg.input == InputMode.SPACEMOUSE:
            return list(self._act[:6])

        if self.cfg.input == InputMode.HELEO:
            if not self._active:
                return None
            if self.acc.i == 0:
                self.acc.position = self._act.copy()
                self.acc.velocity = np.zeros_like(self._act)
            self.acc.position = self._act.copy()

            delta = np.array(self._act[:-1]) - np.array(self._pose)
            delta = np.clip(delta, -self.cfg.cart_scale, self.cfg.cart_scale)
            self._pose = np.array(self._pose) + delta
            return [*delta.tolist(), 0.0, 0.0, 0.0]

        return None

    def step_gripper(self, grip_raw: float) -> int | None:
        """Compute gripper position command in hardware units [0, grip_max].

        :param grip_raw: current gripper position from hardware [0, grip_max]
        :returns: gripper command, or None to skip
        """
        if not self._active:
            return self.cfg.grip_max  # fully open when inactive

        grip_norm = grip_raw / self.cfg.grip_max
        if self._grip is None:
            self._grip = grip_norm

        match self.cfg.input:
            case InputMode.GELLO | InputMode.MODEL:
                if self._leader is None:
                    return None
                new = float(self._leader[-1])
            case InputMode.SPACEMOUSE | InputMode.HELEO:
                if self._act is None:
                    return None
                new = self._grip + float(self._act[-1])
            case _:
                raise NotImplementedError(f"no gripper logic for {self.cfg.input}")

        match self.cfg.input:
            case InputMode.GELLO | InputMode.SPACEMOUSE:
                pass  # no EMA or binning for teleop
            case _:
                new = float(np.clip(new, 0.0, 1.0))
                new = new * self._gema + self._grip * (1 - self._gema)
                self._grip = new

        out = new * self.cfg.grip_max
        if self.cfg.input == InputMode.MODEL:
            bin_sz = self.cfg.grip_max // self.cfg.grip_bins
            out = int(out / bin_sz) * bin_sz

        return int(out)
