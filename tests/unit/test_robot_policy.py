from __future__ import annotations

import numpy as np
import pytest

from xnodes.components.robot import (
    AccelConfigFactory,
    ControlMode,
    InputMode,
    RobotConfig,
    RobotPolicy,
)

JOINTS_7 = np.zeros(7, dtype=np.float64)
NAMES_7 = [f"joint{i}" for i in range(7)]


def _gello_policy() -> RobotPolicy:
    return RobotPolicy(RobotConfig(input=InputMode.GELLO, ctrl=ControlMode.JOINT))


def _model_policy() -> RobotPolicy:
    return RobotPolicy(RobotConfig(input=InputMode.MODEL, ctrl=ControlMode.JOINT))


def _spacemouse_policy() -> RobotPolicy:
    return RobotPolicy(RobotConfig(input=InputMode.SPACEMOUSE, ctrl=ControlMode.CARTESIAN))


# ---------------------------------------------------------------------------
# RobotConfig validation
# ---------------------------------------------------------------------------


class TestRobotConfig:
    def test_gello_requires_joint_control(self) -> None:
        with pytest.raises(AssertionError):
            RobotConfig(input=InputMode.GELLO, ctrl=ControlMode.CARTESIAN)

    def test_spacemouse_requires_cartesian_control(self) -> None:
        with pytest.raises(AssertionError):
            RobotConfig(input=InputMode.SPACEMOUSE, ctrl=ControlMode.JOINT)

    def test_default_config_builds(self) -> None:
        cfg = RobotConfig()
        assert cfg.hz == 200
        assert cfg.grip_max == 850
        assert cfg.grip_bins == 30
        assert cfg.cart_scale == 0.05

    def test_accel_factory_creates_accelerator(self) -> None:
        factory = AccelConfigFactory(a_max=10.0, kp=100.0, kd=5.0)
        acc = factory.create(hz=100)
        assert acc.dt == pytest.approx(0.01)
        assert acc.a_max == 10.0


# ---------------------------------------------------------------------------
# Policy returns None / grip_max when state is missing
# ---------------------------------------------------------------------------


class TestPolicyNotReady:
    def test_step_joints_none_without_joints(self) -> None:
        assert _gello_policy().step_joints() is None

    def test_step_joints_none_without_leader(self) -> None:
        p = _gello_policy()
        p.update_joints(JOINTS_7, NAMES_7)
        assert p.step_joints() is None

    def test_step_cartesian_none_without_state(self) -> None:
        assert _spacemouse_policy().step_cartesian() is None

    def test_step_gripper_returns_open_when_inactive(self) -> None:
        p = _gello_policy()
        assert p.step_gripper(0.0) == p.cfg.grip_max

    def test_step_gripper_returns_open_regardless_of_raw(self) -> None:
        p = _gello_policy()
        assert p.step_gripper(850.0) == p.cfg.grip_max


# ---------------------------------------------------------------------------
# Joint control
# ---------------------------------------------------------------------------


class TestJointControl:
    def test_returns_three_lists_when_ready(self) -> None:
        p = _gello_policy()
        p.on_active(True)
        p.update_joints(JOINTS_7, NAMES_7)
        p._leader = np.array([*JOINTS_7.tolist(), 0.5])
        result = p.step_joints()
        assert result is not None
        displacements, joint_names, velocities = result
        assert len(displacements) == 7
        assert len(joint_names) == 7
        assert len(velocities) == 7

    def test_joint_names_match_input(self) -> None:
        p = _gello_policy()
        p.on_active(True)
        p.update_joints(JOINTS_7, NAMES_7)
        p._leader = np.array([*JOINTS_7.tolist(), 0.5])
        _, joint_names, _ = p.step_joints()
        assert joint_names == NAMES_7

    def test_velocity_ramp_zeros_at_p_zero(self) -> None:
        """At period 0 the velocity multiplier is 0, so all velocities are zero."""
        p = _gello_policy()
        p.on_active(True)
        p.update_joints(JOINTS_7, NAMES_7)
        p._leader = np.array([0.5] * 7 + [0.5])  # non-zero error
        p._p = 0
        _, _, velocities = p.step_joints()
        assert all(v == pytest.approx(0.0) for v in velocities)

    def test_velocity_ramp_nonzero_after_full_period(self) -> None:
        """After hz ticks the ramp is complete; PD response to non-zero error is non-zero."""
        p = _gello_policy()
        p.on_active(True)
        p.update_joints(JOINTS_7, NAMES_7)
        p._leader = np.array([0.5] * 7 + [0.5])  # non-zero error
        p._p = p.cfg.hz
        _, _, velocities = p.step_joints()
        assert any(v != pytest.approx(0.0) for v in velocities)

    def test_inactive_tracks_home_not_leader(self) -> None:
        """When inactive the goal is home.angle, not the leader."""
        p = _gello_policy()
        p.on_active(False)
        p.update_joints(JOINTS_7, NAMES_7)
        # leader far from home — if used it would dominate the output
        p._leader = np.array([10.0] * 7 + [0.5])
        p._p = p.cfg.hz
        displacements, _, _ = p.step_joints()
        home = np.array(p.home.angle)
        # position should be tracking toward home, not toward 10.0
        assert np.max(np.abs(np.array(displacements) - home)) < np.max(np.abs(np.array(displacements) - 10.0))


# ---------------------------------------------------------------------------
# Cartesian control
# ---------------------------------------------------------------------------


class TestCartesianControl:
    def test_spacemouse_returns_six_dof(self) -> None:
        p = _spacemouse_policy()
        p.update_command(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]))
        p.update_pose(np.zeros(6, dtype=np.float32))
        result = p.step_cartesian()
        assert result is not None
        assert len(result) == 6
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    def test_spacemouse_ignores_7th_element(self) -> None:
        """Gripper component should not appear in the twist."""
        p = _spacemouse_policy()
        p.update_command(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.9]))
        p.update_pose(np.zeros(6, dtype=np.float32))
        result = p.step_cartesian()
        assert result is not None
        assert result[5] == pytest.approx(0.0)

    def test_heleo_returns_none_when_inactive(self) -> None:
        cfg = RobotConfig(input=InputMode.HELEO, ctrl=ControlMode.CARTESIAN)
        p = RobotPolicy(cfg)
        p.on_active(False)
        p.update_command(np.ones(7))
        p.update_pose(np.zeros(6, dtype=np.float32))
        assert p.step_cartesian() is None

    # def test_heleo_clips_delta_to_cart_scale(self) -> None:
    # cfg = RobotConfig(input=InputMode.HELEO, ctrl=ControlMode.CARTESIAN)
    # p = RobotPolicy(cfg)
    # p.on_active(True)
    # # act[:-1] far from pose → delta would exceed cart_scale without clipping
    # p.update_command(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]))
    # p.update_pose(np.zeros(6, dtype=np.float32))
    # result = p.step_cartesian()
    # assert result is not None
    # assert len(result) == 6
    # for v in result[:3]:
    # assert abs(v) <= pytest.approx(cfg.cart_scale)

    # def test_heleo_angular_components_are_zero(self) -> None:
    # cfg = RobotConfig(input=InputMode.HELEO, ctrl=ControlMode.CARTESIAN)
    # p = RobotPolicy(cfg)
    # p.on_active(True)
    # p.update_command(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5]))
    # p.update_pose(np.zeros(6, dtype=np.float32))
    # result = p.step_cartesian()
    # assert result is not None
    # # assert result[3] == pytest.approx(0.0)
    # assert result[4] == pytest.approx(0.0)
    # assert result[5] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Gripper control
# ---------------------------------------------------------------------------


class TestGripperControl:
    def test_gello_passes_leader_gripper_through(self) -> None:
        p = _gello_policy()
        p.on_active(True)
        p._leader = np.array([*JOINTS_7.tolist(), 0.4])
        cmd = p.step_gripper(340.0)
        assert cmd == int(0.4 * 850)

    def test_model_mode_applies_ema_and_discretizes(self) -> None:
        p = _model_policy()
        p.on_active(True)
        p._leader = np.array([*JOINTS_7.tolist(), 0.6])
        cmd = p.step_gripper(425.0)
        assert cmd is not None
        bin_sz = p.cfg.grip_max // p.cfg.grip_bins
        assert cmd % bin_sz == 0

    def test_model_mode_discretized_within_range(self) -> None:
        p = _model_policy()
        p.on_active(True)
        p._leader = np.array([*JOINTS_7.tolist(), 0.6])
        cmd = p.step_gripper(425.0)
        assert cmd is not None
        assert 0 <= cmd <= p.cfg.grip_max

    def test_gripper_none_when_leader_missing(self) -> None:
        p = _gello_policy()
        p.on_active(True)
        assert p.step_gripper(425.0) is None

    def test_spacemouse_gripper_adds_delta_to_current(self) -> None:
        p = _spacemouse_policy()
        p.on_active(True)
        p._grip = 0.5  # current normalized grip
        p._act = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])  # gripper delta = 0.1
        cmd = p.step_gripper(425.0)
        assert cmd is not None
        # new = 0.5 + 0.1 = 0.6 → 0.6 * 850 = 510
        assert cmd == int(0.6 * 850)


# ---------------------------------------------------------------------------
# on_active resets
# ---------------------------------------------------------------------------


class TestOnActive:
    def test_resets_leader(self) -> None:
        p = _gello_policy()
        p._leader = np.ones(8)
        p.on_active(True)
        assert p._leader is None

    def test_resets_grip_to_one(self) -> None:
        p = _gello_policy()
        p._grip = 0.3
        p.on_active(True)
        assert p._grip == pytest.approx(1.0)

    def test_resets_period_counter(self) -> None:
        p = _gello_policy()
        p._p = 500
        p.on_active(True)
        assert p._p == 0

    def test_zeros_accelerator_velocity(self) -> None:
        p = _gello_policy()
        p.acc.velocity = np.ones(7)
        p.on_active(True)
        assert np.all(p.acc.velocity == pytest.approx(0.0))

    def test_active_property_reflects_flag(self) -> None:
        p = _gello_policy()
        p.on_active(True)
        assert p.active is True
        p.on_active(False)
        assert p.active is False


# ---------------------------------------------------------------------------
# tick() and update_leader EMA
# ---------------------------------------------------------------------------


class TestTickAndEMA:
    def test_tick_increments_period(self) -> None:
        p = _gello_policy()
        assert p._p == 0
        p.tick()
        assert p._p == 1
        p.tick()
        assert p._p == 2

    def test_update_leader_skipped_when_grip_none(self) -> None:
        p = _gello_policy()
        p.update_joints(JOINTS_7, NAMES_7)
        p._grip = None
        p.update_leader(np.ones(8))
        assert p._leader is None

    def test_update_leader_skipped_when_joints_none(self) -> None:
        p = _gello_policy()
        p._grip = 0.5
        p.update_leader(np.ones(8))
        assert p._leader is None

    def test_update_leader_initializes_from_robot_state(self) -> None:
        """First update seeds leader from current joints, not from raw input."""
        p = _gello_policy()
        p.update_joints(JOINTS_7, NAMES_7)
        p._grip = 0.5
        p.update_leader(np.ones(8))  # raw = ones, but leader seeded from zeros + grip
        assert p._leader is not None
        np.testing.assert_allclose(p._leader[:7], JOINTS_7)
        assert p._leader[-1] == pytest.approx(0.5)

    def test_update_leader_ema_smooths_subsequent_updates(self) -> None:
        """Second update applies EMA toward new target."""
        p = _gello_policy()
        p.update_joints(JOINTS_7, NAMES_7)
        p._grip = 0.5
        p.update_leader(np.ones(8))  # initializes leader to zeros
        prev = p._leader[0]
        p.update_leader(np.ones(8))  # EMA toward 1.0
        assert p._leader[0] > prev  # smoothed toward 1.0
