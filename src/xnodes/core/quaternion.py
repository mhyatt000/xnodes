"""Quaternion helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def q_norm(q: Iterable[float] | np.ndarray) -> np.ndarray:
    """Normalize a quaternion."""

    arr = np.asarray(q, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        msg = "quaternion norm is zero"
        raise ValueError(msg)
    return arr / norm


def q_slerp(q0: Iterable[float] | np.ndarray, q1: Iterable[float] | np.ndarray, t: float) -> np.ndarray:
    """Slerp between q0 and q1."""

    qa = q_norm(q0)
    qb = q_norm(q1)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return q_norm(qa + t * (qb - qa))
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    w0 = np.sin((1.0 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    return qa * w0 + qb * w1


def angle_between(q_a: Iterable[float] | np.ndarray, q_b: Iterable[float] | np.ndarray) -> float:
    """Return angle between two quaternions."""

    qa = q_norm(q_a)
    qb = q_norm(q_b)
    dot = abs(float(np.dot(qa, qb)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


class SlerpSmoother:
    """Incremental quaternion smoother with jump cutoff."""

    def __init__(
        self,
        alpha: float = 0.2,
        q_init: Iterable[float] | np.ndarray | None = None,
        max_angle_rad: float | None = 1.0,
    ) -> None:
        self.alpha = float(alpha)
        self.max_angle_rad = float(max_angle_rad) if max_angle_rad is not None else None
        self.q = q_norm(q_init) if q_init is not None else None

    def update(self, q_meas: Iterable[float] | np.ndarray) -> np.ndarray:
        q_meas_n = q_norm(q_meas)

        if self.q is None:
            self.q = q_meas_n
            return self.q

        # If the change is >= threshold, reject update
        if self.max_angle_rad and angle_between(self.q, q_meas_n) >= self.max_angle_rad:
            return self.q

        self.q = q_slerp(self.q, q_meas_n, self.alpha)
        return self.q


class PoseSmootherXYZ:
    """Incremental smoother for 3D positions (x, y, z)."""

    def __init__(self, alpha: float = 0.2, p_init: Iterable[float] | np.ndarray | None = None) -> None:
        """
        Parameters
        ----------
        alpha:
            Smoothing factor in (0, 1]. Higher = more responsive, lower = smoother.
        p_init:
            Optional initial position [x, y, z]. If None, first update sets the state.
        """

        self.alpha = float(alpha)
        self.p: np.ndarray | None = None
        if p_init is not None:
            p_arr = np.asarray(p_init, dtype=float)
            if p_arr.shape != (3,):
                msg = f"p_init must have shape (3,), got {p_arr.shape}"
                raise ValueError(msg)
            self.p = p_arr

    def update(self, p_meas: Iterable[float] | np.ndarray) -> np.ndarray:
        """Update with a new measured position and return the smoothed position."""

        p_meas_arr = np.asarray(p_meas, dtype=float)
        if p_meas_arr.shape != (3,):
            msg = f"p_meas must have shape (3,), got {p_meas_arr.shape}"
            raise ValueError(msg)

        if self.p is None:
            self.p = p_meas_arr
        else:
            a = self.alpha
            self.p = (1.0 - a) * self.p + a * p_meas_arr
        return self.p
