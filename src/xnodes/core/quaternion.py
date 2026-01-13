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
    """Incremental quaternion smoother."""

    def __init__(self, alpha: float = 0.2, q_init: Iterable[float] | np.ndarray | None = None) -> None:
        self.alpha = float(alpha)
        self.q = q_norm(q_init) if q_init is not None else None

    def update(self, q_meas: Iterable[float] | np.ndarray) -> np.ndarray:
        q_meas_n = q_norm(q_meas)
        if self.q is None:
            self.q = q_meas_n
        else:
            self.q = q_slerp(self.q, q_meas_n, self.alpha)
        return self.q
