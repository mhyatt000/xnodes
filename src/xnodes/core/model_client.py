from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

CAMERA_PREFIX = "/cam"
CAMERA_KEYS = ("low", "side")  # "wrist"
CAMERA_POSTFIX = "image_raw"


class ActionRepresentation(Enum):
    ABS = "absolute"
    REL = "relative"


@dataclass
class ModelClientConfig:
    host: str
    port: int

    rep: ActionRepresentation
    task: str
    ensemble: bool = False  # weighted average across time


NOMODEL = ModelClientConfig(
    host="none",
    port=8000,
    rep=ActionRepresentation.REL,
    task="none",
    ensemble=False,
)


class GaussianConv:
    def __init__(self, kernel_size: int = 5, std: float = 1.0):
        if kernel_size % 2 != 1:
            msg = "kernel_size must be odd for symmetric smoothing"
            raise ValueError(msg)
        self.kernel_size = kernel_size
        self.std = std
        self.half_k = kernel_size // 2

        x = np.arange(kernel_size) - self.half_k
        self.kernel = np.exp(-0.5 * (x / std) ** 2)
        self.kernel /= self.kernel.sum()
        self.buffer = deque(maxlen=kernel_size)

    def __call__(self, x: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray | float:
        arr = np.atleast_1d(x).astype(float)

        if arr.shape[0] == 1:
            self.buffer.append(arr.item())
            buffer_np = np.array(self.buffer)
            n = len(buffer_np)
            i = len(buffer_np) - 1
            start = max(0, i - self.half_k)
            end = min(n, i + self.half_k + 1)
            k_start = self.half_k - (i - start)
            k_end = k_start + (end - start)
            values = buffer_np[start:end]
            weights = self.kernel[k_start:k_end]
            return float(np.average(values, weights=weights))

        self.buffer.clear()
        self.buffer.extend(arr.tolist())
        result: list[float] = []
        n = len(arr)
        for i in range(n):
            start = max(0, i - self.half_k)
            end = min(n, i + self.half_k + 1)
            k_start = self.half_k - (i - start)
            k_end = k_start + (end - start)
            values = arr[start:end]
            weights = self.kernel[k_start:k_end]
            result.append(float(np.average(values, weights=weights)))
        return np.array(result, dtype=np.float64)


def extract_camera_images(
    data: Mapping[str, np.ndarray],
    prefix: str = CAMERA_PREFIX,
    keys: tuple[str, str, str] = CAMERA_KEYS,
) -> dict[str, np.ndarray | None]:
    return {k: data.get(f"{prefix}{k}") for k in keys}


def missing_camera_images(images: Mapping[str, np.ndarray | None]) -> list[str]:
    return [k for k, v in images.items() if v is None]


def build_idle_target(joints: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    return np.concatenate([joints, gripper], axis=-1)


def build_policy_payload(
    *,
    joints: np.ndarray,
    pose: np.ndarray,
    gripper: np.ndarray,
    images: Mapping[str, np.ndarray],
    ensemble: bool = False,
) -> dict[str, Any]:
    if np.any(gripper > 1.0):
        raise ValueError("Gripper out of bounds")

    pose_m = pose.astype(np.float32, copy=True)
    pose_m[:3] /= 1000.0  # mm -> m

    state = build_idle_target(joints, gripper).astype(np.float32)
    pixels = {k: v.astype(np.uint8) for k, v in images.items()}
    obs = {
        "pixels": {k: v.reshape(1, *v.shape) for k, v in pixels.items()},
        "agent_pos": state.reshape(1, *state.shape),
    }
    return {
        "observation": {
            "image_left_wrist": obs["pixels"]["wrist"][None],
            "image_primary": obs["pixels"]["low"][None],
            "image_side": obs["pixels"]["side"][None],
            "proprio_single_arm": np.concatenate(
                [pose_m.reshape(1, 1, -1), state.reshape(1, 1, -1)],
                axis=-1,
            ),
        },
        "ensemble": ensemble,
    }


def extract_action_targets(actions: Mapping[str, Any], resolution: int) -> np.ndarray:
    if resolution < 1:
        msg = "resolution must be >= 1"
        raise ValueError(msg)
    key = "actions" if "actions" in actions else "action"
    if key not in actions:
        msg = "Missing action key in model response"
        raise KeyError(msg)

    act = np.asarray(actions[key]).copy()
    act = act.reshape(-1, act.shape[-1])  # ensure 2D
    return act[::resolution].copy()
