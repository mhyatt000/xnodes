#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import jax
import numpy as np
from rich import print
import tyro
from webpolicy.client import Client

from xnodes.core.model_client import build_policy_payload, extract_action_targets


@dataclass
class Config:
    host: str = "localhost"  # model server host
    port: int = 8000  # model server port
    hz: float = 5.0  # loop frequency
    image_size: int = 128  # square dummy image size
    resolution: int = 1  # action downsample factor
    ensemble: bool = False  # request ensembled policy output
    steps: int = 0  # 0 means run forever


class MyClient(Client):
    def reset(self):
        self.step({"reset": True})


def dummy_payload(image_size: int, ensemble: bool, tick: int) -> dict[str, Any]:
    joints = np.linspace(0.0, 0.6, 7, dtype=np.float32)
    pose = np.array([120.0, 210.0, 350.0, 0.1, -0.2, 0.3], dtype=np.float32)  # mm + rotation
    gripper = np.array([0.35], dtype=np.float32)

    frame = np.full((image_size, image_size, 3), tick % 255, dtype=np.uint8)
    images = {
        "low": frame,
        "side": np.roll(frame, shift=1, axis=1),
        "wrist": np.roll(frame, shift=2, axis=0),
    }

    return build_policy_payload(
        joints=joints,
        pose=pose,
        gripper=gripper,
        images=images,
        ensemble=ensemble,
    )


def model_client_step(client: MyClient, payload: dict[str, Any], resolution: int) -> np.ndarray:
    actions: dict[str, Any] = client.step(payload)
    return extract_action_targets(actions, resolution=resolution)


def spec(tree: dict[str, Any]):
    def info(x):
        if isinstance(x, np.ndarray):
            return f"ndarray{tuple(x.shape)} dtype={x.dtype}"
        if isinstance(x, bool):
            return x
        else:
            return type(x)

    return jax.tree.map(lambda x: info(x), tree)


def run(cfg: Config) -> None:
    client = MyClient(host=cfg.host, port=cfg.port)
    client.reset()

    period = 1.0 / max(cfg.hz, 1.0)
    tick = 0

    print(f"Looping model client calls on {cfg.host}:{cfg.port} at {cfg.hz:.2f} Hz")
    print("Press Ctrl+C to stop.")

    while cfg.steps == 0 or tick < cfg.steps:
        payload = dummy_payload(cfg.image_size, cfg.ensemble, tick)
        print(spec(payload))
        targets = model_client_step(client, payload, cfg.resolution)
        print(f"step={tick:04d} targets.shape={targets.shape}")
        tick += 1
        time.sleep(period)


def main(cfg: Config) -> None:
    run(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))
