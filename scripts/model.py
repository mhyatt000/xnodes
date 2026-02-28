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

from xnodes.core.model_client import extract_action_targets


@dataclass
class Config:
    host: str = "localhost"  # model server host
    port: int = 8000  # model server port
    hz: float = 5.0  # loop frequency
    image_size: int = 128  # square dummy image size
    proprio_dim: int = 14  # proprio vector length (pose + joints + gripper)
    resolution: int = 1  # action downsample factor
    ensemble: bool = False  # request ensembled policy output
    steps: int = 0  # 0 means run forever


def dummy_observation(spec: Any) -> Any:
    return jax.tree.map(
        lambda shape: np.random.rand(*shape).astype(np.float32), spec, is_leaf=lambda x: isinstance(x, tuple)
    )


def dummy_payload(spec: Any, ensemble: bool) -> dict[str, Any]:
    return {"observation": dummy_observation(spec), "ensemble": ensemble}


def describe(tree: Any) -> Any:
    def info(x):
        if isinstance(x, np.ndarray):
            return f"ndarray{tuple(x.shape)} dtype={x.dtype}"
        if isinstance(x, bool):
            return x
        return type(x)

    return jax.tree.map(info, tree)


class MyClient(Client):
    pass


def run(cfg: Config) -> None:
    client = MyClient(host=cfg.host, port=cfg.port)

    s = cfg.image_size
    obs_spec = {
        "image_left_wrist": (1, 1, s, s, 3),
        "image_primary": (1, 1, s, s, 3),
        "image_side": (1, 1, s, s, 3),
        "proprio_single_arm": (1, 1, cfg.proprio_dim),
    }
    client.reset(dummy_payload(obs_spec, cfg.ensemble))

    period = 1.0 / max(cfg.hz, 1.0)
    tick = 0

    print(f"Looping model client calls on {cfg.host}:{cfg.port} at {cfg.hz:.2f} Hz")
    print("Press Ctrl+C to stop.")

    while cfg.steps == 0 or tick < cfg.steps:
        payload = dummy_payload(obs_spec, cfg.ensemble)
        print(describe(payload))
        result: dict[str, Any] = client.step(payload)
        actions = extract_action_targets(result, resolution=cfg.resolution)

        print(f"step={tick:04d} actions.shape={actions.shape}")
        tick += 1
        time.sleep(period)


def main(cfg: Config) -> None:
    run(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))
