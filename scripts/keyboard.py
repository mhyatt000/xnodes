#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import tyro

try:
    from xnodes.client.keyboard import KeyboardPolicy
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from xnodes.client.keyboard import KeyboardPolicy


@dataclass
class Config:
    step: float = 0.25  # linear velocity per key press
    poll_hz: float = 60.0  # keyboard polling rate


def key_to_twist(key: str, step: float) -> tuple[float, float] | None:
    if key in ("up", "w"):
        return 0.0, step
    if key in ("down", "s"):
        return 0.0, -step
    if key == "left":
        return step, 0.0
    if key == "right":
        return -step, 0.0
    return None


def run(cfg: Config) -> None:
    policy = KeyboardPolicy()
    if not policy.enabled:
        raise RuntimeError("stdin is not a TTY. Run in an interactive terminal.")

    print("Keyboard active: arrows(up/down/left/right) + w/s. Press q to quit.")
    print("Output format: key=<name> linear.y=<value> linear.z=<value>")
    period = 1.0 / max(cfg.poll_hz, 1.0)

    try:
        while True:
            pressed = policy.step()
            if "q" in pressed:
                print("Quit requested from keyboard.")
                return

            for key in pressed:
                twist = key_to_twist(key, cfg.step)
                if twist is None:
                    continue
                lin_y, lin_z = twist
                print(f"key={key:>5s} linear.y={lin_y:+.3f} linear.z={lin_z:+.3f}")

            time.sleep(period)
    finally:
        policy.close()


def main(cfg: Config) -> None:
    run(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))
