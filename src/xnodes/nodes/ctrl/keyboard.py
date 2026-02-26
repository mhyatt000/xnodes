from __future__ import annotations

from dataclasses import dataclass

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
import tyro

from xnodes.client.keyboard import KeyboardPolicy


@dataclass
class Config:
    topic: str = "/ctrl/twist"  # output Twist topic
    step: float = 0.25  # linear velocity per key press
    poll_hz: float = 60.0  # keyboard polling rate


class KeyboardNode(Node):
    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__("keyboard_ctrl")

        if cfg is None:
            cfg = Config(
                topic=str(self.declare_parameter("topic", "/ctrl/twist").value),
                step=float(self.declare_parameter("step", 0.25).value),
                poll_hz=float(self.declare_parameter("poll_hz", 60.0).value),
            )
        self.cfg = cfg

        self.policy = KeyboardPolicy()
        if not self.policy.enabled:
            self.get_logger().warn("stdin is not a TTY. Keyboard input is disabled.")
            return

        self.get_logger().info("Keyboard active: arrows(up/down/left/right) + w/s. Press q to quit.")

        self.pub = self.create_publisher(Twist, self.cfg.topic, 10)
        period = 1.0 / max(self.cfg.poll_hz, 1.0)
        self.timer = self.create_timer(period, self._tick)

    def _tick(self) -> None:
        pressed = self.policy.step()
        if "q" in pressed:
            self.get_logger().info("Quit requested from keyboard.")
            rclpy.shutdown()
            return

        for key in pressed:
            msg = self._key_to_twist(key)
            if msg is None:
                continue
            self.pub.publish(msg)

    def _key_to_twist(self, key: str) -> Twist | None:
        msg = Twist()
        step = self.cfg.step

        if key in ("up", "w"):
            msg.linear.z = step
        elif key in ("down", "s"):
            msg.linear.z = -step
        elif key == "left":
            msg.linear.y = step
        elif key == "right":
            msg.linear.y = -step
        else:
            return None

        return msg

    def destroy_node(self) -> bool:
        self.policy.close()
        return super().destroy_node()


def run(cfg: Config | None = None) -> None:
    rclpy.init()
    node = KeyboardNode(cfg=cfg)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(cfg: Config) -> None:
    run(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))
