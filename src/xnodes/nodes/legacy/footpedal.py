from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from evdev import ecodes, InputDevice
import rclpy
from std_msgs.msg import Bool, Int32MultiArray
import tyro

from .base import Base


class FootPedal(Base):
    def __init__(self, cfg: FootPedalConfig | None = None):
        super().__init__("foot_pedal")

        if cfg is None:
            cfg = FootPedalConfig(
                path=str(self.declare_parameter("path", FootPedalConfig.path).value),
                topic_a=self._topic_or_none(self.declare_parameter("topic_a", "").value),
                topic_b=self._topic_or_none(self.declare_parameter("topic_b", "").value),
                topic_c=self._topic_or_none(self.declare_parameter("topic_c", "").value),
            )
        self.cfg = cfg

        self.pub = self.create_publisher(Int32MultiArray, "/xgym/pedal", 10)
        topics = {
            ecodes.KEY_A: cfg.topic_a,
            ecodes.KEY_B: cfg.topic_b,
            ecodes.KEY_C: cfg.topic_c,
        }
        self.pubs = {
            code: self.create_publisher(Bool, topic, 10) for code, topic in topics.items() if topic is not None
        }

        self.pmap = {
            ecodes.KEY_A: 0,
            ecodes.KEY_B: 1,
            ecodes.KEY_C: 2,
        }
        self.value = [0, 0, 0]
        self.state = [False, False, False]

        self.get_logger().info(f"Opening foot pedal device at: {cfg.path}")
        self.device = InputDevice(cfg.path)
        self.device.grab()

        # Start a background thread to continuously read events
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        """
        Continuously reads events from the foot pedal device and publishes
        the legacy integer array plus optional Bool messages for pedal state.
        """

        for event in self.device.read_loop():
            time.sleep(0.02)  # Small delay to prevent CPU overuse and double tap
            if event.type == ecodes.EV_KEY and event.code in self.pmap:
                p = self.pmap[event.code]
                new = event.value
                pressed = event.value != 0

                if self.value[p] != new:
                    self.value[p] = new
                    self.pub.publish(Int32MultiArray(data=self.value))

                if self.state[p] != pressed:
                    self.state[p] = pressed
                    if pub := self.pubs.get(event.code):
                        pub.publish(Bool(data=pressed))
                    self.get_logger().info(f"Pedal {p} -> {self.describe(pressed)}")

    def describe(self, val):
        return "pressed" if val else "released"

    @staticmethod
    def _topic_or_none(value: object) -> str | None:
        if value is None:
            return None
        topic = str(value).strip()
        return topic or None


@dataclass
class FootPedalConfig:
    path: str = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
    topic_a: str | None = None
    topic_b: str | None = None
    topic_c: str | None = None


def main(cfg: FootPedalConfig):
    run(cfg)


def run(cfg: FootPedalConfig | None = None):
    rclpy.init(args=None)
    node = FootPedal(cfg=cfg)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(tyro.cli(FootPedalConfig))
