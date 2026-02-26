from __future__ import annotations

from dataclasses import dataclass
import threading

from evdev import ecodes, InputDevice
import numpy as np
import rclpy
from std_msgs.msg import Int32MultiArray
import tyro

from .base import Base


class FootPedal(Base):
    def __init__(self, path: str = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"):
        super().__init__("foot_pedal")
        self.pub = self.create_publisher(Int32MultiArray, "/xgym/pedal", 10)

        self.pmap = {
            ecodes.KEY_A: 0,
            ecodes.KEY_B: 1,
            ecodes.KEY_C: 2,
        }

        self.get_logger().info(f"Opening foot pedal device at: {path}")
        self.device = InputDevice(path)
        self.device.grab()

        # Start a background thread to continuously read events
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()
        self.value = np.array([0, 0, 0])

    def read(self):
        """
        Continuously reads events from the foot pedal device and publishes
        a 3-element array whenever a pedal's state changes.
        """

        for event in self.device.read_loop():
            if event.type == ecodes.EV_KEY and event.code in self.pmap:
                p = self.pmap[event.code]
                new = event.value  # 0=release, 1=press, 2=hold/repeat

                if changed := (self.value[p] != new):
                    self.value[p] = new
                    msg = Int32MultiArray(data=self.value)
                    self.pub.publish(msg)
                    self.get_logger().info(f"{np.array(msg.data)}")
                    self.get_logger().info(f"Pedal {p} -> {self.describe(new)}")

    def describe(self, val):
        return {0: "released", 1: "pressed", 2: "held"}.get(val, f"unknown({val})")


@dataclass
class FootPedalConfig:
    path: str = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"


def main(path: str):
    rclpy.init(args=None)
    node = FootPedal(path=path)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(tyro.cli(FootPedalConfig).path)
