from __future__ import annotations
import rclpy
from dataclasses import dataclass
import tyro
from typing import Optional, Tuple, List
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped

class TwistToTwistStamped(Node):
    def __init__(self, cfg: Config):
        super().__init__('twist_to_twiststamped')
        self.cfg = cfg
        self.sub = self.create_subscription(Twist, cfg.sub, self.cb, 10)
        self.pub = self.create_publisher(TwistStamped, cfg.pub, 10)

    def cb(self, msg):
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.twist = msg
        self.pub.publish(out)

@dataclass
class Config:
    sub: str # ie "/cmd_vel"
    pub: str # ie "/cmd_vel_stamped"

def main(cfg: Config) -> None:
    rclpy.init()
    try:
        node = TwistToTwistStamped(cfg)
        rclpy.spin(node)
    finally:
        # shutdown might already be called
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main(tyro.cli(Config))

