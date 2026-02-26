#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class RandomHand(Node):
    def __init__(self):
        super().__init__("random_hand")
        self.pub = self.create_publisher(Float32MultiArray, "/hand/points", 10)
        self.timer = self.create_timer(0.1, self.publish_points)  # 10 Hz

    def publish_points(self):
        print("Publishing random hand points")
        pts = np.random.uniform(-0.2, 0.2, (21, 3)).astype(np.float32)
        msg = Float32MultiArray()
        msg.data = pts.flatten().tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = RandomHand()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
