from __future__ import annotations

import rclpy
from rich.pretty import pprint
from std_msgs.msg import Float32MultiArray

from .base import Base
from .controllers import SpaceMouseController


class SpaceMouse(Base):
    """
    Reads input from the SpaceMouse and publishes to /robot_commands.
    """

    def __init__(self):
        super().__init__("controller_node")

        self.publisher = self.create_publisher(Float32MultiArray, "/robot_commands", 10)
        self.controller = SpaceMouseController()

        self.hz = 200
        self.timer = self.create_timer(1 / self.hz, self.publish_command)

        self.get_logger().info("Controller Node Initialized.")

    def publish_command(self):
        """Reads SpaceMouse input and publishes it."""

        action = self.controller.read()
        pprint(action)
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.publisher.publish(msg)
        # self.get_logger().info(f"Published SpaceMouse command: {action.round(2)}")


def main(args=None):
    rclpy.init(args=args)
    node = SpaceMouse()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Controller Node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
