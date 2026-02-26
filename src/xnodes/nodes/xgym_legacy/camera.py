from __future__ import annotations

from dataclasses import dataclass

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


class Camera(Node):
    """Publishes Camera state"""

    def __init__(self, name, idx, size=224):
        name = name if name else f"_{idx}"
        super().__init__(f"cam{idx}_{name}")

        # Retrieve `data_dir` from ROS parameters (passed from ros_sandbox.py)
        # self.declare_parameter("data_dir", "~/data")  # Default if not set
        # self.data_dir = ( self.get_parameter("data_dir").get_parameter_value().string_value)
        # self.get_logger().info(f"Using data directory: {self.data_dir}")

        self.name = name
        self.idx = idx
        self.size = size

        self.cap = cv2.VideoCapture(self.idx)
        # try 60fps... logitech c930e cant do it thought
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.bridge = CvBridge()

        self.hz = 60
        self.qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(CompressedImage, f"/xgym/camera/{self.name}", self.qos)
        self.timer = self.create_timer(1 / self.hz, self.publish_camera_images)
        self.i = 0

        self.get_logger().info("Camera Node Initialized.")

    def publish_camera_images(self):
        """Publishes images from all cameras."""

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # center crop self.size
            _h, _w = frame.shape[:2]
            # dont crop, cuz that cuts field of view
            # frame = cv2.resize(cu.square(frame), (self.size, self.size))
            frame = cv2.resize(frame, (self.size, self.size))

            # msg = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
            msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format="jpg")
            self.pub.publish(msg)
        else:
            self.get_logger().warning(f"Camera {self.name} failed to read frame")

        # if self.i % self.hz == 0:
        # info = f"Published image, shape={frame.shape}" if ret else "Failed publish"
        # self.get_logger().info(info)
        self.i += 1


@dataclass
class CameraConfig:
    name: str = ""  # camera name
    idx: int = 0  # camera device id


def main(cfg: CameraConfig):
    args = None
    rclpy.init(args=args)
    node = Camera(name=cfg.name, idx=cfg.idx)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Environment Node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
