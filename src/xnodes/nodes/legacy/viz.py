from __future__ import annotations

from functools import partial

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from .base import Base


class _FastImageViewer(Base):
    def __init__(self):
        print("init")
        super().__init__("fast_image_viewer")

        self.hz = 30
        self.set_period()
        self.build_cam_subs()
        self.timer = self.create_timer(1 / self.hz, self.show)

        print(self.subs)

    def show(self):
        # rgb to bgr for viewing
        print(self.data.keys())

        """
        for k, v in self.data.items():
            print(k)
            if "cam" in k and (v is not None):
                v = self.annotate(v, k)
                print(v.shape)
                self.data[k] = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)

        frame = np.concatenate(
            [v for k, v in self.data.items() if v is not None], axis=1
        )
        """
        frame = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)

        cv2.imshow("Camera", frame)
        print("show")
        cv2.waitKey(1)

    def annotate(self, frame, key):
        f2 = frame.copy()
        cv2.putText(
            f2,
            key,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return f2.astype(np.uint8)


class FastImageViewer(Node):
    def __init__(self):
        super().__init__("fast_image_viewer")

        self.qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.cams = self.list_camera_topics()
        self.data = dict.fromkeys(self.cams)

        self.subs = {
            k: self.create_subscription(CompressedImage, k, partial(self.set_image, key=k), self.qos) for k in self.cams
        }

        self.bridge = CvBridge()
        self.get_logger().info("Fast Image Viewer Initialized.")

        self.hz = 10
        self.timer = self.create_timer(1 / self.hz, self.show)

    def list_camera_topics(self):
        topics = self.get_topic_names_and_types()
        cams = [t for t, types in topics if "camera" in t]
        return cams

    def set_image(self, msg, key):
        self.get_logger().info(f"Received image from {key}")

        # frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # if it is compressed image,

        frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        """
        frame = cv2.putText(
            frame,
            key,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        """

        # back to bgr since they are published in rgb
        self.data[key] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def show(self):
        self.get_logger().info("show")
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.imshow("Viewer", img)

        cv2.waitKey(1)

        return

        if all(v is None for v in self.data.values()):
            print(self.data)
            return
        else:
            print("showing")
            print({k: (v.shape, v.dtype) for k, v in self.data.items() if v is not None})
            return

        """
        # rgb to bgr for viewing
        for k, v in self.data.items():
            if "cam" in k and (not v is None):
                self.data[k] = self.annotate(v, k)
                self.data[k] = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)
        """

        frame = np.concatenate([v for k, v in self.data.items() if v is not None], axis=1)

        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

    def annotate(frame, key):
        frame = cv2.putText(
            frame,
            key,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return frame


def main(args=None):
    rclpy.init(args=args)
    node = FastImageViewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
