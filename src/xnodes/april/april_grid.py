from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from cv_bridge import CvBridge
from det import detect_aprilgrid, make_apriltag_detector
from geometry_msgs.msg import TransformStamped
import numpy as np
from pupil_apriltags import Detector
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from reader import AprilGridConfig, CalibrationTarget
from rich import print
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster
import tyro


def mat2quat(T):
    # T: 4x4
    return R.from_matrix(T[:3, :3]).as_quat()  # returns [x, y, z, w]


@dataclass
class Config:
    sub: str  # subscription topic
    path: Path = Path("april_6x6_80x80cm.yaml")  # Path to aprilgrid YAML


def _tag_center_T_grid(row: int, col: int, cfg: AprilGridConfig) -> np.ndarray:
    spacing = cfg.tagSize * (1.0 + cfg.tagSpacing)
    cx = (col - (cfg.tagCols - 1) / 2.0) * spacing
    cy = (row - (cfg.tagRows - 1) / 2.0) * spacing
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.array([cx, cy, 0.0], dtype=np.float32)
    return T  # R = I


class AprilGridDetector(Node):
    def __init__(
        self,
        cfg: AprilGridConfig | None = None,
        sub: str | None = None,
    ):
        super().__init__("aprilgrid_shim")

        # --- grid config: from arg if present, else from ROS param ---
        if cfg is not None:
            self.cfg = cfg
        else:
            self.declare_parameter("grid_yaml", "")
            grid_yaml = self.get_parameter("grid_yaml").get_parameter_value().string_value
            if not grid_yaml:
                raise RuntimeError("grid_yaml parameter must be set when no cfg is provided")
            target = CalibrationTarget.load(grid_yaml)
            if not isinstance(target, AprilGridConfig):
                raise TypeError(f"Expected AprilGridConfig, got {type(target)}")
            self.cfg = target

        # --- image topic: from arg if present, else from ROS param ---
        if sub is not None:
            sub = sub
        else:
            self.declare_parameter("sub", "")
            sub = self.get_parameter("sub").get_parameter_value().string_value
            if not sub:
                raise RuntimeError("sub parameter must be set when no sub arg is provided")

        # detector (reuse instance)
        self.detector: Detector = make_apriltag_detector(families=cfg.families)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            sub,
            self.cb_image,
            qos_profile_sensor_data,
        )

        self.bst = TransformBroadcaster(self)
        self.get_logger().info(f"aprilgrid shim listening on {sub} for grid {self.cfg.tagRows}x{self.cfg.tagCols}")

    def cb_image(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        img = np.asarray(img, dtype=np.uint8)

        res = detect_aprilgrid(img, self.cfg, self.detector)
        if res is None:
            self.get_logger().debug("no board detected")
            print(None)
            return

        pts2d, pts3d, tag_obs = res
        self.get_logger().info(f"detected {pts2d.shape[0] // 4} tags ({pts2d.shape[0]} corners)")

        if not hasattr(self, "K"):
            self.K = np.array(
                [
                    [msg.width, 0.0, msg.width / 2.0],
                    [0.0, msg.height, msg.height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            self.dist = np.zeros((5, 1), dtype=np.float32)  # assume no distortion

        # Solve for board ("world") pose in camera frame.
        ok, rvec, tvec = cv2.solvePnP(pts3d, pts2d, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            self.get_logger().warn("solvePnP failed")
            return

        R_cam_grid, _ = cv2.Rodrigues(rvec)
        T_cam_grid = np.eye(4, dtype=np.float32)
        T_cam_grid[:3, :3] = R_cam_grid
        T_cam_grid[:3, 3] = tvec.flatten()

        print(T_cam_grid)

        # For each visible tag, compute T_cam_tag and publish TF
        for obs in tag_obs:
            T_grid_tag = _tag_center_T_grid(obs.row, obs.col, self.cfg)
            T_cam_tag = T_cam_grid @ T_grid_tag

            T = TransformStamped()
            T.header.stamp = self.get_clock().now().to_msg()
            T.header.frame_id = "world"  # self.camera_frame        # parent
            T.child_frame_id = f"tag_{obs.tag_id}"  # child

            T.transform.translation.x = float(T_cam_tag[0, 3])
            T.transform.translation.y = float(T_cam_tag[1, 3])
            T.transform.translation.z = float(T_cam_tag[2, 3])

            print(T.transform.translation)

            q = mat2quat(T_cam_tag)
            T.transform.rotation.x = float(q[0])
            T.transform.rotation.y = float(q[1])
            T.transform.rotation.z = float(q[2])
            T.transform.rotation.w = float(q[3])

            self.bst.sendTransform(T)


def run(args=None):
    rclpy.init(args=args)
    node = AprilGridDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def main(cfg: Config) -> Config:
    print(cfg)

    april = AprilGridConfig.load(str(cfg.path))
    print(april)

    rclpy.init()
    node = AprilGridDetector(cfg=april, sub=cfg.sub)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return cfg


if __name__ == "__main__":
    main(tyro.cli(Config))
