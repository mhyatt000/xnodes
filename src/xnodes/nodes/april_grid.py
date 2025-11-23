from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from cv_bridge import CvBridge
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

from xnodes.april.det import detect_aprilgrid, make_apriltag_detector, project_points


def mat2quat(T):
    # T: 4x4
    return R.from_matrix(T[:3, :3]).as_quat()  # returns [x, y, z, w]


d = Path(__file__).parent.resolve()


@dataclass
class Config:
    sub: str  # subscription topic
    path: Path = d / Path("april_6x6_80x80cm.yaml")  # Path to aprilgrid YAML
    show: bool = False  # Whether to show debug image
    invert: bool = False  # Whether to invert transform
    id: int | None = None  # If set, only publish this tag ID

    def __post_init__(self):
        if self.invert:
            assert self.id is not None, "If invert is set, id must be set"


def _tag_center_T_grid(row: int, col: int, cfg: AprilGridConfig) -> np.ndarray:
    """
    Compute the transform from grid frame to tag frame for a tag at (row, col).
    The grid frame is centered at the center of the grid, with +X to the right,
    +Y up, and +Z out of the grid plane.
    The tag frame is centered at the center of the tag, with +X to the right,
    +Y up, and +Z out of the tag plane.
    """

    spacing = cfg.tagSize * (1.0 + cfg.tagSpacing)
    cx = (col - (cfg.tagCols - 1) / 2.0) * spacing
    cy = (row - (cfg.tagRows - 1) / 2.0) * spacing
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.array([cx, cy, 0.0], dtype=np.float32)
    return T  # R = I


class AprilGridDetector(Node):
    def __init__(
        self,
        cfg: Config | None = None,
    ):
        super().__init__("aprilgrid_shim")

        # --- grid config: from arg if present, else from ROS param ---
        if cfg is not None:
            self.cfg = cfg
            april = AprilGridConfig.load(str(cfg.path))
            self.april = april
        else:
            self.declare_parameter("grid_yaml", "")
            grid_yaml = self.get_parameter("grid_yaml").get_parameter_value().string_value
            if not grid_yaml:
                raise RuntimeError("grid_yaml parameter must be set when no cfg is provided")
            target = CalibrationTarget.load(grid_yaml)
            if not isinstance(target, AprilGridConfig):
                raise TypeError(f"Expected AprilGridConfig, got {type(target)}")
            self.april = target

        # --- image topic: from arg if present, else from ROS param ---
        if cfg is not None:
            sub = cfg.sub
        else:
            self.declare_parameter("sub", "")
            sub = self.get_parameter("sub").get_parameter_value().string_value
            if not sub:
                raise RuntimeError("sub parameter must be set when no sub arg is provided")

        # detector (reuse instance)
        self.detector: Detector = make_apriltag_detector(families=april.families)
        self.april = self.april

        self.bridge = CvBridge()
        self.name = sub.replace("/image_raw", "").split("/")[-1]

        self.sub = self.create_subscription(
            Image,
            sub,
            self.cb_image,
            qos_profile_sensor_data,
        )

        self.bst = TransformBroadcaster(self)
        self.get_logger().info(f"aprilgrid shim listening on {sub} for grid {self.april.tagRows}x{self.april.tagCols}")

    def cb_image(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        img = np.asarray(img, dtype=np.uint8)

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
            self.params = [msg.width, msg.height, msg.width / 2.0, msg.height / 2.0]

        res = detect_aprilgrid(img, self.april, self.detector, self.params)

        if res is None:
            self.get_logger().debug("no board detected")
            if self.cfg.show:
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.imshow("detections", img_color)
                cv2.waitKey(1)
            return

        pts2d, _pts3d, tag_obs, detections = res
        self.get_logger().info(f"detected {pts2d.shape[0] // 4} tags ({pts2d.shape[0]} corners)")

        if self.cfg.show:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for det in detections:
                for i in range(4):
                    pt1 = (int(det.corners[i][0]), int(det.corners[i][1]))
                    pt2 = (int(det.corners[(i + 1) % 4][0]), int(det.corners[(i + 1) % 4][1]))
                    cv2.line(img_color, pt1, pt2, (0, 255, 0), 1)
                c_x = int(det.center[0])
                c_y = int(det.center[1])
                cv2.putText(
                    img_color, str(det.tag_id), (c_x - 10, c_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
                )
            # for p in pts2d:
            # cv2.circle(img_color, (int(p[0]), int(p[1])), 3, (255, 0, 255), -1)

        # [OLD CONVENTION]
        # Solve for board ("world") pose in camera frame.
        # ok, rvec, tvec = cv2.solvePnP(pts3d, pts2d, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        # if not ok:
        # self.get_logger().warn("solvePnP failed")
        # return

        # now rvec tvec should be mean of the april detections pose_R and post_t
        """
        rvecs = []
        tvecs = []
        for det in detections:
            R_det = det.pose_R
            t_det = det.pose_t.flatten()
            rvec_det, _ = cv2.Rodrigues(R_det)
            rvecs.append(rvec_det.flatten())
            tvecs.append(t_det)
        rvec = np.mean(rvecs, axis=0).reshape((3, 1))
        tvec = np.mean(tvecs, axis=0).reshape((3, 1))
        """

        """
        R_cam_grid, _ = cv2.Rodrigues(rvec)
        T_cam_grid = np.eye(4, dtype=np.float32)
        T_cam_grid[:3, :3] = R_cam_grid
        T_cam_grid[:3, 3] = tvec.flatten()

        T_grid_cam = np.linalg.inv(T_cam_grid)
        """

        # print(T_cam_grid)

        if self.cfg.id:
            tag_obs = [obs for obs in tag_obs if obs.tag_id == self.cfg.id]
            detections = [d for d in detections if d.tag_id == self.cfg.id]
            assert len(tag_obs) == len(detections) == 1, "Mismatch in filtered tag observations and detections"
            print(len(tag_obs), len(detections))

        # For each visible tag, compute T_cam_tag and publish TF
        Ts = []
        for obs, d in zip(tag_obs, detections):
            print(d.tag_id)
            # d.pose_R, d.pose_t
            T_cam_tag = np.eye(4, dtype=np.float32)
            T_cam_tag[:3, :3] = d.pose_R
            T_cam_tag[:3, 3] = d.pose_t.flatten()

            # if self.cfg.invert:
            # T_cam_tag = np.linalg.inv(T_cam_tag)

            # T_grid_tag = _tag_center_T_grid(obs.row, obs.col, self.april)
            # T_cam_tag = T_cam_grid @ T_grid_tag

            T = TransformStamped()
            T.header.stamp = self.get_clock().now().to_msg()
            parent, child = self.name, f"tag_{obs.tag_id}"
            if obs.tag_id != self.cfg.id:
                print("Tag ID mismatch")
                continue
            parent, child = child, parent if self.cfg.invert else (parent, child)
            T.header.frame_id = parent
            T.child_frame_id = child

            T.transform.translation.x = float(T_cam_tag[0, 3])
            T.transform.translation.y = float(T_cam_tag[1, 3])
            T.transform.translation.z = float(T_cam_tag[2, 3])

            # print(T.transform.translation)

            q = mat2quat(T_cam_tag)
            T.transform.rotation.x = float(q[0])
            T.transform.rotation.y = float(q[1])
            T.transform.rotation.z = float(q[2])
            T.transform.rotation.w = float(q[3])

            Ts.append(T)
            self.bst.sendTransform(T)

        if self.cfg.show:
            # inverse of T_cam_grid
            # Draw axis for board iff all tags are visible
            if len(tag_obs) == self.april.tagRows * self.april.tagCols:
                axis_length = self.april.tagSize
                # cv2.drawFrameAxes(
                # img_color,
                # self.K,
                # self.dist,
                # rvec,
                # tvec,
                # axis_length,
                # )

            # in red, draw square around 3d projected to 2d for each pts3d
            # using project_points(K, pts3d)
            allp = []
            for obs in tag_obs:
                p3d = obs.corners3d  # (4,3)
                # convert from obj frame to camera frame
                # use T_cam_grid
                continue
                p3d_h = np.hstack((p3d, np.ones((4, 1), dtype=np.float32)))
                p3d = (T_cam_grid @ p3d_h.T).T[:, :3]  # (4,3)

                p2d_proj = project_points(self.K, p3d)

                if any(np.isnan(p2d_proj).flatten()):
                    continue

                for i in range(4):
                    pt1 = (int(p2d_proj[i][0]), int(p2d_proj[i][1]))
                    pt2 = (int(p2d_proj[(i + 1) % 4][0]), int(p2d_proj[(i + 1) % 4][1]))
                    cv2.circle(img_color, pt1, 3, (0, 0, 255), -1)
                    # text at corner
                    cv2.putText(
                        img_color,
                        f"{obs.tag_id}:{i}",
                        (pt1[0] + 5, pt1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )
                    allp.append(pt1)
                    # cv2.line(img_color, pt1, pt2, (0, 0, 255), 1)
                    # cv2.line(img_color,
                    # (int(p2d_proj[0][0]), int(p2d_proj[0][1])), (int(p2d_proj[2][0]),

        if self.cfg.show:
            cv2.imshow("detections", img_color)
            cv2.waitKey(1)


def run(args=None):
    rclpy.init(args=args)
    node = AprilGridDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def main(cfg: Config) -> Config:
    print(cfg)

    rclpy.init()
    node = AprilGridDetector(cfg=cfg)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return cfg


if __name__ == "__main__":
    main(tyro.cli(Config))
