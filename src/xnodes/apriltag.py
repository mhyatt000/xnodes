"""ROS2 AprilTag node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

try:
    from cv_bridge import CvBridge
except ImportError as exc:  # pragma: no cover - ROS env should provide cv_bridge
    raise RuntimeError("cv_bridge is required for AprilTagNode") from exc


R_OPT2CAMLINK = np.array([[0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0]], dtype=float)
T_OPT2CAMLINK = np.eye(4, dtype=float)
T_OPT2CAMLINK[:3, :3] = R_OPT2CAMLINK


def _rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def _inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Return quaternion (x, y, z, w)."""

    q = np.empty(4, dtype=float)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
        return q
    idx = int(np.argmax(np.diag(R)))
    if idx == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[3] = (R[2, 1] - R[1, 2]) / s
        q[0] = 0.25 * s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = (R[0, 2] + R[2, 0]) / s
    elif idx == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[3] = (R[0, 2] - R[2, 0]) / s
        q[0] = (R[0, 1] + R[1, 0]) / s
        q[1] = 0.25 * s
        q[2] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[3] = (R[1, 0] - R[0, 1]) / s
        q[0] = (R[0, 2] + R[2, 0]) / s
        q[1] = (R[1, 2] + R[2, 1]) / s
        q[2] = 0.25 * s
    return q


@dataclass
class AprilTagDetection:
    """AprilTag detection result."""

    tag_id: int  # detected id
    corners: np.ndarray  # TL, TR, BR, BL image pixels
    T_cam_tag: np.ndarray  # tag pose in optical frame
    T_tag_cam: np.ndarray  # optical pose in tag frame
    T_camlink_tag: Optional[np.ndarray] = None
    T_tag_camlink: Optional[np.ndarray] = None


class AprilTagDetector:
    """Detect AprilTags from images."""

    def __init__(self, *, family: str, tag_size: float, refine_corners: bool, to_camlink: bool) -> None:
        self.family = family
        self.tag_size = tag_size
        self.refine_corners = refine_corners
        self.to_camlink = to_camlink
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self._detector = self._build_detector(family)

    @staticmethod
    def _build_detector(family: str) -> cv.aruco.ArucoDetector:
        if not hasattr(cv.aruco, family):
            raise ValueError(f"OpenCV aruco missing {family} (need opencv-contrib ≥ 4.7)")
        dictionary = getattr(cv.aruco, family)
        aruco_dict = cv.aruco.getPredefinedDictionary(dictionary)
        params = cv.aruco.DetectorParameters()
        return cv.aruco.ArucoDetector(aruco_dict, params)

    def set_camera_info(self, msg: CameraInfo) -> None:
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.dist = np.array(msg.d, dtype=float)

    def detect(self, image: np.ndarray, desired_id: int) -> AprilTagDetection:
        if self.K is None or self.dist is None:
            raise RuntimeError("camera intrinsics not available")

        corners, ids, _ = self._detector.detectMarkers(image)
        if ids is None or len(ids) == 0:
            raise RuntimeError("no AprilTags detected")

        ids_flat = ids.flatten()
        if desired_id in ids_flat:
            idx = int(np.where(ids_flat == desired_id)[0][0])
        else:
            idx = 0

        pts = corners[idx].reshape(4, 2).astype(np.float32)
        if self.refine_corners:
            win = (5, 5)
            zero_zone = (-1, -1)
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            cv.cornerSubPix(image, pts, win, zero_zone, term)

        s = self.tag_size / 2.0
        obj = np.array([[-s, s, 0.0],
                        [s, s, 0.0],
                        [s, -s, 0.0],
                        [-s, -s, 0.0]], dtype=float)

        ok, rvec, tvec = cv.solvePnP(obj, pts, self.K, self.dist, flags=cv.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP failed")

        T_cam_tag = _rvec_tvec_to_T(rvec, tvec)
        T_tag_cam = _inv_T(T_cam_tag)

        if self.to_camlink:
            T_camlink_tag = T_OPT2CAMLINK @ T_cam_tag
            T_tag_camlink = _inv_T(T_camlink_tag)
        else:
            T_camlink_tag = None
            T_tag_camlink = None

        return AprilTagDetection(tag_id=int(ids[idx][0]),
                                 corners=pts,
                                 T_cam_tag=T_cam_tag,
                                 T_tag_cam=T_tag_cam,
                                 T_camlink_tag=T_camlink_tag,
                                 T_tag_camlink=T_tag_camlink)


class AprilTagNode(Node):
    """ROS2 node that publishes AprilTag poses."""

    def __init__(self) -> None:
        super().__init__("apriltag_node")
        self.declare_parameter("tag_id", 0)
        self.declare_parameter("tag_size", 0.2286)
        self.declare_parameter("family", "DICT_APRILTAG_36h11")
        self.declare_parameter("refine_corners", True)
        self.declare_parameter("to_camlink", False)
        self.declare_parameter("camera_frame", "camera_optical")

        self._detector = AprilTagDetector(
            family=self.get_parameter("family").get_parameter_value().string_value,
            tag_size=self.get_parameter("tag_size").get_parameter_value().double_value,
            refine_corners=self.get_parameter("refine_corners").get_parameter_value().bool_value,
            to_camlink=self.get_parameter("to_camlink").get_parameter_value().bool_value,
        )

        self._desired_id = self.get_parameter("tag_id").get_parameter_value().integer_value
        self._camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value

        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self._camera_sub = self.create_subscription(CameraInfo, "camera_info", self._camera_info_cb, qos)
        self._image_sub = self.create_subscription(Image, "image", self._image_cb, qos)
        self._pose_pub = self.create_publisher(PoseStamped, "apriltag_pose", qos)
        self._pose_camlink_pub = self.create_publisher(PoseStamped, "apriltag_pose_camlink", qos)
        self._bridge = CvBridge()

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self._detector.set_camera_info(msg)
        if msg.header.frame_id:
            self._camera_frame = msg.header.frame_id

    def _image_cb(self, msg: Image) -> None:
        if self._detector.K is None:
            self.get_logger().debug("waiting for camera info")
            return

        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

        try:
            detection = self._detector.detect(image, self._desired_id)
        except RuntimeError as err:
            self.get_logger().debug(str(err))
            return

        pose_msg = self._to_pose(msg.header.stamp, self._camera_frame, detection.T_cam_tag)
        self._pose_pub.publish(pose_msg)

        if detection.T_camlink_tag is not None:
            pose_camlink = self._to_pose(msg.header.stamp, "camera_link", detection.T_camlink_tag)
            self._pose_camlink_pub.publish(pose_camlink)

    def _to_pose(self, stamp: Time, frame: str, T: np.ndarray) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = frame
        pose.pose.position.x = float(T[0, 3])
        pose.pose.position.y = float(T[1, 3])
        pose.pose.position.z = float(T[2, 3])
        qx, qy, qz, qw = _rotation_to_quaternion(T[:3, :3])
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        return pose


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = AprilTagNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


__all__ = ["AprilTagDetection", "AprilTagDetector", "AprilTagNode", "main"]


if __name__ == "__main__":  # pragma: no cover
    main()

