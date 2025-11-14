from __future__ import annotations
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import tyro

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray, Pose
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from rich import print


R_OPT2CAMLINK = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=float
)
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

    def __init__(self, cfg: AprilConfig):
        self.cfg = cfg

        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self._detector = self._build_detector(self.cfg.family)

    @staticmethod
    def _build_detector(family: str) -> cv.aruco.ArucoDetector:
        if not hasattr(cv.aruco, family):
            raise ValueError(
                f"OpenCV aruco missing {family} (need opencv-contrib ≥ 4.7)"
            )
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

        print(image.shape)

        corners, ids, _ = self._detector.detectMarkers(image)
        if ids is None or len(ids) == 0:
            raise RuntimeError("no AprilTags detected")

        rvecs, tvecs, _ = cv.aruco.estimatePoseSingleMarkers(
            corners, self.cfg.tag_size_m, self.K, self.dist
        )
        print(tvecs)
        return tvecs
        print(corners)

        ids_flat = ids.flatten()
        if desired_id in ids_flat:
            idx = int(np.where(ids_flat == desired_id)[0][0])
        else:
            idx = 0

        pts = corners[idx].reshape(4, 2).astype(np.float32)
        if self.cfg.refine_corners:
            win = (5, 5)
            zero_zone = (-1, -1)
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            cv.cornerSubPix(image, pts, win, zero_zone, term)

        print(self.cfg.tag_size_m)
        s = self.cfg.tag_size_m / 2.0
        obj = np.array(
            [[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=float
        )

        ok, rvec, tvec = cv.solvePnP(
            obj, pts, self.K, self.dist, flags=cv.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise RuntimeError("solvePnP failed")

        T_cam_tag = _rvec_tvec_to_T(rvec, tvec)
        T_tag_cam = _inv_T(T_cam_tag)

        if self.cfg.to_camlink:
            T_camlink_tag = T_OPT2CAMLINK @ T_cam_tag
            T_tag_camlink = _inv_T(T_camlink_tag)
        else:
            T_camlink_tag = None
            T_tag_camlink = None

        return AprilTagDetection(
            tag_id=int(ids[idx][0]),
            corners=pts,
            T_cam_tag=T_cam_tag,
            T_tag_cam=T_tag_cam,
            T_camlink_tag=T_camlink_tag,
            T_tag_camlink=T_tag_camlink,
        )


class AprilTagNode(Node):
    """ROS2 node that publishes AprilTag poses."""

    def __init__(self, cfg: AprilConfig):
        super().__init__("apriltag_node")
        self.cfg = cfg
        self._detector = AprilTagDetector(self.cfg)

        def topic(*p):
            return str(Path(*Path(cfg.topic).parts, *p).absolute())

        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self._camera_sub = self.create_subscription(
            CameraInfo, topic("camera_info"), self._camera_info_cb, qos
        )
        self._image_sub = self.create_subscription(
            Image, topic("image_raw"), self._image_cb, qos
        )

        def topic(*p):
            return str(Path("april", *Path(cfg.topic).parts, *p).absolute())

        self.pub_pose = self.create_publisher(PoseArray, topic("tag_pose"), qos)
        # self._pose_camlink_pub = self.create_publisher(PoseStamped, topic("apriltag_pose_camlink"), qos)
        self.pub_mark = self.create_publisher(MarkerArray, topic("tag_markers"), 10)

        self._bridge = CvBridge()

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self._detector.set_camera_info(msg)
        if msg.header.frame_id:
            self.cfg.camera_frame = msg.header.frame_id

    def _image_cb(self, msg: Image) -> None:
        if self._detector.K is None:
            self.get_logger().debug("waiting for camera info")
            return

        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

        try:
            detection = self._detector.detect(image, self.cfg.tag_id)
        except RuntimeError as err:
            self.get_logger().debug(str(err))
            return

        pose_msg = self._to_pose(
            msg.header.stamp, self.cfg.camera_frame, detection.corners[0]
        )
        # pose_msg = self._to_pose(msg.header.stamp, self.cfg.camera_frame, detection.T_cam_tag)
        print(pose_msg.poses)
        self.pub_pose.publish(pose_msg)
        return

        # markers
        ms = MarkerArray()
        # clr = Marker()
        # clr.action = Marker.DELETEALL
        # ms.markers.append(clr)

        m = Marker()
        m.header = Header(stamp=msg.header.stamp, frame_id=self.cfg.camera_frame)
        print(m.header)
        m.ns = "apriltag"
        m.id = 0
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.1
        m.colors = [ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)] + [
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5) for _ in range(5)
        ]
        m.lifetime.sec = 0
        pose = pose_msg.pose
        print(pose)
        corners = detection.corners
        m.points = [
            Point(x=pose.position.x, y=pose.position.y, z=pose.position.z),
            *[
                Point(
                    x=pose.position.x + (corner[0] - self._detector.K[0, 2]) * 0.01,
                    y=pose.position.y + (corner[1] - self._detector.K[1, 2]) * 0.01,
                    z=pose.position.z,
                )
                for corner in corners
            ],
        ]

        print(pose.position)
        ms.markers.append(m)

        self.pub_mark.publish(ms)

        # if detection.T_camlink_tag is not None:
        # pose_camlink = self._to_pose(msg.header.stamp, "camera_link", detection.T_camlink_tag)
        # self._pose_camlink_pub.publish(pose_camlink)

    def _to_pose(self, stamp: Time, frame: str, T: np.ndarray) -> PoseStamped:
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = frame

        p = Pose()
        p.position.x = float(T[0, 3])
        p.position.y = float(T[1, 3])
        p.position.z = float(T[2, 3])
        qx, qy, qz, qw = _rotation_to_quaternion(T[:3, :3])
        p.orientation.x = float(qx)
        p.orientation.y = float(qy)
        p.orientation.z = float(qz)
        p.orientation.w = float(qw)

        msg.poses.append(p)
        return msg


@dataclass
class AprilConfig:
    topic: str  # base topic for camera info and images ie: /video0
    tag_id: int = 0
    size: float = 9.0  # in april tag size inches
    family: str = "DICT_APRILTAG_36h11"

    refine_corners: bool = True
    to_camlink: bool = False
    camera_frame: str = "camera_optical"

    @property
    def tag_size_m(self) -> None:
        """Convert size from inches to meters."""
        return self.size * 0.0254


def main(cfg: AprilConfig):
    rclpy.init()
    node = AprilTagNode(cfg)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(tyro.cli(AprilConfig))
