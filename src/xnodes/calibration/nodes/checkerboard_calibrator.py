from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger


@dataclass
class CalibrationResult:
    matrix: np.ndarray
    distortion: np.ndarray
    rvecs: Sequence[np.ndarray]
    tvecs: Sequence[np.ndarray]


class CheckerboardCalibratorNode(Node):
    """Calibrates a monocular camera from checkerboard image streams."""

    def __init__(self, node_name: str = "checkerboard_calibrator", **node_kwargs: object) -> None:
        super().__init__(node_name, **node_kwargs)
        cols = self.declare_parameter("checkerboard_cols", 6).value
        rows = self.declare_parameter("checkerboard_rows", 9).value
        self._checkerboard = (int(cols), int(rows))
        square_size = float(self.declare_parameter("square_size", 1.0).value)
        self._criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            int(self.declare_parameter("corner_refinement_iters", 30).value),
            float(self.declare_parameter("corner_refinement_eps", 1e-3).value),
        )
        self._required_samples = int(self.declare_parameter("required_samples", 20).value)
        self._fast_check = bool(self.declare_parameter("use_fast_check", False).value)
        self._image_topic = self.declare_parameter("image_topic", "/image").value
        self._camera_frame = self.declare_parameter("camera_frame", "camera").value
        self._distortion_model = self.declare_parameter("distortion_model", "plumb_bob").value

        self._obj_template = self._build_object_points(self._checkerboard, square_size)
        self._objpoints: List[np.ndarray] = []
        self._imgpoints: List[np.ndarray] = []
        self._last_gray_shape: Optional[Tuple[int, int]] = None
        self._last_header_stamp: Optional[Time] = None
        self._result: Optional[CalibrationResult] = None
        self._camera_info: Optional[CameraInfo] = None
        self._calibration_complete = False

        self._camera_info_pub = self.create_publisher(CameraInfo, "camera_info", 10)
        self._image_sub = self.create_subscription(Image, self._image_topic, self._handle_image, 10)
        self.create_service(Trigger, "reset_calibration", self._handle_reset)

        self.get_logger().info(
            "Listening for images on '%s' with checkerboard %dx%d", self._image_topic, cols, rows
        )

    @property
    def calibration_complete(self) -> bool:
        return self._calibration_complete

    @property
    def camera_info(self) -> Optional[CameraInfo]:
        return self._camera_info

    @property
    def result(self) -> Optional[CalibrationResult]:
        return self._result

    def _handle_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._objpoints.clear()
        self._imgpoints.clear()
        self._last_gray_shape = None
        self._last_header_stamp = None
        self._result = None
        self._camera_info = None
        self._calibration_complete = False
        response.success = True
        response.message = "calibration state cleared"
        self.get_logger().info("Calibration state reset")
        return response

    def _handle_image(self, msg: Image) -> None:
        if self._calibration_complete:
            return

        gray = self._image_to_gray(msg)
        if gray is None:
            return

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        if self._fast_check:
            flags |= cv2.CALIB_CB_FAST_CHECK

        found, corners = cv2.findChessboardCorners(gray, self._checkerboard, flags)
        if not found:
            self.get_logger().debug("Checkerboard not found in incoming frame")
            return

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self._criteria)
        self._objpoints.append(self._obj_template.copy())
        self._imgpoints.append(refined)
        self._last_gray_shape = gray.shape[::-1]
        self._last_header_stamp = msg.header.stamp
        self.get_logger().info("Accepted checkerboard frame %d", len(self._imgpoints))

        if len(self._imgpoints) >= self._required_samples:
            self._perform_calibration()

    def _perform_calibration(self) -> None:
        if not self._last_gray_shape or not self._imgpoints:
            self.get_logger().warning("Insufficient data to calibrate")
            return

        ret, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            self._objpoints,
            self._imgpoints,
            self._last_gray_shape,
            None,
            None,
        )
        if not ret:
            self.get_logger().error("Calibration failed")
            return

        self._result = CalibrationResult(matrix=matrix, distortion=distortion, rvecs=rvecs, tvecs=tvecs)
        self._publish_camera_info(matrix, distortion)
        self._calibration_complete = True
        self.get_logger().info("Calibration succeeded with reprojection error %.4f", ret)

    def _publish_camera_info(self, matrix: np.ndarray, distortion: np.ndarray) -> None:
        if self._last_gray_shape is None:
            return

        info = CameraInfo()
        info.header.stamp = self._last_header_stamp if self._last_header_stamp else Time()
        info.header.frame_id = self._camera_frame
        info.width, info.height = self._last_gray_shape
        info.distortion_model = self._distortion_model
        info.d = distortion.ravel().tolist()
        info.k = matrix.reshape(-1).tolist()
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [
            matrix[0, 0],
            0.0,
            matrix[0, 2],
            0.0,
            0.0,
            matrix[1, 1],
            matrix[1, 2],
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        self._camera_info = info
        self._camera_info_pub.publish(info)

    @staticmethod
    def _build_object_points(board: Tuple[int, int], square_size: float) -> np.ndarray:
        cols, rows = board
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
        objp = np.zeros((grid.shape[0], 3), dtype=np.float32)
        objp[:, :2] = grid * square_size
        return objp

    def _image_to_gray(self, msg: Image) -> Optional[np.ndarray]:
        array = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding == "mono8":
            if array.size != msg.height * msg.width:
                self.get_logger().warning("Invalid mono image payload size")
                return None
            return array.reshape((msg.height, msg.width))
        if msg.encoding in {"bgr8", "rgb8"}:
            expected = msg.height * msg.width * 3
            if array.size != expected:
                self.get_logger().warning("Invalid color image payload size")
                return None
            image = array.reshape((msg.height, msg.width, 3))
            if msg.encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.get_logger().warning("Unsupported image encoding '%s'", msg.encoding)
        return None


def main(args: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=args)
    node = CheckerboardCalibratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
