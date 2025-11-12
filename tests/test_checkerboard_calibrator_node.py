from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
import pytest

SKIP_KWARGS = {"exc_type": ImportError}

pytest.importorskip("cv2", reason="OpenCV required for calibration tests", **SKIP_KWARGS)
pytest.importorskip("rclpy", reason="ROS 2 client library required", **SKIP_KWARGS)
pytest.importorskip("sensor_msgs.msg", reason="ROS 2 message definitions required", **SKIP_KWARGS)
pytest.importorskip("std_msgs.msg", reason="Standard ROS 2 messages required", **SKIP_KWARGS)
pytest.importorskip("std_srvs.srv", reason="Standard ROS 2 services required", **SKIP_KWARGS)

import cv2
import rclpy
from builtin_interfaces.msg import Time
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from std_srvs.srv import Trigger

from xnodes.calibration.nodes import CheckerboardCalibratorNode


@pytest.fixture(scope="module", autouse=True)
def rclpy_runtime() -> Iterable[None]:
    rclpy.init(args=None)
    yield
    rclpy.shutdown()


def _camera_matrix(focal_length: float, cx: float, cy: float) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] = focal_length
    matrix[1, 1] = focal_length
    matrix[0, 2] = cx
    matrix[1, 2] = cy
    return matrix


def _rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def _render_checkerboard(
    board: Tuple[int, int],
    square_size: float,
    image_size: Tuple[int, int],
    matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    width, height = image_size
    image = np.full((height, width), 255, dtype=np.uint8)
    squares_x, squares_y = board[0] + 1, board[1] + 1
    for y in range(squares_y):
        for x in range(squares_x):
            color = 0 if (x + y) % 2 == 0 else 255
            square = np.array(
                [
                    [x * square_size, y * square_size, 0.0],
                    [(x + 1) * square_size, y * square_size, 0.0],
                    [(x + 1) * square_size, (y + 1) * square_size, 0.0],
                    [x * square_size, (y + 1) * square_size, 0.0],
                ],
                dtype=np.float32,
            )
            projected, _ = cv2.projectPoints(square, rvec, tvec, matrix, None)
            polygon = np.round(projected).astype(np.int32)
            cv2.fillConvexPoly(image, polygon, color)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)


def _image_message(image: np.ndarray, stamp: Time) -> Image:
    msg = Image()
    msg.height, msg.width = image.shape[:2]
    msg.encoding = "bgr8"
    msg.step = msg.width * 3
    msg.data = image.tobytes()
    msg.header = Header()
    msg.header.frame_id = "camera"
    msg.header.stamp = stamp
    return msg


def test_calibration_matches_expected_intrinsics() -> None:
    board = (6, 9)
    square_size = 0.04
    width, height = 640, 480
    focal = 800.0
    cx, cy = width / 2.0, height / 2.0
    matrix = _camera_matrix(focal, cx, cy)
    overrides = [
        Parameter(name="checkerboard_cols", value=board[0]),
        Parameter(name="checkerboard_rows", value=board[1]),
        Parameter(name="square_size", value=square_size),
        Parameter(name="required_samples", value=8),
        Parameter(name="image_topic", value="/synthetic"),
    ]
    node = CheckerboardCalibratorNode(parameter_overrides=overrides)

    try:
        for index in range(8):
            angles = np.deg2rad(np.array([(-5 + index), 2.5 * np.sin(index), 1.5 * np.cos(index)]))
            rotation = _rotation_matrix(*angles)
            rvec, _ = cv2.Rodrigues(rotation)
            tvec = np.array([[0.0], [0.0], [1.2 + 0.02 * index]], dtype=np.float32)
            image = _render_checkerboard(board, square_size, (width, height), matrix, rvec, tvec)
            stamp = Time(sec=0, nanosec=index * 1000000)
            msg = _image_message(image, stamp)
            node._handle_image(msg)

        assert node.calibration_complete, "Calibration did not complete"
        camera_info = node.camera_info
        assert camera_info is not None
        calibrated = np.array(camera_info.k, dtype=np.float64).reshape(3, 3)
        np.testing.assert_allclose(calibrated, matrix, rtol=5e-2, atol=1.0)
        np.testing.assert_allclose(camera_info.d, np.zeros_like(camera_info.d), atol=5e-2)
    finally:
        node.destroy_node()


def test_reset_clears_internal_state() -> None:
    overrides = [
        Parameter(name="required_samples", value=1),
    ]
    node = CheckerboardCalibratorNode(parameter_overrides=overrides)
    try:
        result = node._handle_reset(Trigger.Request(), Trigger.Response())
        assert result.success
        assert node.camera_info is None
        assert not node.calibration_complete
    finally:
        node.destroy_node()
