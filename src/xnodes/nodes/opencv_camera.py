from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import cv2
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
import rclpy
from sensor_msgs.msg import CameraInfo, Image

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:  # pragma: no cover - optional dependency
    from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover - optional dependency
    get_package_share_directory = None


class OpenCVCameraNode(Node):
    """Minimal OpenCV-based camera publisher for /image_raw and /camera_info."""

    def __init__(self) -> None:
        super().__init__("opencv_camera")
        self.bridge = CvBridge()

        self.video_device = self.declare_parameter("video_device", "/dev/video0").value
        image_size = self.declare_parameter("image_size", [640, 480]).value
        self.encoding = self.declare_parameter("output_encoding", "bgr8").value
        self.camera_name = self.declare_parameter("camera_name", "camera").value
        self.frame_id = self.declare_parameter("frame_id", f"{self.camera_name}_optical_frame").value
        self.fps = float(self.declare_parameter("fps", 30.0).value)
        self.camera_info_url = self.declare_parameter("camera_info_url", "").value

        self.width, self.height = self._parse_image_size(image_size)
        self.cap = self._open_camera(self.video_device)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.image_pub = self.create_publisher(Image, "image_raw", qos)
        self.info_pub = self.create_publisher(CameraInfo, "camera_info", qos)

        self.base_info = self._load_camera_info()

        period = 1.0 / self.fps if self.fps > 0 else 0.1
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f"Publishing {self.video_device} as /image_raw at {self.fps:.1f} FPS ({self.width}x{self.height})"
        )

    def _parse_image_size(self, value: Sequence[float | int]) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError("image_size must contain [width, height]")
        return int(value[0]), int(value[1])

    def _open_camera(self, device: str) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {device}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        return cap

    def _load_camera_info(self) -> CameraInfo:
        if not self.camera_info_url:
            return CameraInfo()
        path = self._resolve_url(self.camera_info_url)
        if path is None:
            self.get_logger().warn(f"Unsupported camera_info_url: {self.camera_info_url}")
            return CameraInfo()
        if yaml is None:
            self.get_logger().warn("PyYAML is not installed; skipping camera_info file load")
            return CameraInfo()
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except Exception as exc:  # pragma: no cover - filesystem errors
            self.get_logger().warn(f"Failed to read camera info {path}: {exc}")
            return CameraInfo()
        return self._dict_to_camera_info(data)

    def _resolve_url(self, url: str) -> Path | None:
        parsed = urlparse(url)
        if not parsed.scheme or parsed.scheme == "file":
            path = Path(parsed.path or url).expanduser()
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            return path
        if parsed.scheme == "package" and get_package_share_directory is not None:
            try:
                pkg_dir = Path(get_package_share_directory(parsed.netloc))
            except Exception as exc:  # pragma: no cover - depends on ROS env
                self.get_logger().warn(f"Unable to resolve package:// URL {url}: {exc}")
                return None
            return (pkg_dir / parsed.path.lstrip("/")).resolve()
        return None

    def _dict_to_camera_info(self, data: dict) -> CameraInfo:
        info = CameraInfo()
        info.width = int(data.get("image_width", 0) or 0)
        info.height = int(data.get("image_height", 0) or 0)
        info.distortion_model = data.get("distortion_model", "plumb_bob")
        info.d = list(data.get("distortion_coefficients", {}).get("data", []))
        info.k = list(data.get("camera_matrix", {}).get("data", []))
        info.r = list(data.get("rectification_matrix", {}).get("data", []))
        info.p = list(data.get("projection_matrix", {}).get("data", []))
        return info

    def _tick(self) -> None:
        if not self.cap.isOpened():
            self.get_logger().warn("Camera closed; stopping timer")
            self.timer.cancel()
            return

        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn("Failed to grab frame", throttle_duration_sec=5.0)
            return

        frame = self._ensure_size(frame)
        stamp = self.get_clock().now().to_msg()
        image = self._frame_to_msg(frame)
        image.header.stamp = stamp
        image.header.frame_id = self.frame_id
        self.image_pub.publish(image)

        info = copy.deepcopy(self.base_info)
        self._populate_camera_info(info, frame)
        info.header = image.header
        self.info_pub.publish(info)

    def _ensure_size(self, frame):
        h, w = frame.shape[:2]
        if w == self.width and h == self.height:
            return frame
        return cv2.resize(frame, (self.width, self.height))

    def _frame_to_msg(self, frame):
        encoding = self.encoding
        if encoding == "bgr8":
            converted = frame
        elif encoding == "rgb8":
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif encoding == "mono8":
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self.get_logger().warn(f"Unsupported encoding {encoding}, defaulting to bgr8")
            self.encoding = "bgr8"
            converted = frame
            encoding = "bgr8"
        return self.bridge.cv2_to_imgmsg(converted, encoding=encoding)

    def _populate_camera_info(self, info: CameraInfo, frame) -> None:
        if info.width == 0 or info.height == 0:
            info.width = frame.shape[1]
            info.height = frame.shape[0]
        cx = info.width / 2.0
        cy = info.height / 2.0
        if not info.d:
            info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        if not info.k:
            info.k = [1.0, 0.0, cx, 0.0, 1.0, cy, 0.0, 0.0, 1.0]
        if not info.r:
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        if not info.p:
            info.p = [1.0, 0.0, cx, 0.0, 0.0, 1.0, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.distortion_model = info.distortion_model or "plumb_bob"
        info.width = int(info.width)
        info.height = int(info.height)

    def destroy_node(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = OpenCVCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover - interactive node
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
