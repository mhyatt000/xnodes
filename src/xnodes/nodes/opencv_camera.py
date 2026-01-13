from __future__ import annotations

import base64
import copy
from dataclasses import dataclass, field
import enum
import os
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rich import print
from sensor_msgs.msg import CameraInfo, Image
import tyro
import yaml
from webpolicy.client import Client


class RES(enum.Enum):
    HD = (1280, 720)
    FHD = (1920, 1080)
    QHD = (2560, 1440)
    UHD_4K = (3840, 2160)

    VGA = (640, 480)
    SVGA = (800, 600)

    RES_224 = (224, 224)


@dataclass
class CamConfig:
    device: str = "/dev/video0"
    video_id: int = -1
    image_size: Sequence[int] = field(default_factory=lambda: [640, 480])
    output_encoding: str = "bgr8"
    camera_name: str = "camera"
    frame_id: str = "camera_optical_frame"
    fps: float = 30.0
    camera_info_url: str = ""
    attr: str = ""
    webpolicy_host: str | None = None
    webpolicy_port: int | None = None


class OpenCVCameraNode(Node):
    """Minimal OpenCV-based camera publisher for /image_raw and /camera_info."""

    def __init__(self, cfg: CamConfig | None = None) -> None:
        super().__init__("opencv_camera")
        self.bridge = CvBridge()

        if cfg is None:
            video_device = self.declare_parameter("video_device", "/dev/video0").value
            video_id = int(self.declare_parameter("video_id", -1).value)
            width = int(self.declare_parameter("width", 640).value)
            height = int(self.declare_parameter("height", 480).value)
            camera_name = self.declare_parameter("camera_name", "camera").value
            cfg = CamConfig(
                device=video_device if video_id < 0 else str(video_id),
                video_id=video_id,
                image_size=[width, height],
                output_encoding=self.declare_parameter("output_encoding", "bgr8").value,
                camera_name=camera_name,
                frame_id=self.declare_parameter("frame_id", f"{camera_name}_optical_frame").value,
                fps=float(self.declare_parameter("fps", 30.0).value),
                camera_info_url=self.declare_parameter("camera_info_url", "").value,
                attr=self.declare_parameter("attr", "").value,
                webpolicy_host=self.declare_parameter("webpolicy_host", "").value,
                webpolicy_port=int(self.declare_parameter("webpolicy_port", 0).value),
            )

        self.cfg = cfg
        self.encoding = self.cfg.output_encoding
        self.camera_name = self.cfg.camera_name
        self.frame_id = self.cfg.frame_id
        self.fps = float(self.cfg.fps)
        self.camera_info_url = self.cfg.camera_info_url
        self.width, self.height = self._parse_image_size(self.cfg.image_size)
        self.video_id = int(self.cfg.video_id)
        self.attr = self.cfg.attr
        self.webpolicy_host = (self.cfg.webpolicy_host or "").strip()
        self.webpolicy_port = int(self.cfg.webpolicy_port or 0)

        device_name = self.video_id if self.video_id >= 0 else self.cfg.device
        self.device = self._normalize_device_name(device_name)
        print(self.cfg)

        self.cap: cv2.VideoCapture | None = None
        self.web_client: Client | None = None
        if self.attr == "bridge:web":
            self._init_webpolicy_client()
        else:
            self.cap = self._open_camera(self.cfg.device)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            if self.width > 0 and self.height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
            if self.fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        prefix = f"cam/c{self.device}"
        self.image_pub = self.create_publisher(Image, f"{prefix}/image_raw", qos)
        self.info_pub = self.create_publisher(CameraInfo, f"{prefix}/camera_info", qos)

        self.base_info = self._load_camera_info()

        period = 1.0 / self.fps if self.fps > 0 else 0.1
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            # f"Publishing {self.video_device} as /image_raw at {self.fps:.1f} FPS ({self.width}x{self.height})"
            f"Publishing {self.cfg.device} as {prefix}/image_raw at {self.fps:.1f} FPS ({self.width}x{self.height})"
        )

    @property
    def url(self) -> Path:
        """default camera URL"""
        root = os.environ.get("PIXI_PROJECT_ROOT")
        path = Path(root) / "cam" / f"c{self.cfg.device}"
        return path

    def _parse_image_size(self, value: Sequence[float | int]) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError("image_size must contain [width, height]")
        return int(value[0]), int(value[1])

    def _open_camera(self, device: str) -> cv2.VideoCapture:
        device = int(device) if isinstance(device, int) or str(device).isdigit() else device
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {device}")
        return cap

    def _init_webpolicy_client(self) -> None:
        if not self.webpolicy_host or self.webpolicy_port <= 0:
            raise ValueError("webpolicy_host and webpolicy_port are required when attr == 'bridge:web'")
        self.video_id = self.video_id if self.video_id >= 0 else 0
        self.web_client = Client(host=self.webpolicy_host, port=self.webpolicy_port)
        self.get_logger().info(
            f"Using webpolicy bridge at {self.webpolicy_host}:{self.webpolicy_port} for camera id {self.video_id}"
        )

    def _load_camera_info(self) -> CameraInfo:
        url = self.camera_info_url
        if url:
            path = self._resolve_url(url)
        if not path:
            self.get_logger().warn(f"Unsupported camera_info_url: {self.camera_info_url}")
            path = self.url

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
        frame = self._read_frame()
        if frame is None:
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

    def _read_frame(self):
        if self.cap is not None:
            if not self.cap.isOpened():
                self.get_logger().warn("Camera closed; stopping timer")
                self.timer.cancel()
                return None
            ok, frame = self.cap.read()
            if not ok:
                self.get_logger().warn("Failed to grab frame", throttle_duration_sec=5.0)
                return None
            return frame

        if self.web_client is None:
            self.get_logger().warn("No capture source available", throttle_duration_sec=5.0)
            return None

        try:
            response = self.web_client.step({"id": self.video_id})
        except Exception as exc:  # pragma: no cover - depends on network
            self.get_logger().warn(f"Webpolicy request failed: {exc}", throttle_duration_sec=5.0)
            return None

        frame = self._decode_webpolicy_frame(response)
        if frame is None:
            self.get_logger().warn("Webpolicy response did not contain a valid image", throttle_duration_sec=5.0)
        return frame

    def _decode_webpolicy_frame(self, response):
        payload = response.get("image") if isinstance(response, dict) else response
        if isinstance(payload, bytes | bytearray):
            data = bytes(payload)
        elif isinstance(payload, str):
            try:
                data = base64.b64decode(payload)
            except Exception:
                return None
        elif isinstance(payload, np.ndarray):
            return payload
        else:
            return None

        buffer = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def _ensure_size(self, frame):
        h, w = frame.shape[:2]
        if w == self.width and h == self.height:
            return frame
        if self.width <= 0 or self.height <= 0:
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
        if not info.k.any():
            info.k = [1.0, 0.0, cx, 0.0, 1.0, cy, 0.0, 0.0, 1.0]
        if not info.r.any():
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        if not info.p.any():
            info.p = [1.0, 0.0, cx, 0.0, 0.0, 1.0, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.distortion_model = info.distortion_model or "plumb_bob"
        info.width = int(info.width)
        info.height = int(info.height)

    def _normalize_device_name(self, value: str | int) -> str:
        name = str(value).strip()
        if not name:
            return "camera"
        name = name.lstrip("/")
        for ch in "/\\: ":
            name = name.replace(ch, "_")
        return name

    def destroy_node(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def run(cfg: CamConfig | None = None):
    rclpy.init()
    node = OpenCVCameraNode(cfg)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def main(cfg: CamConfig):
    run(cfg)


if __name__ == "__main__":  # pragma: no cover - script entry
    main(tyro.cli(CamConfig))
