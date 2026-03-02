from __future__ import annotations

from dataclasses import dataclass
import json
import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tyro
from webpolicy.client import Client
from websockets import exceptions as ws_exceptions


@dataclass
class Config:
    hz: float = 20.0  # publish rate in Hz
    pub: str = "/webpolicy/state"  # output topic name
    host: str = "localhost"  # webpolicy host
    port: int = 8000  # webpolicy port


class BridgeNode(Node):
    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__("webpolicy_bridge")

        if cfg is None:
            cfg = Config(
                hz=float(self.declare_parameter("hz", 20.0).value),
                pub=str(self.declare_parameter("pub", "/webpolicy/state").value),
                host=str(self.declare_parameter("host", "localhost").value),
                port=int(self.declare_parameter("port", 8000).value),
            )
        self.cfg = cfg

        if self.cfg.hz <= 0:
            raise ValueError(f"hz must be > 0, got {self.cfg.hz}")

        self.client: Client | None = None
        self._last_connect_attempt_ns = 0
        self._connect_retry_s = 2.0
        self.pub = self.create_publisher(String, self.cfg.pub, 10)
        self.timer = self.create_timer(1.0 / self.cfg.hz, self._tick)

    def _tick(self) -> None:
        if not self._ensure_client():
            return

        try:
            payload = self.client.step({})
        except (RuntimeError, OSError, EOFError, ws_exceptions.WebSocketException) as exc:
            self.get_logger().warn(f"webpolicy step failed: {exc}. Reconnecting...")
            self.client = None
            return

        msg = String()
        try:
            msg.data = json.dumps(payload, default=self._json_default)
        except (TypeError, ValueError) as exc:
            self.get_logger().error(f"Failed to serialize webpolicy payload to JSON: {exc}")
            return
        self.pub.publish(msg)

    def _ensure_client(self) -> bool:
        if self.client is not None:
            return True

        now_ns = self.get_clock().now().nanoseconds
        retry_ns = int(self._connect_retry_s * 1e9)
        if now_ns - self._last_connect_attempt_ns < retry_ns:
            return False

        self._last_connect_attempt_ns = now_ns
        if not self._tcp_ready():
            self.get_logger().warn(
                f"webpolicy endpoint not reachable at {self.cfg.host}:{self.cfg.port}. "
                f"Retrying every {self._connect_retry_s:.1f}s."
            )
            return False

        try:
            self.client = Client(host=self.cfg.host, port=self.cfg.port)
        except (ConnectionRefusedError, OSError, EOFError, ws_exceptions.WebSocketException) as exc:
            self.get_logger().warn(
                f"webpolicy connect failed for ws://{self.cfg.host}:{self.cfg.port}: {exc}. "
                f"Retrying every {self._connect_retry_s:.1f}s."
            )
            return False

        self.get_logger().info(f"Connected to webpolicy server at ws://{self.cfg.host}:{self.cfg.port}")
        return True

    def _tcp_ready(self) -> bool:
        try:
            with socket.create_connection((self.cfg.host, self.cfg.port), timeout=0.2):
                return True
        except OSError:
            return False

    @staticmethod
    def _json_default(obj: object) -> object:
        tolist = getattr(obj, "tolist", None)
        if callable(tolist):
            return tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run(cfg: Config | None = None) -> None:
    rclpy.init(args=None)
    node = BridgeNode(cfg=cfg)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(cfg: Config) -> None:
    run(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))
