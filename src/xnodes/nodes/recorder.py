from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import signal
import subprocess

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import tyro


@dataclass
class Config:
    signal_topic: str = "/xgym/active"  # Bool topic — rising edge starts, falling edge stops
    output_dir: str = "/tmp/episodes"  # where episode bags are written
    storage: str = "mcap"  # mcap or sqlite3
    topics: list[str] = field(default_factory=lambda: ["/joint_states"])


class RecorderNode(Node):
    def __init__(self, cfg: Config | None = None) -> None:
        super().__init__("recorder")

        if cfg is None:
            topics_raw = self.declare_parameter("topics", ["/joint_states"]).value
            cfg = Config(
                signal_topic=str(self.declare_parameter("signal_topic", "/xgym/active").value),
                output_dir=str(self.declare_parameter("output_dir", "/tmp/episodes").value),
                storage=str(self.declare_parameter("storage", "mcap").value),
                topics=list(topics_raw),
            )
        self.cfg = cfg

        self._proc: subprocess.Popen | None = None
        self._episode: int = 0
        self._active: bool = False

        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

        self.create_subscription(Bool, self.cfg.signal_topic, self._on_signal, 10)
        self.get_logger().info(f"Recorder ready. Listening on '{self.cfg.signal_topic}'. Output: {self.cfg.output_dir}")

    def _on_signal(self, msg: Bool) -> None:
        if msg.data and not self._active:
            self._start()
        elif not msg.data and self._active:
            self._stop()

    def _start(self) -> None:
        self._active = True
        bag_path = Path(self.cfg.output_dir) / f"episode_{self._episode:04d}"
        cmd = [
            "ros2",
            "bag",
            "record",
            "--storage",
            self.cfg.storage,
            "-o",
            str(bag_path),
            *self.cfg.topics,
        ]
        self._proc = subprocess.Popen(cmd)
        self.get_logger().info(f"[ep {self._episode:04d}] Recording started -> {bag_path}")

    def _stop(self) -> None:
        self._active = False
        if self._proc is not None:
            self._proc.send_signal(signal.SIGINT)
            self._proc.wait()
            self._proc = None
        self.get_logger().info(f"[ep {self._episode:04d}] Recording stopped.")
        self._episode += 1

    def destroy_node(self) -> bool:
        if self._active:
            self._stop()
        return super().destroy_node()


def run(cfg: Config | None = None) -> None:
    rclpy.init()
    node = RecorderNode(cfg=cfg)
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
