from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flax.traverse_util import flatten_dict
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
import tyro
import yaml


@dataclass
class Config:
    spec: Path = Path("stub.yaml")  # Nested YAML dict with tuple/list shape leaves
    name: str = "stub_node"  # ROS node name
    rate_hz: float = 1.0  # Publish frequency


class StubNode(Node):
    def __init__(self, cfg: Config):
        super().__init__(cfg.name)
        self.cfg = cfg
        self.spec = self._read_spec(cfg.spec)
        flat = flatten_dict(self.spec, sep="/")

        self._shapes: dict[str, tuple[int, ...]] = {}
        self._pubs: dict[str, object] = {}
        for topic, shape in flat.items():
            topic_str = str(topic)
            self._shapes[topic_str] = self._parse_shape(shape, topic_str)
            self._pubs[topic_str] = self.create_publisher(Float32MultiArray, topic_str, 10)

        if cfg.rate_hz <= 0:
            raise ValueError(f"rate_hz must be > 0, got {cfg.rate_hz}")
        self.create_timer(1.0 / cfg.rate_hz, self._tick)

        self.get_logger().info(f"Stub spec loaded from {cfg.spec}; publishers={len(self._pubs)}")

    def _read_spec(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"spec file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            spec = yaml.load(f, Loader=yaml.FullLoader)
        if not isinstance(spec, dict):
            raise TypeError(f"spec must be a dict, got {type(spec).__name__}")
        return spec

    def _parse_shape(self, value: object, topic: str) -> tuple[int, ...]:
        if isinstance(value, int):
            if value <= 0:
                raise ValueError(f"shape dim must be > 0 for {topic}, got {value}")
            return (value,)
        if isinstance(value, tuple | list):
            shape = tuple(value)
            if not shape:
                raise ValueError(f"shape for {topic} cannot be empty")
            for dim in shape:
                if not isinstance(dim, int) or dim <= 0:
                    raise ValueError(f"invalid shape dim for {topic}: {shape}")
            return shape
        raise TypeError(f"shape for {topic} must be int/list/tuple, got {type(value).__name__}")

    def _make_layout(self, shape: tuple[int, ...]) -> MultiArrayLayout:
        dims: list[MultiArrayDimension] = []
        for i, size in enumerate(shape):
            stride = int(np.prod(shape[i:]))
            dims.append(MultiArrayDimension(label=f"dim_{i}", size=int(size), stride=stride))
        return MultiArrayLayout(dim=dims, data_offset=0)

    def _msg(self, shape: tuple[int, ...]) -> Float32MultiArray:
        arr = np.zeros(shape, dtype=np.float32)
        msg = Float32MultiArray()
        msg.layout = self._make_layout(shape)
        msg.data = arr.reshape(-1).tolist()
        return msg

    def _tick(self) -> None:
        for topic, pub in self._pubs.items():
            pub.publish(self._msg(self._shapes[topic]))


def run(cfg: Config | None = None) -> None:
    if cfg is None:
        cfg = Config()
    rclpy.init()
    node = StubNode(cfg)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(cfg: Config) -> None:
    run(cfg)


if __name__ == "__main__":
    main(tyro.cli(Config))
