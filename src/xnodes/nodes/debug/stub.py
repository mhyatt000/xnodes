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
    name: str = "stub_node"  # ROS node name (used for direct CLI execution)
    rate_hz: float = 1.0  # Publish frequency


class StubNode(Node):
    def __init__(self, cfg: Config | None = None):
        node_name = "stub_node" if cfg is None else cfg.name
        super().__init__(node_name, automatically_declare_parameters_from_overrides=cfg is None)

        if cfg is None:
            rate_hz = float(self.declare_parameter("rate_hz", 1.0).value)
            spec = self._read_spec_from_params()
            spec_source = "ros parameters"
        else:
            rate_hz = cfg.rate_hz
            spec = self._read_spec_file(cfg.spec)
            spec_source = str(cfg.spec)

        self.spec = spec
        flat = flatten_dict(self.spec, sep="/")

        self._shapes: dict[str, tuple[int, ...]] = {}
        self._pubs: dict[str, object] = {}
        for topic, shape in flat.items():
            topic_str = str(topic)
            self._shapes[topic_str] = self._parse_shape(shape, topic_str)
            self._pubs[topic_str] = self.create_publisher(Float32MultiArray, topic_str, 10)

        if rate_hz <= 0:
            raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
        self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info(f"Stub spec loaded from {spec_source}; publishers={len(self._pubs)}")

    @property
    def logger(self) -> rclpy.logging.Logger:
        return self.get_logger()

    def _read_spec_file(self, path: Path) -> dict:
        p = Path(path).expanduser().resolve()
        self.logger.info(f"Reading spec from file {p}")
        if not p.exists():
            raise FileNotFoundError(f"spec file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            spec = yaml.load(f, Loader=yaml.FullLoader)
        if not isinstance(spec, dict):
            raise TypeError(f"spec must be a dict, got {type(spec).__name__}")
        return spec

    def _read_spec_from_params(self) -> dict:
        if self.has_parameter("spec"):
            spec = self.get_parameter("spec").value
            if isinstance(spec, str):
                return self._read_spec_file(Path(spec))

        prefixed = self.get_parameters_by_prefix("spec")
        if not prefixed:
            raise ValueError("Missing spec. Set ros__parameters.spec as a dict or a string file path.")

        root: dict[str, object] = {}
        for dotted_name, param in prefixed.items():
            parts = dotted_name.split(".")
            cur = root
            for part in parts[:-1]:
                nxt = cur.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    cur[part] = nxt
                cur = nxt
            cur[parts[-1]] = param.value
        return root

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
    rclpy.init(args=None)
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
