from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from builtin_interfaces.msg import Time
import numcodecs
from numcodecs import Blosc
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool
import zarr


def topic_to_key(topic: str) -> str:
    key = topic.strip("/").replace("/", "__")
    return key or "root"


def time_to_ns(stamp: Time) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def now_ns(node: Node) -> int:
    return node.get_clock().now().nanoseconds


def get_msg_stamp_ns(msg: Any, fallback_ns: int) -> int:
    # Common stamped layouts.
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        return time_to_ns(msg.header.stamp)
    if hasattr(msg, "stamp") and isinstance(msg.stamp, Time):
        return time_to_ns(msg.stamp)
    return fallback_ns


def qos_for_topic_name(topic: str) -> QoSProfile:
    # Conservative defaults. Image-like topics are typically sensor-data/best-effort.
    if "image" in topic or "camera" in topic:
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=50,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


@dataclass
class TopicState:
    topic: str
    ros_type: str
    msg_cls: type
    latest_msg: Any | None = None
    latest_stamp_ns: int = -1
    seen_count: int = 0


class EpisodeWriter:
    """Per-episode Zarr writer with chunked append semantics.

    Design:
    - One group per episode.
    - `steps/timestamp_ns`: global aligned step timestamps.
    - Per-topic subgroup under `topics/<sanitized_topic>`.
    - Known types get typed datasets.
    - Unknown/custom types fall back to JSON payload strings.
    """

    def __init__(self, root: Path, episode_idx: int, flush_chunk_size: int) -> None:
        self.root = root
        self.episode_idx = episode_idx
        self.flush_chunk_size = flush_chunk_size
        self.episode_dir = root / f"episode_{episode_idx:06d}.zarr"
        self.store = zarr.open_group(str(self.episode_dir), mode="w")
        self.store.attrs["episode_idx"] = episode_idx
        self.store.attrs["format_version"] = 1
        self.store.attrs["flush_chunk_size"] = flush_chunk_size
        self._compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        self._steps = self.store.create_group("steps")
        self._topics = self.store.create_group("topics")
        self._steps.create_dataset(
            "timestamp_ns",
            shape=(0,),
            chunks=(max(128, flush_chunk_size),),
            dtype="i8",
            compressor=self._compressor,
        )
        self._topic_groups: dict[str, zarr.Group] = {}
        self._topic_modes: dict[str, str] = {}
        self._frame_shapes: dict[str, tuple[int, ...]] = {}

    def _ensure_topic_group(self, topic: str, ros_type: str) -> zarr.Group:
        key = topic_to_key(topic)
        if key not in self._topic_groups:
            grp = self._topics.create_group(key)
            grp.attrs["topic"] = topic
            grp.attrs["ros_type"] = ros_type
            self._topic_groups[key] = grp
        return self._topic_groups[key]

    def append_steps(self, step_times_ns: np.ndarray) -> None:
        ds = self._steps["timestamp_ns"]
        old_n = ds.shape[0]
        ds.resize(old_n + len(step_times_ns))
        ds[old_n:] = step_times_ns

    def append_topic_batch(self, topic: str, ros_type: str, stamps_ns: np.ndarray, payloads: list[Any]) -> None:
        grp = self._ensure_topic_group(topic, ros_type)
        mode = self._topic_modes.get(topic)
        if mode is None:
            sample = next((p for p in payloads if p is not None), None)
            mode = self._infer_mode(sample)
            self._topic_modes[topic] = mode
            grp.attrs["mode"] = mode
            self._init_topic_storage(grp, mode, sample, len(stamps_ns))

        self._append_stamps(grp, stamps_ns)
        self._append_payloads(grp, mode, payloads)

    def finalize(self, metadata: dict[str, Any]) -> None:
        self.store.attrs["metadata"] = json.dumps(metadata)

    def _infer_mode(self, sample: Any | None) -> str:
        if sample is None:
            return "json"
        if isinstance(sample, np.ndarray):
            return "image"
        if isinstance(sample, dict) and {"name", "position", "velocity", "effort"} <= set(sample.keys()):
            return "joint_state"
        if isinstance(sample, dict) and {"format", "data"} <= set(sample.keys()):
            return "compressed_image"
        return "json"

    def _init_topic_storage(self, grp: zarr.Group, mode: str, sample: Any | None, chunk_hint: int) -> None:
        grp.create_dataset(
            "stamp_ns",
            shape=(0,),
            chunks=(max(128, chunk_hint),),
            dtype="i8",
            compressor=self._compressor,
        )
        if mode == "image" and isinstance(sample, np.ndarray):
            shape = sample.shape
            self._frame_shapes[grp.attrs["topic"]] = shape
            grp.attrs["frame_shape"] = list(shape)
            grp.create_dataset(
                "data",
                shape=(0, *shape),
                chunks=(max(1, min(16, chunk_hint)), *shape),
                dtype=sample.dtype,
                compressor=self._compressor,
            )
        elif mode == "joint_state":
            grp.create_dataset(
                "name_json",
                shape=(0,),
                chunks=(max(128, chunk_hint),),
                dtype=zarr.string_dtype(),
                compressor=self._compressor,
            )
            for field in ("position", "velocity", "effort"):
                grp.create_dataset(
                    field,
                    shape=(0,),
                    chunks=(max(128, chunk_hint),),
                    dtype=object,
                    object_codec=numcodecs.VLenArray(np.float64),
                )
        elif mode == "compressed_image":
            grp.create_dataset(
                "format",
                shape=(0,),
                chunks=(max(128, chunk_hint),),
                dtype=zarr.string_dtype(),
                compressor=self._compressor,
            )
            grp.create_dataset(
                "data",
                shape=(0,),
                chunks=(max(128, chunk_hint),),
                dtype=object,
                object_codec=numcodecs.VLenArray(np.uint8),
            )
        else:
            grp.create_dataset(
                "json",
                shape=(0,),
                chunks=(max(128, chunk_hint),),
                dtype=zarr.string_dtype(),
                compressor=self._compressor,
            )

    def _append_stamps(self, grp: zarr.Group, stamps_ns: np.ndarray) -> None:
        ds = grp["stamp_ns"]
        old_n = ds.shape[0]
        ds.resize(old_n + len(stamps_ns))
        ds[old_n:] = stamps_ns

    def _append_payloads(self, grp: zarr.Group, mode: str, payloads: list[Any]) -> None:
        if mode == "image":
            ds = grp["data"]
            old_n = ds.shape[0]
            frames = np.stack(list(payloads), axis=0)
            ds.resize(old_n + frames.shape[0], axis=0)
            ds[old_n:] = frames
            return

        if mode == "joint_state":
            old_n = grp["name_json"].shape[0]
            n = len(payloads)
            grp["name_json"].resize(old_n + n)
            grp["position"].resize(old_n + n)
            grp["velocity"].resize(old_n + n)
            grp["effort"].resize(old_n + n)
            for i, p in enumerate(payloads):
                grp["name_json"][old_n + i] = json.dumps(p["name"])
                grp["position"][old_n + i] = np.asarray(p["position"], dtype=np.float64)
                grp["velocity"][old_n + i] = np.asarray(p["velocity"], dtype=np.float64)
                grp["effort"][old_n + i] = np.asarray(p["effort"], dtype=np.float64)
            return

        if mode == "compressed_image":
            old_n = grp["format"].shape[0]
            n = len(payloads)
            grp["format"].resize(old_n + n)
            grp["data"].resize(old_n + n)
            for i, p in enumerate(payloads):
                grp["format"][old_n + i] = p["format"]
                grp["data"][old_n + i] = np.asarray(p["data"], dtype=np.uint8)
            return

        ds = grp["json"]
        old_n = ds.shape[0]
        ds.resize(old_n + len(payloads))
        ds[old_n:] = [json.dumps(p) for p in payloads]


class RecordNode(Node):
    def __init__(self) -> None:
        super().__init__("record_node")

        self.declare_parameter(
            "signal_topic",
            "/xgym/active",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            "output_dir",
            "/tmp/episodes",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING),
        )
        self.declare_parameter(
            "topics",
            [""],
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY),
        )
        self.declare_parameter("flush_chunk_size", 32)
        self.declare_parameter("buf_hz", 50.0)
        self.declare_parameter("max_topic_age_ms", 200.0)
        self.declare_parameter("auto_discover_timeout_sec", 10.0)

        self.signal_topic = str(self.get_parameter("signal_topic").value)
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.topic_names = [t for t in self.get_parameter("topics").value if t]
        self.flush_chunk_size = int(self.get_parameter("flush_chunk_size").value)
        self.buf_hz = float(self.get_parameter("buf_hz").value)
        self.max_topic_age_ms = float(self.get_parameter("max_topic_age_ms").value)
        self.auto_discover_timeout_sec = float(self.get_parameter("auto_discover_timeout_sec").value)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.recording = False
        self.episode_idx = self._next_episode_idx()
        self.writer: EpisodeWriter | None = None
        self.topic_states: dict[str, TopicState] = {}
        self.subscriptions = []
        self._flush_step_times: list[int] = []
        self._flush_payloads: dict[str, list[Any]] = defaultdict(list)
        self._flush_payload_stamps: dict[str, list[int]] = defaultdict(list)
        self._episode_start_ns: int | None = None
        self._episode_samples = 0

        self.signal_sub = self.create_subscription(
            Bool,
            self.signal_topic,
            self._signal_cb,
            qos_for_topic_name(self.signal_topic),
        )

        self._discovery_deadline_ns = now_ns(self) + int(self.auto_discover_timeout_sec * 1e9)
        self._discovery_timer = self.create_timer(0.5, self._discovery_tick)
        self._buf_timer = self.create_timer(1.0 / self.buf_hz, self._buffer_tick)

        self.get_logger().info(f"signal_topic={self.signal_topic}")
        self.get_logger().info(f"output_dir={self.output_dir}")
        self.get_logger().info(f"flush_chunk_size={self.flush_chunk_size}, buf_hz={self.buf_hz}")

    def _next_episode_idx(self) -> int:
        existing = sorted(self.output_dir.glob("episode_*.zarr"))
        if not existing:
            return 0
        last = existing[-1].stem
        m = re.search(r"(\d+)$", last)
        return (int(m.group(1)) + 1) if m else len(existing)

    def _discovery_tick(self) -> None:
        remaining = [t for t in self.topic_names if t not in self.topic_states]
        if not remaining:
            self.get_logger().info("all configured topics discovered; stopping discovery timer")
            self.destroy_timer(self._discovery_timer)
            return

        discovered = dict(self.get_topic_names_and_types())
        for topic in list(remaining):
            types = discovered.get(topic, [])
            if len(types) == 1:
                self._create_dynamic_subscription(topic, types[0])
            elif len(types) > 1:
                self.get_logger().error(
                    f"topic {topic} has multiple types {types}; pick one explicitly or fix publisher setup"
                )

        if now_ns(self) > self._discovery_deadline_ns:
            missing = [t for t in self.topic_names if t not in self.topic_states]
            self.get_logger().warning(f"discovery timeout; missing topics: {missing}")
            self.destroy_timer(self._discovery_timer)

    def _create_dynamic_subscription(self, topic: str, ros_type: str) -> None:
        msg_cls = get_message(ros_type)
        state = TopicState(topic=topic, ros_type=ros_type, msg_cls=msg_cls)
        self.topic_states[topic] = state

        def _cb(msg: Any, *, _topic: str = topic) -> None:
            self._topic_cb(_topic, msg)

        sub = self.create_subscription(msg_cls, topic, _cb, qos_for_topic_name(topic))
        self.subscriptions.append(sub)
        self.get_logger().info(f"subscribed: {topic} [{ros_type}]")

    def _topic_cb(self, topic: str, msg: Any) -> None:
        state = self.topic_states[topic]
        recv_ns = now_ns(self)
        state.latest_msg = msg
        state.latest_stamp_ns = get_msg_stamp_ns(msg, recv_ns)
        state.seen_count += 1

    def _signal_cb(self, msg: Bool) -> None:
        if msg.data and not self.recording:
            self._start_episode()
        elif not msg.data and self.recording:
            self._stop_episode()

    def _start_episode(self) -> None:
        self.recording = True
        self._episode_start_ns = now_ns(self)
        self._episode_samples = 0
        self._flush_step_times.clear()
        self._flush_payloads.clear()
        self._flush_payload_stamps.clear()
        self.writer = EpisodeWriter(self.output_dir, self.episode_idx, self.flush_chunk_size)
        self.writer.store.attrs["configured_topics"] = list(self.topic_names)
        discovered_types = {t: s.ros_type for t, s in self.topic_states.items()}
        self.writer.store.attrs["discovered_types"] = json.dumps(discovered_types)
        self.get_logger().info(f"start episode {self.episode_idx:06d}")

    def _stop_episode(self) -> None:
        self._flush()
        if self.writer is not None:
            end_ns = now_ns(self)
            metadata = {
                "episode_idx": self.episode_idx,
                "start_ns": self._episode_start_ns,
                "end_ns": end_ns,
                "num_samples": self._episode_samples,
                "buf_hz": self.buf_hz,
                "flush_chunk_size": self.flush_chunk_size,
                "max_topic_age_ms": self.max_topic_age_ms,
            }
            self.writer.finalize(metadata)
            self.get_logger().info(
                f"stop episode {self.episode_idx:06d}: samples={self._episode_samples}, path={self.writer.episode_dir}"
            )
        self.recording = False
        self.writer = None
        self.episode_idx += 1

    def _buffer_tick(self) -> None:
        if not self.recording or self.writer is None:
            return

        tick_ns = now_ns(self)
        max_age_ns = int(self.max_topic_age_ms * 1e6)
        row_payloads: dict[str, Any] = {}
        row_stamps: dict[str, int] = {}

        for topic, state in self.topic_states.items():
            if state.latest_msg is None:
                continue
            if tick_ns - state.latest_stamp_ns > max_age_ns:
                continue
            row_payloads[topic] = self._convert_msg(state.latest_msg)
            row_stamps[topic] = state.latest_stamp_ns

        if not row_payloads:
            return

        self._flush_step_times.append(tick_ns)
        for topic, payload in row_payloads.items():
            self._flush_payloads[topic].append(payload)
            self._flush_payload_stamps[topic].append(row_stamps[topic])
        self._episode_samples += 1

        if len(self._flush_step_times) >= self.flush_chunk_size:
            self._flush()

    def _flush(self) -> None:
        if not self._flush_step_times or self.writer is None:
            return

        step_times = np.asarray(self._flush_step_times, dtype=np.int64)
        self.writer.append_steps(step_times)
        for topic, payloads in self._flush_payloads.items():
            stamps = np.asarray(self._flush_payload_stamps[topic], dtype=np.int64)
            self.writer.append_topic_batch(topic, self.topic_states[topic].ros_type, stamps, payloads)

        self._flush_step_times.clear()
        self._flush_payloads.clear()
        self._flush_payload_stamps.clear()

    def _convert_msg(self, msg: Any) -> Any:
        if isinstance(msg, Image):
            return self._image_to_numpy(msg)
        if isinstance(msg, CompressedImage):
            return {
                "format": msg.format,
                "data": np.frombuffer(msg.data, dtype=np.uint8),
            }
        if isinstance(msg, JointState):
            return {
                "name": list(msg.name),
                "position": list(msg.position),
                "velocity": list(msg.velocity),
                "effort": list(msg.effort),
            }
        return self._jsonify_ordereddict(message_to_ordereddict(msg))

    def _image_to_numpy(self, msg: Image) -> np.ndarray:
        dtype_map = {
            "rgb8": np.uint8,
            "bgr8": np.uint8,
            "rgba8": np.uint8,
            "bgra8": np.uint8,
            "mono8": np.uint8,
            "mono16": np.uint16,
        }
        dtype = dtype_map.get(msg.encoding)
        if dtype is None:
            # Fall back to raw bytes as JSON-compatible representation.
            return np.frombuffer(msg.data, dtype=np.uint8)

        channels = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
            "mono16": 1,
        }[msg.encoding]
        arr = np.frombuffer(msg.data, dtype=dtype)
        if channels == 1:
            return arr.reshape(msg.height, msg.width)
        return arr.reshape(msg.height, msg.width, channels)

    def _jsonify_ordereddict(self, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: self._jsonify_ordereddict(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._jsonify_ordereddict(v) for v in data]
        if isinstance(data, np.ndarray):
            return data.tolist()
        if hasattr(data, "tolist") and not isinstance(data, (str, bytes)):
            try:
                return data.tolist()
            except Exception:
                pass
        if isinstance(data, bytes):
            return list(data)
        if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
            return None
        return data


def main() -> None:
    rclpy.init()
    node = RecordNode()
    try:
        rclpy.spin(node)
    finally:
        if node.recording:
            node._stop_episode()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
