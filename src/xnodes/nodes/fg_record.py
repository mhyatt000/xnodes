from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any, Callable

import foxglove
import foxglove.channels as fg_channels
from foxglove.messages import CompressedImage, JointState, JointStates, RawImage, Timestamp
import rclpy
from rclpy.node import Node
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message

from xnodes.nodes.legacy.active import ActiveFlag


def _stamp(msg) -> Timestamp:
    return Timestamp(msg.header.stamp.sec, msg.header.stamp.nanosec)


def _to_raw_image(msg) -> RawImage:
    return RawImage(
        timestamp=_stamp(msg),
        frame_id=msg.header.frame_id,
        width=msg.width,
        height=msg.height,
        encoding=msg.encoding,
        step=msg.step,
        data=bytes(msg.data),
    )


def _to_compressed_image(msg) -> CompressedImage:
    return CompressedImage(
        timestamp=_stamp(msg),
        frame_id=msg.header.frame_id,
        format=msg.format,
        data=bytes(msg.data),
    )


def _to_joint_states(msg) -> JointStates:
    return JointStates(
        timestamp=_stamp(msg),
        joints=[
            JointState(name=n, position=p, velocity=v, effort=e)
            for n, p, v, e in zip(msg.name, msg.position, msg.velocity, msg.effort)
        ],
    )


_DISPATCH: dict[str, tuple[type, Callable]] = {
    "sensor_msgs/msg/Image": (fg_channels.RawImageChannel, _to_raw_image),
    "sensor_msgs/msg/CompressedImage": (fg_channels.CompressedImageChannel, _to_compressed_image),
    "sensor_msgs/msg/JointState": (fg_channels.JointStatesChannel, _to_joint_states),
}


class FgRecordFlex(Node):
    def __init__(self):
        super().__init__("fg_record_flex")
        self.declare_parameter("output_dir", "/tmp/episodes")
        self.declare_parameter("note", "base")
        self.declare_parameter("signal_topic", "/xgym/active")
        self.declare_parameter("topics", ["/chatter"])
        self.declare_parameter("auto_discover_timeout_sec", 10.0)
        self.declare_parameter("signal_toggle", True)

        self._output_dir = Path(self.get_parameter("output_dir").get_parameter_value().string_value)
        self._note = self.get_parameter("note").get_parameter_value().string_value
        self._topic_names = [t for t in self.get_parameter("topics").get_parameter_value().string_array_value if t]
        signal_topic = self.get_parameter("signal_topic").get_parameter_value().string_value
        signal_toggle = self.get_parameter("signal_toggle").get_parameter_value().bool_value
        discover_timeout = self.get_parameter("auto_discover_timeout_sec").get_parameter_value().double_value

        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._recording = False
        self._mcap = None
        self._ctx: foxglove.Context | None = None
        self._channels: dict[str, tuple[Any, Callable]] = {}
        self._topic_info: dict[str, str] = {}  # topic -> ros_type
        self._subs: list = []

        self._pending = set(self._topic_names)
        self._discover_deadline_ns = self.get_clock().now().nanoseconds + int(discover_timeout * 1e9)
        self._discover_timer = self.create_timer(0.5, self._discovery_tick)
        self.create_timer(1.0, self._status_tick)

        self._active = ActiveFlag(self, topic=signal_topic, toggle=signal_toggle, on_change=self._on_active)

    def _discovery_tick(self) -> None:
        if not self._pending:
            self.destroy_timer(self._discover_timer)
            return

        discovered = dict(self.get_topic_names_and_types())
        for topic in list(self._pending):
            types = discovered.get(topic, [])
            if len(types) == 1:
                self._register_topic(topic, types[0])
                self._pending.discard(topic)
            elif len(types) > 1:
                self.get_logger().error(f"{topic}: multiple types {types}, fix publisher setup")

        if self.get_clock().now().nanoseconds > self._discover_deadline_ns:
            if self._pending:
                self.get_logger().warning(f"Discovery timeout; missing: {sorted(self._pending)}")
            self.destroy_timer(self._discover_timer)

    def _register_topic(self, topic: str, ros_type: str) -> None:
        self._topic_info[topic] = ros_type
        msg_cls = get_message(ros_type)
        self._subs.append(self.create_subscription(msg_cls, topic, self._make_cb(topic), 10))
        self.get_logger().info(f"Discovered {topic} ({ros_type})")

    def _make_cb(self, topic: str) -> Callable:
        def cb(msg) -> None:
            if not self._recording:
                return
            entry = self._channels.get(topic)
            if entry is None:
                return
            channel, converter = entry
            now = self.get_clock().now().nanoseconds
            channel.log(converter(msg), log_time=now)

        return cb

    def _status_tick(self) -> None:
        if not self._recording:
            return
        elapsed = (self.get_clock().now().nanoseconds - self._episode_start_ns) / 1e9
        self.get_logger().info(f"Recording ... {elapsed:.1f}s")

    def _on_active(self, active: bool) -> None:
        if active and not self._recording:
            self._start_episode()
        elif not active and self._recording:
            self._stop_episode()

    def _next_path(self) -> Path:
        date = datetime.now().strftime("%y%m%d")
        existing = list(self._output_dir.glob(f"{date}_episode_*.mcap"))
        indices = [int(m.group(1)) for f in existing if (m := re.search(r"episode_(\d+)", f.stem))]
        idx = (max(indices) + 1) if indices else 0
        return self._output_dir / f"{date}_episode_{idx:06d}_{self._note}.mcap"

    def _start_episode(self) -> None:
        path = self._next_path()
        self._episode_path = path
        self._ctx = foxglove.Context()
        self._mcap = foxglove.open_mcap(str(path), context=self._ctx)
        self._channels.clear()
        self._episode_start_ns = self.get_clock().now().nanoseconds

        for topic, ros_type in self._topic_info.items():
            entry = _DISPATCH.get(ros_type)
            if entry:
                channel_cls, converter = entry
                channel = channel_cls(topic, context=self._ctx)
                schema = "foxglove"
            else:
                channel = foxglove.Channel(topic, schema={"type": "object"}, context=self._ctx)
                converter = message_to_ordereddict
                schema = "json"
            self._channels[topic] = (channel, converter)
            self.get_logger().info(f"  {topic} ({ros_type}) [{schema}]")

        self._episode_meta = {
            "start_ns": str(self._episode_start_ns),
            "note": self._note,
            "topics": ",".join(self._topic_info.keys()),
            "ros_types": ",".join(self._topic_info.values()),
        }
        self._mcap.write_metadata("episode", self._episode_meta)

        self._recording = True
        self.get_logger().info(f"Episode started: {path.name}")

    def _stop_episode(self) -> None:
        self._recording = False
        self._channels.clear()
        if self._mcap is not None:
            end_ns = self.get_clock().now().nanoseconds
            duration_ns = end_ns - self._episode_start_ns
            end_meta = {
                "end_ns": str(end_ns),
                "duration_ns": str(duration_ns),
                "duration_s": f"{duration_ns / 1e9:.3f}",
            }
            self._mcap.write_metadata("episode_end", end_meta)
            self._mcap.close()
            self._mcap = None

            log = self.get_logger()
            log.info(f"Episode stopped: {self._episode_path}")
            log.info("--- metadata ---")
            for k, v in {**self._episode_meta, **end_meta}.items():
                log.info(f"  {k}: {v}")
            log.info("----------------")

        self._ctx = None

    def close(self) -> None:
        if self._recording:
            self._stop_episode()


def main_flex() -> None:
    rclpy.init()
    node = FgRecordFlex()
    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main_flex()
