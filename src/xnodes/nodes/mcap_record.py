from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap_protobuf.writer import Writer
import rclpy
from rclpy.node import Node
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String


def create_message_types():
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "xnodes/mcap_record.proto"
    file_proto.package = "xnodes"
    file_proto.syntax = "proto3"

    chatter_proto = file_proto.message_type.add()
    chatter_proto.name = "Chatter"

    field = chatter_proto.field.add()
    field.name = "data"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    record_proto = file_proto.message_type.add()
    record_proto.name = "FlexRecord"

    fields = [
        ("topic", 1),
        ("ros_type", 2),
        ("json_data", 3),
    ]
    for name, number in fields:
        field = record_proto.field.add()
        field.name = name
        field.number = number
        field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)

    if hasattr(message_factory, "GetMessageClass"):
        get_class = message_factory.GetMessageClass
    else:
        factory = message_factory.MessageFactory(pool)
        get_class = factory.GetPrototype

    return (
        get_class(pool.FindMessageTypeByName("xnodes.Chatter")),
        get_class(pool.FindMessageTypeByName("xnodes.FlexRecord")),
    )


Chatter, FlexRecord = create_message_types()


class McapRecordNode(Node, ABC):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.declare_parameter("output_path", "output.mcap")
        self._path = Path(self.get_parameter("output_path").get_parameter_value().string_value)
        self._file = self._path.open("wb")
        self._writer = Writer(self._file)
        self._writer.__enter__()

    @abstractmethod
    def start_recording(self):
        pass

    def write_message(self, topic: str, message):
        now = self.get_clock().now().nanoseconds
        self._writer.write_message(
            topic=topic,
            message=message,
            log_time=now,
            publish_time=now,
        )

    def close(self):
        if self._writer is not None:
            self._writer.__exit__(None, None, None)
            self._writer = None
        if self._file is not None:
            self._file.close()
            self._file = None


class McapRecordDemo(McapRecordNode):
    def __init__(self):
        super().__init__("mcap_record_demo")
        self._sub = self.create_subscription(String, "/chatter", self._callback, 10)
        self.start_recording()

    def start_recording(self):
        self.get_logger().info(f"Recording /chatter to {self._path}")

    def _callback(self, msg: String):
        self.write_message("/chatter", Chatter(data=msg.data))


class McapRecordFlex(McapRecordNode):
    def __init__(self):
        super().__init__("mcap_record_flex")
        self.declare_parameter("topics", ["/chatter"])
        self._subs = []
        self.start_recording()

    def start_recording(self):
        topic_names = self.get_parameter("topics").get_parameter_value().string_array_value
        topic_types = dict(self.get_topic_names_and_types())

        for topic in topic_names:
            types = topic_types.get(topic)
            if not types:
                self.get_logger().warning(f"Skipping {topic}: topic not found")
                continue

            ros_type = types[0]
            msg_type = get_message(ros_type)
            self._subs.append(
                self.create_subscription(
                    msg_type,
                    topic,
                    self._create_callback(topic, ros_type),
                    10,
                )
            )
            self.get_logger().info(f"Recording {topic} ({ros_type}) to {self._path}")

    def _create_callback(self, topic: str, ros_type: str):
        def callback(msg):
            self.write_message(
                topic,
                FlexRecord(
                    topic=topic,
                    ros_type=ros_type,
                    json_data=json.dumps(message_to_ordereddict(msg)),
                ),
            )

        return callback


def main():
    rclpy.init()
    node = McapRecordDemo()
    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
