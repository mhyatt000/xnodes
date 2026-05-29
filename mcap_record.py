from __future__ import annotations

from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap_protobuf.writer import Writer
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def create_chatter_message_type():
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "xnodes/chatter.proto"
    file_proto.package = "xnodes"
    file_proto.syntax = "proto3"

    msg_proto = file_proto.message_type.add()
    msg_proto.name = "Chatter"

    field = msg_proto.field.add()
    field.name = "data"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    descriptor = pool.FindMessageTypeByName("xnodes.Chatter")

    if hasattr(message_factory, "GetMessageClass"):
        return message_factory.GetMessageClass(descriptor)

    return message_factory.MessageFactory(pool).GetPrototype(descriptor)


Chatter = create_chatter_message_type()


class McapRecordNode(Node):
    def __init__(self):
        super().__init__("mcap_record")
        self._path = Path("output.mcap")
        self._file = self._path.open("wb")
        self._writer = Writer(self._file)
        self._writer.__enter__()
        self._sub = self.create_subscription(String, "/chatter", self._callback, 10)
        self.get_logger().info(f"Recording /chatter to {self._path}")

    def close(self):
        if self._writer is not None:
            self._writer.__exit__(None, None, None)
            self._writer = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def _callback(self, msg: String):
        now = self.get_clock().now().nanoseconds
        self._writer.write_message(
            topic="/chatter",
            message=Chatter(data=msg.data),
            log_time=now,
            publish_time=now,
        )


def main():
    rclpy.init()
    node = McapRecordNode()
    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
