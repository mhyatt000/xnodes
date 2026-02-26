# hand_viz_node.py
from __future__ import annotations

from geometry_msgs.msg import Point
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Header
from visualization_msgs.msg import Marker, MarkerArray

# 21-joint skeleton edges (wrist=0)
EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (5, 9),
    (9, 13),
    (13, 17),
    (17, 5),  # palm loop (optional)
]


class HandViz(Node):
    def __init__(self):
        super().__init__("hand_viz")
        self.frame_id = self.declare_parameter("frame_id", "world").get_parameter_value().string_value

        print("tmp")
        self.random_pub = self.create_subscription(Float32MultiArray, "/hand/points", self.cb_a, 10)

        self.a_sub = self.create_subscription(Float32MultiArray, "/hand/a", self.cb_a, 10)
        self.b_sub = self.create_subscription(Float32MultiArray, "/hand/b", self.cb_b, 10)
        self.a_pub = self.create_publisher(MarkerArray, "hand/markers_a", 10)
        self.b_pub = self.create_publisher(MarkerArray, "hand/markers_b", 10)

        self.radius = self.declare_parameter("joint_radius", 0.01).get_parameter_value().double_value
        self.line_width = self.declare_parameter("bone_width", 0.005).get_parameter_value().double_value

        self.get_logger().info("Hand node started")

    def cb_a(self, msg: Float32MultiArray):
        out = self.cb(msg)
        self.a_pub.publish(out)

    def cb_b(self, msg: Float32MultiArray):
        out = self.cb(msg)
        self.b_pub.publish(out)

    def cb(self, msg: Float32MultiArray):
        print("cb")
        data = np.array(msg.data, dtype=np.float32)
        if data.size != 63:
            self.get_logger().warn(f"Expected 63 floats, got {data.size}")
            return
        pts = data.reshape(21, 3)

        now = self.get_clock().now().to_msg()
        arr = MarkerArray()

        # joints
        m_pts = Marker()
        m_pts.header = Header(stamp=now, frame_id=self.frame_id)
        m_pts.ns = "hand"
        m_pts.id = 0
        m_pts.type = Marker.SPHERE_LIST
        m_pts.action = Marker.ADD
        m_pts.scale.x = self.radius
        m_pts.scale.y = self.radius
        m_pts.scale.z = self.radius
        m_pts.color.r = 0.1
        m_pts.color.g = 0.9
        m_pts.color.b = 0.1
        m_pts.color.a = 1.0
        m_pts.lifetime.sec = 0  # persist until updated
        m_pts.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in pts]
        arr.markers.append(m_pts)

        # bones
        m_lines = Marker()
        m_lines.header = Header(stamp=now, frame_id=self.frame_id)
        m_lines.ns = "hand"
        m_lines.id = 1
        m_lines.type = Marker.LINE_LIST
        m_lines.action = Marker.ADD
        m_lines.scale.x = self.line_width
        m_lines.color.r = 0.1
        m_lines.color.g = 0.3
        m_lines.color.b = 1.0
        m_lines.color.a = 1.0
        for i, j in EDGES:
            p, q = pts[i], pts[j]
            m_lines.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            m_lines.points.append(Point(x=float(q[0]), y=float(q[1]), z=float(q[2])))
        arr.markers.append(m_lines)

        return arr


def main():
    print("tmp1")
    rclpy.init()
    node = HandViz()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
