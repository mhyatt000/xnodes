from __future__ import annotations

from moveit_msgs.msg import PlanningScene
import rclpy
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray


class PlanningSceneBoxViz(Node):
    """Visualize latest box collision objects from /planning_scene as RViz markers."""

    def __init__(self) -> None:
        super().__init__("planning_scene_box_viz")

        # id -> list[(frame_id, pose, [sx, sy, sz])]
        self._boxes: dict[str, list[tuple[str, Pose, list[float]]]] = {}

        self._sub = self.create_subscription(
            PlanningScene,
            "/planning_scene",
            self._scene_cb,
            10,
        )
        self._pub = self.create_publisher(MarkerArray, "/collision_boxes", 10)
        self._timer = self.create_timer(0.1, self._tick)

    def _scene_cb(self, scene: PlanningScene) -> None:
        """Update cache with latest boxes per collision_object.id."""

        for obj in scene.world.collision_objects:
            frame_id = obj.header.frame_id or "world"

            obj_boxes: list[tuple[str, Pose, list[float]]] = []
            for prim, pose in zip(obj.primitives, obj.primitive_poses):
                if prim.type != SolidPrimitive.BOX:
                    continue
                if len(prim.dimensions) < 3:
                    continue

                dims = prim.dimensions[:3]
                obj_boxes.append((frame_id, pose, dims))

            # Only overwrite if this object currently has boxes
            if obj_boxes:
                self._boxes[obj.id] = obj_boxes

    def _tick(self) -> None:
        if not self._boxes:
            self.get_logger().debug("No cached collision boxes to publish.")
            return

        num_ids = len(self._boxes)
        num_boxes = sum(len(v) for v in self._boxes.values())
        self.get_logger().debug(f"Publishing {num_boxes} box markers from {num_ids} collision object ids.")

        now = self.get_clock().now().to_msg()
        ma = MarkerArray()

        marker_id = 0
        for obj_id, boxes in self._boxes.items():
            for frame_id, pose, dims in boxes:
                m = Marker()
                m.header.frame_id = frame_id
                m.header.stamp = now
                m.ns = f"planning_scene_boxes/{obj_id}"
                m.id = marker_id
                marker_id += 1

                m.type = Marker.CUBE
                m.action = Marker.ADD

                m.pose = pose
                m.scale.x, m.scale.y, m.scale.z = dims

                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
                m.color.a = 0.4

                m.lifetime.sec = 0
                m.lifetime.nanosec = 0

                ma.markers.append(m)

        self._pub.publish(ma)


def main() -> None:
    rclpy.init()
    node = PlanningSceneBoxViz()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
