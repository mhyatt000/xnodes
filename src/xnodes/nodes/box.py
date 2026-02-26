from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rich import print
from shape_msgs.msg import SolidPrimitive


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Add/remove a collision box to MoveIt's planning scene.")
    p.add_argument(
        "--topic",
        default="/planning_scene",
        help="Planning scene input topic (respect namespace, e.g. /xarm/planning_scene).",
    )
    p.add_argument("--id", default="scene_box", help="Collision object ID.")
    p.add_argument("--frame", default="base_link", help="Frame for the box pose.")
    p.add_argument(
        "--size",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[2.0, 2.0, 0.2],
        help="Box size in meters.",
    )
    p.add_argument(
        "--xyz",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, -0.0],
        help="Box position in meters.",
    )
    p.add_argument(
        "--rpy",
        type=float,
        nargs=3,
        metavar=("R", "P", "Y"),
        default=[0.0, 0.0, 0.0],
        help="Orientation in radians (roll, pitch, yaw).",
    )
    p.add_argument("--remove", action="store_true", help="Remove the box with the given ID.")
    p.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Times to publish the diff (helps ensure delivery).",
    )
    p.add_argument("--interval", type=float, default=0.25, help="Seconds between publishes.")
    return p.parse_args(argv)


@dataclass
class BoxConfig:
    topic: str = "/planning_scene"
    id: str = "scene_box"
    frame: str = "world"
    size: tuple[float, float, float] = (2.0, 2.0, 0.2)
    xyz: tuple[float, float, float] = (0.0, 0.0, -0.1)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    remove: bool = False

    repeats: int = 5
    interval: float = 0.25


def rpy_to_quat(roll, pitch, yaw):
    # Standard XYZ (roll, pitch, yaw) → quaternion
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


class AddBox(Node):
    def __init__(self, args):
        super().__init__("add_box_to_scene")
        self.args = args

        print(args)
        # Publisher to the planning scene diff topic (input to move_group)
        qos = QoSProfile(depth=10)  # standard reliable queue
        self.pub = self.create_publisher(PlanningScene, args.topic, qos)
        # Build the PlanningScene diff message once
        self.scene_msg = self._build_scene_msg()

        # Try to wait until move_group subscribes (optional but nice)
        self._wait_for_subscriber(timeout_sec=3.0)

        # Publish a few times to be safe (ROS 2 has no latching)
        self._publish_repeated()

        # Done
        self.get_logger().info("Done. Check RViz > MotionPlanning > Scene Objects.")

    def _wait_for_subscriber(self, timeout_sec: float):
        start = time.time()
        while (time.time() - start) < timeout_sec:
            if self.pub.get_subscription_count() > 0:
                self.get_logger().info(f"Subscriber detected on {self.args.topic} (move_group).")
                return
            time.sleep(0.1)
        self.get_logger().warn(
            f"No subscribers detected on {self.args.topic}. "
            "If box doesn't appear, confirm move_group is running and topic is correct."
        )

    def _build_scene_msg(self) -> PlanningScene:
        co = CollisionObject()
        co.id = self.args.id
        co.header.frame_id = self.args.frame

        print(self.args.frame)
        if self.args.remove:
            co.operation = CollisionObject.REMOVE
            self.get_logger().info(f"Preparing REMOVE of '{co.id}' in frame '{co.header.frame_id}'.")
        else:
            # Define primitive
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = list(self.args.size)  # [x, y, z]

            # Pose
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = self.args.xyz
            qx, qy, qz, qw = rpy_to_quat(*self.args.rpy)
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw

            co.primitives.append(prim)
            co.primitive_poses.append(pose)
            co.operation = CollisionObject.ADD

            self.get_logger().info(
                f"Preparing ADD '{co.id}' size={self.args.size} "
                f"xyz={self.args.xyz} rpy={self.args.rpy} frame='{co.header.frame_id}'."
            )

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(co)
        return scene

    def _publish_repeated(self):
        for i in range(self.args.repeats):
            self.pub.publish(self.scene_msg)
            self.get_logger().info(f"Published PlanningScene diff ({i + 1}/{self.args.repeats}) to {self.args.topic}")
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(self.args.interval)


def build():
    # args = parse_args(argv)

    l, w = 0.9144, 0.4572
    voffset = -0.100635
    hoffset = -0.07
    towall = hoffset + l
    mini = 0.02

    floor = BoxConfig(
        id="floor",
        xyz=(hoffset + l / 2, 0.0, -(l / 2)),
        size=(l, w, l),
    )
    todesk = BoxConfig(
        id="todesk",
        size=(2.0, mini, 2.0),
        xyz=(0.0, 0.55, 0.0),
    )
    tofront = BoxConfig(
        id="tofront",
        size=(mini, 2.0, 2.0),
        xyz=(towall + mini, 0.0, 0.0),
    )
    rclpy.init()
    # AddBox(args)

    AddBox(floor)
    AddBox(todesk)
    AddBox(tofront)

    rclpy.shutdown()


def main():
    build()


if __name__ == "__main__":
    main()
