from dataclasses import dataclass
from typing import List, Dict, Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat

import numpy as np

#helpers so we can use transform3d which works with pixi/ros 
def quat_xyzw_to_matrix4(q):
    # q is [x, y, z, w]
    x, y, z, w = q
    R = quat2mat([w, x, y, z])  # transforms3d expects [w, x, y, z]
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    return T

def matrix4_to_quat_xyzw(T):
    # mat2quat returns [w, x, y, z]
    w, x, y, z = mat2quat(T[:3, :3])
    return np.array([x, y, z, w], dtype=float)


@dataclass
class Config:
    # TF frame for anchor
    anchor: str
    # TF frames to publish for each floater
    floaters: List[str]
    # topic that publishes PoseStamped for the anchor->tracker link
    at: str
    # topics that publish PoseStamped for each floater->tracker link (same order as floaters)
    ft: List[str]
    # global frame to publish results in (e.g., "world" or "map")
    world: str = "world"


# convert PoseStamped to 4x4 transform matrix
# rotation comes from p.pose.orientation, translation from p.pose.position
def pose_to_mat(p: PoseStamped) -> np.ndarray:
    q = [p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w]
    T = quat_xyzw_to_matrix4(q)
    T[:3, 3] = [p.pose.position.x, p.pose.position.y, p.pose.position.z]
    return T


# convert 4x4 matrix into a TransformStamped with given parent/child frames
def mat_to_tf(T: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child

    # translation is the last column of the matrix
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = T[:3, 3]
    q = matrix4_to_quat_xyzw(T)
    msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = q
    return msg


class StereoAnchorNode(Node):
    """
    Publishes global TFs for a set of floaters using an anchor frame and live poses.

    Chain used for each floater:
        world -> anchor   (from TF)
        anchor -> at      (PoseStamped on self.cfg.at)
        at -> floater     (PoseStamped on each topic in self.cfg.ft)

    The product gives world -> floater. That TF is broadcast at 30 Hz.
    """

    def __init__(self):
        super().__init__("stereo_anchor_node")

        # load params into config object
        self.cfg = Config(
            anchor=self.declare_parameter("anchor", "anchor").value,
            floaters=self.declare_parameter("floaters", ["cam_low", "cam_side"]).value,
            at=self.declare_parameter("at", "/anchor_tracker/pose").value,
            ft=self.declare_parameter("ft", ["/cam_low/pose", "/cam_side/pose"]).value,
            world=self.declare_parameter("world", "world").value,
        )

        # check floaters and ft lists are same length
        if len(self.cfg.floaters) != len(self.cfg.ft):
            self.get_logger().error("floaters and ft lists must be same length")
            raise RuntimeError("Config mismatch")

        # TF interfaces. Buffer stores TF tree, listener fills it, broadcaster publishes new TFs.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # latest messages on the pose topics
        # anchor -> at pose
        self.latest_at: Optional[PoseStamped] = None
        # at -> floater poses by floater name
        self.latest_ft: Dict[str, PoseStamped] = {}

        # subscriptions for the live poses
        self.create_subscription(PoseStamped, self.cfg.at, self._on_at, 10)
        for floater, topic in zip(self.cfg.floaters, self.cfg.ft):
            self.create_subscription(PoseStamped, topic, self._make_ft_cb(floater), 10)

        # loop at 30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self._tick)

    def _on_at(self, msg: PoseStamped):
        """Callback for the anchor-tracker pose: anchor -> at."""
        self.latest_at = msg

    def _make_ft_cb(self, floater_name: str):
        """Factory that returns a callback which stores at -> floater for a given floater."""
        def cb(msg: PoseStamped):
            self.latest_ft[floater_name] = msg
        return cb

    def _tick(self):
        """Periodic step. Compute world -> floater for each floater and publish as TF."""
        # need anchor -> at before we can compute anything
        if self.latest_at is None:
            return

        # use the node clock for TF timestamps
        stamp = self.get_clock().now().to_msg()

        # get world -> anchor from the TF tree
        try:
            t_anchor = self.tf_buffer.lookup_transform(self.cfg.world, self.cfg.anchor, rclpy.time.Time())
            T_world_anchor = self._tf_to_mat(t_anchor)
        except Exception as e:
            # throttle log output so it doesn't spam
            self.get_logger().warn(
                f"Waiting for TF {self.cfg.world}->{self.cfg.anchor}: {e}",
                throttle_duration_sec=5.0,
            )
            return

        # convert anchor -> at to matrix
        T_anchor_at = pose_to_mat(self.latest_at)

        # for each floater we have a recent at -> floater pose
        for floater, ft_pose in self.latest_ft.items():
            T_at_ft = pose_to_mat(ft_pose)

            # chain transforms: world->floater = (world->anchor) * (anchor->at) * (at->floater)
            T_world_ft = T_world_anchor @ T_anchor_at @ T_at_ft

            # publish TF for this floater
            tf_msg = mat_to_tf(T_world_ft, self.cfg.world, floater, stamp)
            self.tf_broadcaster.sendTransform(tf_msg)

    @staticmethod
    def _tf_to_mat(t: TransformStamped) -> np.ndarray:
        """Convert a TransformStamped to a 4x4 matrix."""
        q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
        T = quat_xyzw_to_matrix4(q)
        T[:3, 3] = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
        return T


def main():
    """Entrypoint. Initialize ROS, spin the node, then shut down."""
    rclpy.init()
    node = StereoAnchorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

