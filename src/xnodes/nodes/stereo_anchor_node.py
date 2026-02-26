from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
from pytransform3d.rotations import matrix_from_euler
from pytransform3d.rotations import (
    matrix_from_quaternion as quat2mat,
)  # aparently transform3d is not maintained anymore so pytransform3d is replacement
from pytransform3d.rotations import (
    quaternion_from_matrix as mat2quat,
)
import rclpy
from rclpy.node import Node
from rich import print
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
import tyro
import yaml


# helpers so we can use transform3d which works with pixi/ros
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
class StereoConfig:
    # TF frame for anchor
    anchor: str
    # TF frames to publish for each floater
    # ie: over_optical_frame
    floaters: list[str]
    # topic that publishes PoseStamped for the anchor->tracker link
    atf: str
    # topics that publish PoseStamped for each floater->tracker link (same order as floaters)
    ftf: list[str]
    # global frame to publish results in (e.g., "world" or "map")
    world: str = "world"


# convert PoseStamped to 4x4 transform matrix
# rotation comes from p.pose.orientation, translation from p.pose.position
def pose2mat(p: PoseStamped) -> np.ndarray:
    q = [
        p.pose.orientation.x,
        p.pose.orientation.y,
        p.pose.orientation.z,
        p.pose.orientation.w,
    ]
    T = quat_xyzw_to_matrix4(q)
    T[:3, 3] = [p.pose.position.x, p.pose.position.y, p.pose.position.z]
    return T


def tf2mat(t: TransformStamped) -> np.ndarray:
    q = [
        t.transform.rotation.x,
        t.transform.rotation.y,
        t.transform.rotation.z,
        t.transform.rotation.w,
    ]
    T = quat_xyzw_to_matrix4(q)
    T[:3, 3] = [
        t.transform.translation.x,
        t.transform.translation.y,
        t.transform.translation.z,
    ]
    return T


# convert 4x4 matrix into a TransformStamped with given parent/child frames
def mat2tf(T: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child

    # translation is the last column of the matrix
    (
        msg.transform.translation.x,
        msg.transform.translation.y,
        msg.transform.translation.z,
    ) = T[:3, 3]
    q = matrix4_to_quat_xyzw(T)
    (
        msg.transform.rotation.x,
        msg.transform.rotation.y,
        msg.transform.rotation.z,
        msg.transform.rotation.w,
    ) = q
    return msg


class StereoAnchorNode(Node):
    """
    Publishes global TFs for a set of floaters using an anchor frame and live poses.

    Chain used for each floater:
        world -> anchor   (from TF)
        anchor -> atf      (PoseStamped on self.cfg.atf)
        atf -> floater     (PoseStamped on each topic in self.cfg.ftf)

    The product gives world -> floater. That TF is broadcast atf 30 Hz.
    """

    def __init__(self, cfg: StereoConfig | None = None):
        super().__init__("stereo_anchor_node")

        # load params into config object
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = StereoConfig(
                anchor=self.declare_parameter("anchor", "anchor").value,
                floaters=self.declare_parameter("floaters", ["cam_low", "cam_side"]).value,
                atf=self.declare_parameter("atf", "/anchor_tracker/pose").value,
                ftf=self.declare_parameter("ftf", ["/cam_low/pose", "/cam_side/pose"]).value,
                world=self.declare_parameter("world", "world").value,
            )

        print(self.cfg)

        # check floaters and ftf lists are same length
        if len(self.cfg.floaters) != len(self.cfg.ftf):
            self.get_logger().error("floaters and ftf lists must be same length")
            raise RuntimeError("StereoConfig mismatch")

        # TF interfaces. Buffer stores TF tree, listener fills it, broadcaster publishes new TFs.
        self.tfbuf = Buffer()
        self.lst = TransformListener(self.tfbuf, self)
        self.bst = TransformBroadcaster(self)

        # latest messages on the pose topics
        # anchor -> atf pose
        self.latest_ap: PoseStamped | None = None
        # atf -> floater poses by floater name
        # self.latest_fp: Dict[str, PoseStamped] = {}

        # subscriptions for the live poses
        # self.create_subscription(PoseStamped, self.cfg.atf, self._on_ap, 10)
        # for floater, topic in zip(self.cfg.floaters, self.cfg.ftf):
        # self.create_subscription(PoseStamped, topic, self._make_fp_cb(floater), 10)

        self.d = Path().home().resolve() / ".xnodes" / "t_world_cam"
        self.d.mkdir(parents=True, exist_ok=True)

        # loop at 30 Hz
        self.timer = self.create_timer(1.0 / 30.0, self._tick)

    # def _on_ap(self, msg: PoseStamped):
    # """Callback for the anchor-tracker pose: anchor -> atf."""
    # self.get_logger().debug(f"Received anchor->atf pose at {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
    # self.latest_ap = msg

    def _make_fp_cb(self, floater_name: str):
        """Factory that returns a callback which stores at -> floater for a given floater."""

        def cb(msg: PoseStamped):
            self.get_logger().debug(
                f"Received atf->{floater_name} pose at {msg.header.stamp.sec}.{msg.header.stamp.nanosec}"
            )
            self.latest_fp[floater_name] = msg

        return cb

    def safe_lookup(self, target, source, timeout=0.005) -> TransformStamped | None:
        """
        Attempt to get target ← source transform.
        On failure: log warning once per attempt, return False.
        """
        try:
            t: TransformStamped = self.tfbuf.lookup_transform(
                target_frame=target,
                source_frame=source,
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout),
            )
            return t

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed ({source} -> {target}): {e}")
            return None

    def _tick(self):
        """Periodic step. Compute world -> floater for each floater and publish as TF."""
        # need anchor -> atf before we can compute anything

        # if self.latest_ap is None:
        # return

        # use the node clock for TF timestamps
        stamp = self.get_clock().now().to_msg()

        # get world -> anchor from the TF tree
        t_anchor: TransformStamped = self.safe_lookup(self.cfg.world, self.cfg.anchor)
        ap = self.safe_lookup(self.cfg.anchor, self.cfg.atf)
        # convert anchor -> atf to matrix
        if any(v is None for v in [t_anchor, ap]):
            return

        T_anchor_ap = tf2mat(ap)
        print(T_anchor_ap)

        print({floater: (floater, topic) for floater, topic in zip(self.cfg.floaters, self.cfg.ftf)})

        self.fps = {
            floater: self.safe_lookup(floater, topic) for floater, topic in zip(self.cfg.floaters, self.cfg.ftf)
        }
        print(self.fps)

        if any(v is None for v in self.fps.values()):
            return

        T_world_anchor = tf2mat(t_anchor)

        # for each floater we have a recent atf -> floater pose
        for floater, fp_pose in self.fps.items():
            T_float_fp = tf2mat(fp_pose)

            Rz = matrix_from_euler([0.0, 0.0, np.pi], 0, 1, 2, extrinsic=False)  # 180° about Z
            Tz = np.eye(4, dtype=float)
            Tz[:3, :3] = Rz

            # chain transforms: world->floater = (world->anchor) * (anchor->atf) * (atf->floater)
            print("all")
            inv = np.linalg.inv
            T_world_fp = T_world_anchor @ T_anchor_ap @ inv(T_float_fp)  # @ Tz)
            print(T_world_fp)

            # publish TF for this floater
            self.save_to_file(floater, T_world_fp)
            tf_msg = mat2tf(T_world_fp, self.cfg.world, floater, stamp)
            self.bst.sendTransform(tf_msg)

    def save_to_file(self, name: str, mat: np.ndarray):
        t = mat[:3, 3]
        R_ = mat[:3, :3]
        quat = R.from_matrix(R_).as_quat()  # x y z w

        data = {
            "T_world_cam": mat.tolist(),
            "R": R_.tolist(),
            "t": t.tolist(),
            "quat_xyzw": quat.tolist(),
        }
        file = self.d / f"{name}.yaml"
        with open(str(file), "w") as f:
            yaml.safe_dump(data, f)


def run():
    """Entrypoint. Initialize ROS, spin the node, then shut down."""
    rclpy.init()
    node = StereoAnchorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def main(cfg: StereoConfig):
    """convenient entry python file.py"""
    rclpy.init()
    node = StereoAnchorNode(cfg)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(tyro.cli(StereoConfig))
