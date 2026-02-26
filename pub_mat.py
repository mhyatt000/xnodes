#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib

from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

try:
    import tf_transformations as tft  # ROS tf quaternion utils
except Exception as e:
    raise RuntimeError("pip/rosdep install 'tf_transformations'") from e


def Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])


def Rz(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])


def _hom(R):
    T = np.eye(4)
    T[:3, :3] = R
    return T


# Fixed conversions (REP-105)
_R_link_to_opt = Rx(-np.pi / 2) @ Rz(-np.pi / 2)  # camera_link → camera_optical_frame
_R_opt_to_link = _R_link_to_opt.T  # camera_optical_frame → camera_link
_Rz = Rz(-np.pi)


def ocv2tflink(T_world_opt: np.ndarray) -> np.ndarray:
    """
    Convert world→camera_optical_frame pose to world→camera_link.
    Only rotation changes. Translation is unchanged.
    """
    return T_world_opt @ _hom(_R_opt_to_link)


def tflink2ocv(T_world_link: np.ndarray) -> np.ndarray:
    """
    Convert world→camera_link pose to world→camera_optical_frame.
    """
    return T_world_link @ _hom(_R_link_to_opt)


def rot_ocv2tflink(R_opt: np.ndarray) -> np.ndarray:
    return R_opt @ _R_opt_to_link


def rot_tflink2ocv(R_link: np.ndarray) -> np.ndarray:
    return R_link @ _R_link_to_opt


# ---------------------- config ----------------------
@dataclass
class Config:
    parent: str = "world"
    child: str = "pub_mat"
    rate_hz: float = 10.0

    # One of: path OR values. If both None -> error.
    path: str | None = None  # .npy / .npz / .json / .yaml
    npz_key: str | None = None  # key for .npz
    values: list[float] | None = None  # 16 floats row-major

    invert: bool = False  # publish inverse(T)
    transpose: bool = False  # transpose the 3x3 if your source is col-major
    check_det: bool = True  # sanity check R determinant ~ +1
    quiet: bool = False


# ---------------------- loader ----------------------
def load_matrix(cfg: Config) -> np.ndarray:
    if cfg.values is not None:
        vals = np.asarray(cfg.values, dtype=float)
        if vals.size != 16:
            raise ValueError("--values must contain 16 numbers")
        T = vals.reshape(4, 4)
    elif cfg.path:
        p = pathlib.Path(cfg.path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() == ".npy":
            T = np.load(p)
        elif p.suffix.lower() == ".npz":
            z = np.load(p)
            key = cfg.npz_key or next(iter(z.keys()))
            T = z[key]
        elif p.suffix.lower() in (".json",):
            with open(p, "r") as f:
                obj = json.load(f)
            # Try common keys
            T = np.array(obj.get("T") or obj.get("matrix") or obj, dtype=float)
        elif p.suffix.lower() in (".yml", ".yaml"):
            import yaml  # pip install pyyaml

            with open(p, "r") as f:
                obj = yaml.safe_load(f)
            T = np.array(obj.get("T") or obj.get("matrix") or obj, dtype=float)
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")
    else:
        raise ValueError("Provide --path or --values")

    # Accept 3x4, 4x4
    if T.shape == (3, 4):
        T = np.vstack([T, [0, 0, 0, 1]])
    if T.shape != (4, 4):
        raise ValueError(f"Matrix must be 4x4 (or 3x4). Got {T.shape}")

    R = T[:3, :3].copy()
    t = T[:3, 3].copy()

    if cfg.transpose:
        R = R.T
        T[:3, :3] = R

    if cfg.check_det:
        det = float(np.linalg.det(R))
        if not (0.9 <= det <= 1.1):
            raise ValueError(f"det(R) ≈ {det:.3f} not near +1 (bad rotation?)")

    T = ocv2tflink(T)
    T = T @ _hom(_Rz)
    if cfg.invert:
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        T = T_inv

    return T


# ---------------------- node ----------------------
class MatrixTF(Node):
    def __init__(self, cfg: Config, T: np.ndarray):
        super().__init__("matrix_tf_pub")
        self.cfg = cfg
        self.T = T
        self.bst = TransformBroadcaster(self)
        self.timer = self.create_timer(1.0 / cfg.rate_hz, self._tick)
        if not cfg.quiet:
            R, t = T[:3, :3], T[:3, 3]
            det = float(np.linalg.det(R))
            self.get_logger().info(
                f"Publishing {cfg.parent} → {cfg.child} @ {cfg.rate_hz}Hz | t={t.tolist()} | det(R)={det:.6f}"
            )

    def _tick(self):
        tf = TransformStamped()
        # tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.cfg.parent
        tf.child_frame_id = self.cfg.child

        t = self.T[:3, 3]
        tf.transform.translation.x = float(t[0])
        tf.transform.translation.y = float(t[1])
        tf.transform.translation.z = float(t[2])

        qx, qy, qz, qw = tft.quaternion_from_matrix(self.T)
        tf.transform.rotation.x = float(qx)
        tf.transform.rotation.y = float(qy)
        tf.transform.rotation.z = float(qz)
        tf.transform.rotation.w = float(qw)

        print({"t": {"x": t[0], "y": t[1], "z": t[2]}, "q": {"x": qx, "y": qy, "z": qz, "w": qw}})
        self.bst.sendTransform(tf)


# ---------------------- main ----------------------
if __name__ == "__main__":
    # Parse CLI
    try:
        import tyro  # pip install tyro
    except Exception as e:
        raise RuntimeError("pip install tyro") from e

    cfg = tyro.cli(Config, description="Publish a TF from a 4x4 matrix")

    T = load_matrix(cfg)

    rclpy.init()
    node = MatrixTF(cfg, T)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
