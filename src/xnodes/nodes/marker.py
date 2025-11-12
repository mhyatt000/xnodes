from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import time
import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
import tyro
import yaml

def build_marker(points_xyz: np.ndarray, frame: str, scale: float,
                 rgba: Tuple[float, float, float, float],
                 ns: str = "points", mid: int = 0) -> MarkerArray:
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3, "points must be (n,3)"
    m = Marker()
    m.header.frame_id = frame
    m.ns = ns
    m.id = mid
    m.type = Marker.SPHERE_LIST
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = float(scale)
    m.color.r, m.color.g, m.color.b, m.color.a = map(float, rgba)
    m.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in points_xyz]
    arr = MarkerArray()
    arr.markers.append(m)
    return arr

def load_points_from_yaml(path: Path) -> Tuple[np.ndarray, Optional[dict]]:
    """Returns (points[n,3], meta) where meta may include frame/scale/color from file."""
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    meta = {}
    if "points" in y:
        pts_raw: List = y["points"]
        if isinstance(pts_raw[0], dict):
            pts = np.array([[p["x"], p["y"], p["z"]] for p in pts_raw], dtype=float)
        else:
            pts = np.array(pts_raw, dtype=float)
        return pts, meta
    if "markers" in y and isinstance(y["markers"], list) and y["markers"]:
        m0 = y["markers"][0]
        pts_raw = m0.get("points", [])
        if not pts_raw:
            raise ValueError("YAML markers[0] has no points")
        pts = np.array([[p["x"], p["y"], p["z"]] for p in pts_raw], dtype=float)
        # Optional meta
        hdr = m0.get("header", {})
        if "frame_id" in hdr:
            meta["frame"] = hdr["frame_id"]
        if "scale" in m0:
            s = m0["scale"]
            # accept either uniform or x-only
            meta["scale"] = float(s["x"]) if isinstance(s, dict) else float(s)
        if "color" in m0:
            c = m0["color"]
            meta["rgba"] = (float(c["r"]), float(c["g"]), float(c["b"]), float(c["a"]))
        return pts, meta
    raise ValueError("YAML must contain either 'points' or 'markers' with 'points'")

@dataclass
class Config:
    # Input
    source: Optional[str] = None                 # topic to listen for Float64MultiArray (n*3)
    file: Optional[Path] = None                  # YAML file. If provided and source=None, publish these points.
    # Output
    target: Optional[str] = None                 # MarkerArray topic. Defaults based on source.
    frame: str = "world"
    # Viz
    scale: float = 0.03
    rgba: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    ns: str = "points"
    mid: int = 0
    # Random mode (used when source=None and file=None)
    n: int = 50
    bounds: Tuple[float, float] = (-0.25, 0.25)    # uniform range per axis
    hz: float = 5.0 # publish rate
    once: bool = False                           # if True, publish once then exit
    seed: Optional[int] = 0

class MarkerCLINode(Node):
    def __init__(self, cfg: Config):
        super().__init__("marker_cli")
        target = cfg.target or (f"/{cfg.source}/markers" if cfg.source else "visualization_marker_array")
        self.pub = self.create_publisher(MarkerArray, target, 10)
        self.cfg = cfg

        if cfg.source:
            self.sub = self.create_subscription(Float64MultiArray, cfg.source, self._cb, 10)
            self.get_logger().info(f"listen='{cfg.source}' → publish='{target}' frame='{cfg.frame}'")
        else:
            # immediate publish path (file or random)
            if cfg.file:
                pts, meta = load_points_from_yaml(cfg.file)
                frame = meta.get("frame", cfg.frame)
                scale = meta.get("scale", cfg.scale)
                rgba  = meta.get("rgba",  cfg.rgba)
                msg = build_marker(pts, frame, scale, rgba, cfg.ns, cfg.mid)
                self.pub.publish(msg)
                self.get_logger().info(f"published {len(pts)} points from '{cfg.file}' to '{target}' frame='{frame}'")
                if cfg.once:
                    # exit quickly
                    rclpy.shutdown()
                    return
            else:
                # random generator timer
                if cfg.seed is not None:
                    np.random.seed(cfg.seed)
                lo, hi = cfg.bounds
                self.timer = self.create_timer(max(1e-3, 1.0 / max(1e-6, cfg.hz)), lambda: self._tick_random(lo, hi))
                if cfg.once:
                    # publish once on first timer tick then shutdown
                    # small delay to allow publisher to connect
                    self.create_timer(0.05, self._publish_once_and_exit)

    def _cb(self, msg: Float64MultiArray) -> None:
        data = np.asarray(msg.data, dtype=float)
        if data.size % 3 != 0:
            self.get_logger().warn(f"len={data.size} not divisible by 3. drop.")
            return
        pts = data.reshape(-1, 3)
        out = build_marker(pts, self.cfg.frame, self.cfg.scale, self.cfg.rgba, self.cfg.ns, self.cfg.mid)
        self.pub.publish(out)

    def _tick_random(self, lo: float, hi: float) -> None:
        pts = np.random.uniform(lo, hi, size=(self.cfg.n, 3)).astype(float)
        out = build_marker(pts, self.cfg.frame, self.cfg.scale, self.cfg.rgba, self.cfg.ns, self.cfg.mid)
        self.pub.publish(out)

    def _publish_once_and_exit(self) -> None:
        # Trigger one random publish then exit
        self._tick_random(self.cfg.bounds[0], self.cfg.bounds[1])
        # give RViz a moment to receive
        time.sleep(0.05)
        rclpy.shutdown()

def main(cfg: Config) -> None:
    rclpy.init()
    try:
        node = MarkerCLINode(cfg)
        rclpy.spin(node)
    finally:
        # shutdown might already be called
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main(tyro.cli(Config))

