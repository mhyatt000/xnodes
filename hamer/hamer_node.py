#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cv_bridge import CvBridge

# import  hamer_node_helper as hh
import jax
from mano_pipe_v3 import remap_keys, select_keys
import numpy as np
import rclpy
from rclpy.node import Node
from rich.pretty import pprint
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import tyro
from webpolicy.webpolicy.deploy.client import WebsocketClientPolicy
from xgym.rlds.util import (
    add_col,
    apply_persp,
    perspective_projection,
    remove_col,
)
import yaml

"""
{
│   'box_center': (2, 2),
│   'box_size': (2,),
│   'focal_length': (2, 2),
│   'img': (240, 320, 3),
│   'img_size': (2, 2),
│   'img_wrist': (2, 256, 256, 3),
│   'personid': (2,),
│   'pred_cam': (2, 3),
│   'pred_cam_t': (2, 3),
│   'pred_cam_t_full': (2, 3),
│   'pred_keypoints_2d': (2, 21, 2),
│   'pred_keypoints_3d': (2, 21, 3),
│   'pred_mano_params.betas': (2, 10),
│   'pred_mano_params.global_orient': (2, 1, 3, 3),
│   'pred_mano_params.hand_pose': (2, 15, 3, 3),
│   'pred_vertices': (2, 778, 3),
│   'right': (2,),
│   'scaled_focal_length': ()
}
"""


# 4x4 matx
T_xflip = np.array(
    [
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def spec(thing: dict[str, np.ndarray]):
    """Returns the shape of each key in the dict."""
    return jax.tree.map(lambda x: x.shape, thing)


def vsm_mul(xs, mat: np.ndarray) -> dict[str, np.ndarray]:
    """Applies a 4x4 matrix to a batch of 3D points in xs."""
    xs = add_col(xs)
    pprint((xs.shape, mat.shape))
    xs = remove_col(xs @ mat)[None]
    return xs


from scipy.spatial.transform import Rotation as R


def calib2mat(myeob: Path = Path().home() / "eob.calib"):
    """Converts a calibration dict to a 4x4 matrix."""

    with open(myeob, "r") as f:
        myeob = yaml.safe_load(f)

    t = myeob["transform"]["translation"]
    t = np.array([t["x"], t["y"], t["z"]])
    r = myeob["transform"]["rotation"]
    r = np.array(
        R.from_quat(
            [r["x"], r["y"], r["z"], r["w"]]
            # scalar_first=True
        ).as_matrix()
    )

    T = np.eye(4)
    T[:3, 3] = t
    # T[:3, :3] = np.linalg.inv(r)
    _r = np.eye(4)
    pprint(r.shape)
    T[:3, :3] = r[:3, :3]

    return T  # ,_r


class HamerNode(Node):
    def __init__(self, cfg):
        super().__init__("hamer_image_processor")
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, "/image_raw", self.listener_callback, 10)
        self.a = self.create_publisher(Float32MultiArray, "hand/a", 50)
        self.b = self.create_publisher(Float32MultiArray, "hand/b", 50)
        self.timer = self.create_timer(0.05, self.publish_points)  # 50 Hz

        # cfg = Config(host='localhost', port=8002,src=hh.Camera())
        self.client = WebsocketClientPolicy(host="carina.cs.luc.edu", port=8002)

        self.cfg = cfg
        self.points = None
        self.P = []
        if Path("points.yaml").exists():
            with open("points.yaml", "r") as f:
                data = yaml.safe_load(f)
                self.P = [np.array(p) for p in data["points"]]

        self.T = calib2mat()
        self.Tinv = np.linalg.inv(self.T)

        self.get_logger().info("HamerNode node started")

    def publish_points(self):
        for i, x in enumerate([self.a, self.b]):
            if self.points is not None:
                pts = self.points[min(i, len(self.points) - 1)]
                print(f"selected pts {i}/{len(self.points) - 1}")
            else:
                pts = np.random.uniform(-0.2, 0.2, (21, 3)).astype(np.float32)
            if pts.shape != (21, 3):
                return

            msg = Float32MultiArray()
            msg.data = pts.flatten().tolist()

            print("Publishing hand points to", x.topic)
            x.publish(msg)

    def fwd(self, frame):
        pack = {"img": frame, "fx": self.cfg.fx}
        out = self.client.infer(pack)
        if not out:
            self.points = None
            return {}

        out = self.postprocess(out, frame)

        k = "keypoints_3d"
        pprint(out[k].shape)
        for i in range(len(out[k])):
            # out[k][i] = vsm_mul( out[k][i], self.r)
            out[k][i] = vsm_mul(out[k][i], self.Tinv)

        # pick 0-n randomly
        n = len(out["keypoints_3d"])
        idx = np.random.randint(0, n)
        self.points = out["keypoints_3d"]

        pprint(self.points.shape)
        # # pprint(self.points)

        # self.P.append(self.points)

        #  with open('points.yaml', 'w') as f:
        #  yaml.safe_dump({'points': [p.tolist() for p in self.P]}, f)

        # self.points = out['pred_keypoints_3d'][0]
        return out

    def postprocess(self, out: dict[str, np.ndarray], frame) -> dict[str, np.ndarray]:
        out = jax.tree.map(lambda x: x.copy(), out)

        # pprint({'kp3d': out["pred_keypoints_3d"][0][0]})

        right = out["right"].astype(bool)
        left = ~right

        print(left)

        cam_t_full = out["pred_cam_t_full"][0]

        rot = out["pred_mano_params.global_orient"][0].reshape(3, 3)
        out = remap_keys(out)

        pprint((out["keypoints_3d"].shape, out["keypoints_3d"].dtype))
        kp3d = []
        for i in range(len(left)):
            n = len(out["keypoints_3d"])
            print(n)
            if left[i]:
                k = add_col(out["keypoints_3d"][i])
                k = remove_col((k @ T_xflip)[None])[0]
                pprint(k.shape)
            else:
                k = out["keypoints_3d"][i]
            kp3d.append(k)
        out["keypoints_3d"] = np.array(kp3d)
        pprint((out["keypoints_3d"].shape, out["keypoints_3d"].dtype))
        pprint((out["cam_t_full"].shape, out["cam_t_full"].dtype))

        imwrist = out["img_wrist"][0]
        # f = out["focal_length"][0][0]
        out = select_keys(out)
        f = out["scaled_focal_length"]

        # pprint({'kp3d': out["keypoints_3d"][0][0]})

        # pprint(rot)

        # pprint({'scaled_focal_length': f})
        P = perspective_projection(f, H=frame.shape[0], W=frame.shape[1])
        # P[:3, :3] = cal.intr.mat
        points2d = apply_persp(out["keypoints_3d"], P)[0, :, :-1]
        return out

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        pprint(frame.shape)
        out = self.fwd(frame)
        pprint(spec(out))
        pprint(out.keys())
        return
        # out_msg = self.bridge.cv2_to_imgmsg(out_img, encoding="bgr8")
        self.publisher.publish(out_msg)


@dataclass
class Config:
    host: str = "carina.cs.luc.edu"
    port: int = 8002

    fx: float | None = None  # override focal length


def main(cfg: Config):
    rclpy.init(args=None)
    node = HamerNode(cfg)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(tyro.cli(Config))
