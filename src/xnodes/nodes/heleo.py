"""
HELEO [H]and T[ELEO]perator node.
Reads input from the HaMeR server and publishes to /robot_commands.
"""

from __future__ import annotations

from collections import deque
import threading
import time

import cv2
import jax
import numpy as np
from webpolicy.client import Client
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from xarm_msgs.msg import RobotMsg

# TODO fix import
from xgym.calibrate.urdf.robot import RobotTree
from xgym.rlds.util import add_col, apply_persp, perspective_projection, remove_col

from xnodes.nodes.base import Base

def remap_keys(out: dict):
    out = {k.replace("pred_", "").replace("_params", ""): v for k, v in out.items()}
    return out


def select_keys(out: dict):
    """Selects keys to keep in the output.
    as a side effect, prepares the keypoints_3d for further processing
    TODO break into separate func?
    """

    out["keypoints_3d"] += out.pop("cam_t_full")[:, None, :]

    keep = [
        # "box_center",
        # "box_size",
        "img",
        "img_wrist",
        # "img_size",
        # "personid",
        # "cam",
        # "cam_t",
        # "cam_t_full", # used above
        "keypoints_2d",  # these are wrt the box not full img ... solve for 2d
        "keypoints_3d",
        "mano.betas",
        "mano.global_orient",
        "mano.hand_pose",
        # "vertices",
        "right",
        # "focal_length", # i think we use the scaled one instead
        "scaled_focal_length",
    ]

    return {k: out[k] for k in keep if k in out}


class MyClient(Client):
    def reset(self):
        self.infer({"reset": True})


class MyCamera:
    def __repr__(self) -> str:
        return f"MyCamera(device_id={'TODO'})"

    def __init__(self, cam: int | cv2.VideoCapture):
        self.cam = cv2.VideoCapture(cam) if isinstance(cam, int) else cam
        self.thread = None

        self.fps = 30
        self.dt = 1 / self.fps

        # while
        # _, self.img = self.cam.read()
        self.start()
        time.sleep(1)

    def start(self):
        self._recording = True

        def _record():
            while self._recording:
                tick = time.time()
                # ret, self.img = self.cam.read()
                self.cam.grab()

                toc = time.time()
                elapsed = toc - tick
                time.sleep(max(0, self.dt - elapsed))

        self.thread = threading.Thread(target=_record, daemon=True)
        self.thread.start()

    def stop(self):
        self._recording = False
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to finish
            del self.thread

    def read(self):
        ret, img = self.cam.retrieve()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ret, img


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


class Heleo(Base):
    """
    [H]and T[ELEO]perator node.
    Reads input from the HaMeR server and publishes to /robot_commands.
    """

    def __init__(self, cfg):
        super().__init__("controller_node")

        self.publisher = self.create_publisher(Float32MultiArray, "/robot_commands", 10)
        self.cam = MyCamera(3)  # fix
        self.client = MyClient(host=cfg.host, port=cfg.port)
        self.urdf = RobotTree()

        self.joints = None
        self.pose = None
        self.action = None
        self.target = None

        self.smooth = 10
        self.palms = deque(maxlen=self.smooth)

        from xgym import BASE

        self.cam2base = np.load(BASE / "base.npz")["arr_0"]

        self.hz = 10
        self.ctrl_hz = 50
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._command_loop, daemon=True)
        self._thread.start()

        self.subs = {
            "joints": self.create_subscription(JointState, "/xarm/joint_states", self.set_joints, 10),
            "pose": self.create_subscription(RobotMsg, "/xarm/robot_states", self.set_pose, 10),
        }

        self.act_timer = self.create_timer(1 / self.ctrl_hz, self.control)

        self.get_logger().info("Controller Node Initialized.")

    def _command_loop(self):
        rate = 1.0 / self.hz
        while not self._stop_event.is_set():
            tic = time.time()
            self.step()
            toc = time.time()
            time.sleep(max(0, rate - (toc - tic)))

    def destroy_node(self):
        self._stop_event.set()
        self._thread.join()
        super().destroy_node()

    def step(self):
        """Reads SpaceMouse input and publishes it."""

        if self.joints is None or self.pose is None:
            return

        if not self.active:
            action = np.zeros(7)
            msg = Float32MultiArray()
            msg.data = action.tolist()
            self.publisher.publish(msg)
            return

        try:
            ret, frame = self.cam.read()
            assert ret and frame is not None, "Failed to read frame from camera"

            pack = {"img": frame}
            out = self.client.infer(pack)

            if out is not None:
                out = jax.tree.map(lambda x: x.copy(), out)

                right = bool(out["right"][0])
                left = not right

                cam_t_full = out["pred_cam_t_full"][0]

                rot = out["pred_mano_params.global_orient"][0].reshape(3, 3)
                out = remap_keys(out)

                if left:
                    n = len(out["keypoints_3d"])
                    kp3d = add_col(out["keypoints_3d"])
                    kp3d = remove_col((kp3d[0] @ T_xflip)[None])
                    out["keypoints_3d"] = np.concatenate([kp3d for _ in range(n)])

                imwrist = out["img_wrist"][0]
                out = select_keys(out)

                # pprint(rot)

                f = out["scaled_focal_length"]
                P = perspective_projection(f, H=frame.shape[0], W=frame.shape[1])
                points2d = apply_persp(out["keypoints_3d"], P)[0, :, :-1]

                palm = points2d[0]
                palm3d = out["keypoints_3d"][0][0]

        except Exception as e:
            self.get_logger().error(f"Error during inference: {e}")
            return

        # grippiness is mean distance of all fingers
        # 21 joints... 0 is palm. openpose

        finger_idxs = [
            1,
            2,
            3,
            4,  # thumb
            5,
            6,
            7,
            8,  # index
            9,
            10,
            11,
            12,  # middle
            13,
            14,
            15,
            16,  # ring
            17,
            18,
            19,
            20,
        ]
        finger_idxs = [4, 8, 12, 16, 20]
        pairs = [(4, f) for f in finger_idxs[1:]]
        kp = out["keypoints_3d"][0]
        pairs = [np.array(kp[p1] - kp[p2]) for p1, p2 in pairs]
        pairs = [np.linalg.norm(p) for p in pairs]
        grip = np.mean(pairs)

        # map 0.12,0.02 to 1,0
        dist = 0.12 - 0.02
        grip = np.clip((grip - 0.02) / dist, 0, 1)
        # pprint(f"Grippiness: {grip}")

        if self.joints is None:
            return
        kin = self.urdf.set_pose(self.joints, end_only=True).get_matrix()[0]
        kin = self.cam2base @ kin.numpy()
        tvec = kin[:3, 3]
        # pprint(f"tvec: {tvec}")

        thome = np.array([0.107, 0.050, 1.207])
        tvec = tvec - thome
        # pprint(f"tvec: {tvec}")

        self.palms.append(palm3d)
        # print(f"palms len: {len(self.palms)}")
        if len(self.palms) > self.smooth:
            return
        palm3d = np.mean(self.palms, axis=0)

        # pprint(f"palm3d: {palm3d}")

        def min_max_scale(_x, min_val=None, max_val=None):
            """Scales input _x to [0, 1] range using min-max scaling."""
            _x = np.asarray(_x)
            if min_val is None:
                min_val = _x.min()
            if max_val is None:
                max_val = _x.max()
            return (_x - min_val) / (max_val - min_val + 1e-8)

        pbounds = {
            "x": (np.float64(-0.5), np.float64(0.5)),  # left to right
            "y": (np.float64(-0.5), np.float64(0.5)),  # bottom to top
            "z": (np.float64(0.0), np.float64(1.5)),  # cam space
        }
        palm3d = np.array(
            [min_max_scale(item, *(mm)) for item, mm in zip(palm3d, [pbounds["x"], pbounds["y"], pbounds["z"]])]
        )

        # pprint(f"palm3d scaled: {palm3d}")
        T = np.array(
            [
                [0, 1, 0, 0],  # x to -y
                [0, 0, 1, 0],  # y to -z
                [1, 0, 0, 0],  # z to -x
                [0, 0, 0, 1],
            ]
        )
        palm3d = (add_col(palm3d) @ T)[:-1]
        # pprint(f"palm3d tform: {palm3d}")

        palm3d[1] = 1 - np.clip(palm3d[1], 0, 1)
        palm3d[2] = 1 - np.clip(palm3d[2], 0, 1)
        bounds = {
            "min": np.array([200, -200, 30]),
            "max": np.array([560, 200, 250]),
        }

        # unscale to bounds

        def min_max_unscale(_x, a, b):
            """Reverses min-max scaling, mapping [0, 1] values back to original range."""
            _x = np.clip(np.asarray(_x), 0, 1)
            return (_x * (b - a)) + min(a, b)

        palm3d = np.array([min_max_unscale(item, a, b) for item, a, b in zip(palm3d, bounds["min"], bounds["max"])])

        # print(f"palm3d unscaled: {palm3d}")

        action = np.zeros(7)
        action[:3] = palm3d
        action[-1] = (2 * (grip - 0.8)) * 50
        self.target = action

    def control(self):
        if self.target is None:
            return
        # action = self.target.copy()
        # action[:3] -= self.pose[:3]

        msg = Float32MultiArray()
        msg.data = self.target.tolist()
        self.publisher.publish(msg)

    def set_pose(self, msg: RobotMsg):
        """
        RobotMsg message has:
            header: std_msgs/Header header
            pose: List[float64]
            ...others...
        """
        self.pose = np.array(msg.pose).astype(np.float32)

    def set_joints(self, msg: JointState):
        if len(msg.position) == 6:
            return
        self.joints = np.array(msg.position)
