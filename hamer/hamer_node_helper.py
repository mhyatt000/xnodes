from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import jax

# from get_calibration import MyCamera
from mano_pipe_v3 import remap_keys, select_keys
import matplotlib.pyplot as plt
import numpy as np
from rich.pretty import pprint
import tyro
from webpolicy.deploy.client import WebsocketClientPolicy
from xgym.calibrate.april import Calibrator
from xgym.rlds.util import (
    add_col,
    apply_persp,
    perspective_projection,
    remove_col,
)

np.set_printoptions(suppress=True, precision=3)


def spec(thing: dict[str, np.ndarray]):
    """Returns the shape of each key in the dict."""
    return jax.tree.map(lambda x: x.shape, thing)


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


class NPZVideoReader:
    def __init__(self, dir: str):
        self.dir = dir
        self.files = list(Path(dir).rglob("ep*.npz"))
        self._load_next_episode()

    def _load_next_episode(self):
        if not self.files:
            self.frames = None
            return
        f = self.files.pop(0)
        ep = np.load(f, allow_pickle=True)
        ep = {k: ep[k] for k in ep}
        pprint(spec(ep))  # assuming spec() is defined elsewhere
        self.frames = next(iter(ep.keys()))
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.frames is None:
            raise StopIteration
        if self.idx >= self.frames.shape[0]:
            self._load_next_episode()
            return self.__next__()
        frame = self.frames[self.idx]
        self.idx += 1
        return True, frame

    def read(self):
        return self.__next__()


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


@dataclass
class Source:
    pass


@dataclass
class Camera(Source):
    idx: int = 3  # the camera idx to use


@dataclass
class File(Source):
    dir: str

    # typ: Literal["mp4", "npz"] = "npz" # the type of video file


@dataclass
class Config:
    host: str
    port: int

    src: Camera | File
    # cam: int = 3  # the camera idx to use
    # mode: Mode = Mode.CAM  # the mode to use, either CAM or FILE


def main(cfg: Config):
    pprint(cfg)

    client = WebsocketClientPolicy(host=cfg.host, port=cfg.port)

    # cv2 video file reader
    # cap = cv2.VideoCapture(str(Path().home() / 'hand.mp4'))

    match cfg.src:
        case Camera():
            cap = cv2.VideoCapture(cfg.src.idx)
            cam = MyCamera(cap)
        case File():
            cam = NPZVideoReader(cfg.src.dir)

    cal = Calibrator()
    # cam2base = np.load(BASE / "base.npz")["arr_0"]

    elapses = deque(maxlen=50)
    hist = deque(maxlen=10)
    base_hist = []
    while True:
        _ret, frame = cam.read()

        # import xgym.utils.camera as cu
        # frame = cu.square(frame)
        # frame = cv2.resize(frame, (224, 224))
        # frame = cv2.flip(frame, 1)

        pack = {"img": frame}

        tic = time.time()

        out = client.infer(pack)

        toc = time.time()
        print("infer time", elapse := toc - tic)
        elapses.append(elapse)
        print(
            "fps",
            round(1 / np.mean(elapses), 2),
            "ms",
            round(1000 * np.mean(elapses), 2),
        )

        if out is not None:
            out = jax.tree.map(lambda x: x.copy(), out)
            # pprint(spec(out))

            box = {
                "center": out["box_center"][0],
                "size": out["box_size"][0],
            }

            # pprint({'kp3d': out["pred_keypoints_3d"][0][0]})

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
            # pprint({'kp3d': out["keypoints_3d"][0][0]})

            # pprint(rot)

            f = out["scaled_focal_length"]
            # pprint({'scaled_focal_length': f})
            P = perspective_projection(f, H=frame.shape[0], W=frame.shape[1])
            # P[:3, :3] = cal.intr.mat
            points2d = apply_persp(out["keypoints_3d"], P)[0, :, :-1]

            _origin = out["keypoints_3d"][0][0]
            _xax = _origin + rot[:, 0] * 0.1
            _yax = _origin + rot[:, 1] * 0.1
            _zax = _origin + rot[:, 2] * 0.1
            axes = np.array([[_origin, _xax, _yax, _zax]])
            # pprint({"axes": axes.shape})
            axes = apply_persp(axes, P)[0, :, :-1]

            # out = solve_2d(
            # {
            # "img": out["img"],
            # "scaled_focal_length": out["scaled_focal_length"],
            # "keypoints_3d": out["keypoints_3d"][0],
            # }
            # )

            squeeze = lambda arr: jax.tree.map(lambda x: x.squeeze(), arr)
            pprint(spec(squeeze(out)))
            # quit()

            #
            # TODO there is something weird
            # with solve_2d versus apply_persp
            # maybe because it crops
            #

            palm = points2d[0]
            palm3d = out["keypoints_3d"][0][0]
            print(palm3d)

            cvtype = lambda thing: tuple(thing.astype(int))
            _o, _x, _y, _z = axes

            # negative axes
            _dx, _dy, _dz = _x - _o, _y - _o, _z - _o
            _nx, _ny, _nz = _o - _dx, _o - _dy, _o - _dz

            _o, _x, _y, _z = map(cvtype, (_o, _x, _y, _z))
            _nx, _ny, _nz = map(cvtype, (_nx, _ny, _nz))

            red, green, blue = (255, 0, 0), (0, 255, 0), (0, 0, 255)
            black, gray = (0, 0, 0), (128, 128, 128)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.line(frame, _o, _x, red, 2)  # X axis - Red
            # cv2.line(frame, _o, _nx, red, 2)  # X axis - Red
            cv2.putText(frame, "X", (_x[0] + 10, _x[1]), font, 0.5, red, 2)
            cv2.line(frame, _o, _y, green, 2)
            # cv2.line(frame, _o, _ny, green, 2)
            cv2.putText(frame, "Y", (_y[0] + 10, _y[1]), font, 0.5, green, 2)
            cv2.line(frame, _o, _z, blue, 2)
            # cv2.line(frame, _o, _nz, blue, 2)
            cv2.putText(frame, "Z", (_z[0] + 10, _z[1]), font, 0.5, blue, 2)

            # cal.intr.fx = P[0, 0]
            # cal.intr.fy = P[1, 1]
            # cal.intr.cx = P[0, 2]
            # cal.intr.cy = P[1, 2]
            """
            rvec, _ = cv2.Rodrigues(rot[:3, :3])
            pprint({"rvec": rvec})
            cal.draw_pose_axes(
                frame,
                rvec=np.zeros_like(rvec),
                tvec= np.zeros_like(palm3d[:3]),
                length=0.1,
                # origin=palm3d[:3],
                # offset2d = (int(palm[0]), int(palm[1])),
            )
            """

            hand_wrt_base = np.eye(4)
            # hand_wrt_base[:3, :3] = rot
            hand_wrt_base[:3, 3] = palm3d[:3]
            # hand_wrt_base = cam2base @ hand_wrt_base

            if len(base_hist) > 2:
                x, y, z = np.array(base_hist).T
                # pprint({ 'x': (x.min(), x.max()), 'y' :(y.min(), y.max()), 'z': (z.min(), z.max()), })
                norm = lambda a: (a - a.mean()) / a.std()
                # x,y,z = x[:-1]-x[1:], y[:-1]-y[1:], z[:-1]-z[1:]
                # x,y,z = norm(x), norm(y), norm(z)
                # smoothing
                x = np.convolve(x, np.ones(5) / 5, mode="same")
                y = np.convolve(y, np.ones(5) / 5, mode="same")
                z = np.convolve(z, np.ones(5) / 5, mode="same")
            else:
                x, y, z = [], [], []

            fig = plt.figure(figsize=(5, 4), dpi=100)
            # ax = fig.add_subplot(projection="3d")
            ax = fig.add_subplot()
            length = 2.5
            if False:
                # X axis (red)
                ax.quiver(
                    0,
                    0,
                    0,
                    length,
                    0,
                    0,
                    color="r",
                    arrow_length_ratio=0.1,
                    linewidth=1.5,
                )
                # Y axis (green)
                ax.quiver(
                    0,
                    0,
                    0,
                    0,
                    length,
                    0,
                    color="g",
                    arrow_length_ratio=0.1,
                    linewidth=1.5,
                )
                # Z axis (blue)
                ax.quiver(
                    0,
                    0,
                    0,
                    0,
                    0,
                    length,
                    color="b",
                    arrow_length_ratio=0.1,
                    linewidth=1.5,
                )
                ax.scatter(x, y, z, c="k", marker="o", s=10)
            else:
                ax.scatter(np.arange(len(x)), x, c="r", marker="o", s=10, label="x")
                ax.scatter(np.arange(len(y)), y, c="g", marker="o", s=10, label="y")
                ax.scatter(np.arange(len(z)), z, c="b", marker="o", s=10, label="z")
                ax.legend()

            fn = "temp_plot.png"
            plt.savefig(fn)  # writes out PNG

            # 3) Read it back with OpenCV
            img = cv2.imread(fn)  # BGR image
            cv2.imshow("plot via file", img)
            cv2.waitKey(1)

            plt.pause(0.01)
            plt.clf()

            # pprint({'palm3d': palm3d})

            # print("hand_wrt_base")
            bh = hand_wrt_base[:3, 3]
            base_hist.append(bh)
            # print(np.array(base_hist)[:-1] - np.array(base_hist)[1:])

            hist.append(palm)
            for point in points2d.reshape(-1, 2):
                # for point in out['keypoints_2d'][0]:
                # the original points are [-1->1] wrt the box
                # point = point * box["size"] + box["center"]
                # print(point)
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, black, -1)
                # cv2.circle(imwrist, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

            for a, b in zip(hist, list(hist)[1:]):
                cv2.line(
                    frame,
                    (int(a[0]), int(a[1])),
                    (int(b[0]), int(b[1])),
                    black,
                    2,
                )
                # cv2.line(imwrist, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 0, 0), 2)

        cv2.imshow("frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # cv2.imshow("wrist", cv2.cvtColor(imwrist, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


if __name__ == "__main__":
    main(tyro.cli(Config))
