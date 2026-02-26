"""reads memmaps of a schema"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

# import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def fold_angle(theta: np.ndarray) -> np.ndarray:
    return np.where(
        theta > np.pi / 2,
        theta - np.pi,
        np.where(theta < -np.pi / 2, theta + np.pi, theta),
    )


def read_all(root=Path().cwd(), show=False):
    paths = list(root.rglob("*.dat"))
    print(paths)
    for p in paths:
        _info, data = read(p)
        if show:
            view(data)


def read_filtered(path, dtype, n):
    # doesnt need specified length, just dtype
    data = np.memmap(path, dtype=dtype, mode="r", shape=n)

    data = list(data)
    # filter rows where any nan element
    # data = [d for d in data if not any([np.any(np.array(x).isnan()) for x in d])]
    # filter rows where all zeros
    data = [d for d in data if not np.all([np.all(np.array(x) == 0) for x in d])]
    return data


def read(path):
    ###
    ### read info / schema
    ###
    spath = str(path).replace(".dat", ".json")
    with open(spath, "r") as f:
        info = json.load(f)
    schema = info["schema"]

    schema = {"time": 1} | schema
    dtype = np.dtype(
        # [("time", np.float32)] +  # time is in schema now
        [(k, np.float32 if isinstance(v, int) else np.uint8, v) for k, v in schema.items()]
    )

    # TODO add time to the writer schema
    # print(schema)
    # print(dtype)

    ###
    ### read data
    ###
    n = info["info"]["len"]
    data = read_filtered(path, dtype, n)

    n = len(data)
    info["nrecords"] = n

    ###
    ### preprocess
    ###
    data = {k: np.array([d[i] for d in data]) for i, k in enumerate(schema.keys())}
    data = {k: np.stack(d).reshape(n, -1) for k, d in data.items()}

    # print({k: v.shape for k, v in data.items()})

    data["time"] = data["time"] - data["time"][0]
    for k, values in data.items():
        if "cam" in k:
            data[k] = values.reshape((-1, *schema[k]))

    return info, data


def view(data):
    print("viewing")
    n = len(data["time"])
    for i in tqdm(range(n)):
        keys = [k for k in data if "cam" in k]
        frames = [data[k][i] for k in keys]
        frame = np.concatenate([cv2.resize(f, (480, 480)) for f in frames], axis=1)

        # num frames with all zeros
        nzero = np.sum([frame.sum() == 0 for frame in frames])
        print(nzero)
        if frame.sum() == 0:
            continue
        # cv2.imshow('cam', cv2.resize(frame, (640, 4*640)))
        print(i)
        cv2.imshow("cam", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.waitKey(10)  # 50hz is 20ms wait
        # cv2.waitKey(300)
        # cv2.waitKey(0)

    """
    # import matplotlib
    # matplotlib.use('Agg')
    # from matplotlib import plt
    fig, axs = plt.subplots(3, 1)
    for k, values in data.items():
        if k == "time" or "cam" in k:
            continue

        for i in range(values.shape[1]):
            v = values[:-1, i]

            label = k

            if k == "xarm_gripper":
                v = v / 850

            color = "blue" if "xarm" in k else "red"
            linestyle = "dashed" if "grip" in k else "solid"
            if "gello" in k and i == 7:
                linestyle = "dashed"
                label = "gripper"

            if k == "xarm_pose":
                if i > 2:
                    v = fold_angle(v)
                    v = v - v[0]
                    label = "xarm_orient"
                    linestyle = "dashed"
                else:
                    v = v - v[0]
                    label = ["x", "y", "z"][i]
                    color = ["red", "green", "blue"][i]

            ax = axs[0] if "xarm_pose" not in k else axs[1] if i < 3 else axs[2]

            ax.plot(data["time"][:-1], v, color=color, linestyle=linestyle, label=label)

    for ax in axs:
        ax.grid()
        ax.legend()
    plt.show(block=False)  # non-blocking code
    """

    # iif cv2.waitKey(0) & 0xFF == ord("q"):
    # plt.clf()
    # plt.close("all")
    # icv2.destroyWindow()
