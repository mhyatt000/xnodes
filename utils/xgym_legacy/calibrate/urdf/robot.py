from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_kinematics as pk
import pytorch_kinematics.transforms as tf
from rich.pretty import pprint
import torch
from tqdm import tqdm
from xgym.names import BASE

# mpl webagg
# import matplotlib as mpl
# mpl.use("webagg")

# name of this file
FNAME = Path(__file__)
DNAME = Path(__file__).parent

urdf = DNAME / "xarm7_standalone.urdf"


class RobotTree:
    def __init__(self, path: Path | str = urdf):
        self.path = path

        with open(path, "rb") as f:
            data = f.read()
        self.chain = pk.build_chain_from_urdf(data)

        try:
            # self.chain.print_tree()
            print(self.chain)
        except Exception as e:
            print(f"Error printing tree: {e}")

        # extract a specific serial chain such as for inverse kinematics
        # link_eef xarm_gripper_base_link link_tcp
        self.serial = pk.SerialChain(self.chain, "xarm_gripper_base_link", "link_base")
        # self.serial.print_tree()

        self.names = self.chain.get_joint_parameter_names()
        pprint(self.names)

        self.pose = np.zeros((7,))
        self.kin = self.set_pose(self.pose)

    def set_pose(self, pose: np.ndarray, end_only=False) -> np.ndarray:
        """Set the pose of the robot"""
        assert pose.shape[-1] == 7
        self.pose = torch.Tensor(pose)
        self.kin = self.serial.forward_kinematics(self.pose, end_only=end_only)
        return self.kin

    def j(self, q: torch.Tensor):
        """Convert joint angles to tensor"""
        if not torch.is_tensor(q):
            q = torch.Tensor(q)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        J = self.serial.jacobian(q)
        return J


@dataclass
class RobotPose:
    pass


def plot_frame(mat, ax, length=0.025):
    """Plot a frame in 3D space in the orientation of the 4x4 matrix"""
    # convert to x,y,z
    x = mat[0, 3]
    y = mat[1, 3]
    z = mat[2, 3]

    # get rotation matrix
    r = mat[:3, :3]

    # plot the axes
    ax.quiver(x, y, z, r[0, 0], r[1, 0], r[2, 0], color="r", length=length)
    ax.quiver(x, y, z, r[0, 1], r[1, 1], r[2, 1], color="g", length=length)
    ax.quiver(x, y, z, r[0, 2], r[1, 2], r[2, 2], color="b", length=length)


def main():
    robot = RobotTree(str(urdf))

    viz = False
    if viz:
        _fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": "3d"})

    # plot x,y,z axes at origin
    # plot_frame(np.eye(4), ax, length=0.1)

    ways = np.load(BASE / "ways.npz")
    ways = {k: ways[k] for k in ways.files}

    centers = []
    frames = []
    X, Y, Z = [], [], []
    kinvs = []
    for i in tqdm(range(len(ways["joints"]))):
        d = {k: v[i] for k, v in ways.items()}

        joints = d["joints"]
        _pose = d["poses"]
        frame = d["frames"]

        kin: tf.Transform3d = robot.set_pose(joints, end_only=True).get_matrix()
        KIN: dict[tf.Transform3d] = robot.set_pose(joints, end_only=False)
        KIN = {k: v.get_matrix() for k, v in KIN.items() if k.startswith("link")}
        kinvs.append(kinv := np.linalg.inv(kin))

        if not viz:
            continue

        # kin = kinvs[-1]
        # x, y, z = kin[:, :3, 3].flatten()

        # plot_frame(kin[0].numpy(), ax)

        x, y, z = kin[:, :3, 3].flatten().numpy()
        X.append(x)
        Y.append(y)
        Z.append(z)

        print(KIN.keys())
        _X, _Y, _Z = [], [], []
        for k, mat in KIN.items():
            if viz and k == "link_base":
                # plot_frame(mat[0].numpy(), ax, length=0.1)
                plot_frame(kinv[0], ax, length=0.1)

            _x, _y, _z = mat[0][:3, 3].flatten().numpy()
            _X.append(_x)
            _Y.append(_y)
            _Z.append(_z)
        if viz:
            ax.plot(_X, _Y, _Z, alpha=0.8, color="gray")

    if viz:
        ax.scatter(X, Y, Z, c="r", marker="o", label="End Effector")
        plt.show()

    np.savez(BASE / "kinvs.npz", kinvs=kinvs)


if __name__ == "__main__":
    main()


def sandbox():
    # prints out list of joint names

    th = np.array([0.0, -np.pi / 4.0, 0.0, np.pi / 2.0, 0.0, np.pi / 4.0, 0.0])

    _fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": "3d"})

    allpositions = {}
    positions = []
    scale = 0.5
    scale = 2
    for j in tqdm(joints := range(7)):
        for i in np.linspace(-np.pi / scale, np.pi / scale, 50):
            t = th.copy()
            t[j] += i

            ret = serial_chain.forward_kinematics(t, end_only=False)
            tg = ret["link_tcp"]
            m = tg.get_matrix()
            pos = m[:, :3, 3]
            rot = pk.matrix_to_quaternion(m[:, :3, :3])
            # print(f"joint {j} angle {i:.2f} pos {pos} rot {rot}")

            ap = {k: ret[k].get_matrix()[:, :3, 3].flatten().numpy() for k in ret if k.startswith("link")}
            for k, v in ap.items():
                if k not in allpositions:
                    allpositions[k] = []
                allpositions[k].append(v)

            # convert to x,y,z
            positions.append(pos.flatten().numpy())

    allpositions = {k: np.array(v) for k, v in allpositions.items()}
    for k, v in allpositions.items():
        x, y, z = v[:, 0], v[:, 1], v[:, 2]
        ax.scatter(x, y, z, marker="o", label=k)

    # scatter link_base with large circle
    base = allpositions["link_base"]
    x, y, z = base[:, 0], base[:, 1], base[:, 2]
    ax.scatter(x, y, z, c="b", marker="x", label="Base")

    # plot all joints together
    keys = [
        "link_base",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "link_eef",
        "link_tcp",
    ]
    for i in range(0, len(base), 10):
        x = [allpositions[k][i, 0] for k in keys]
        y = [allpositions[k][i, 1] for k in keys]
        z = [allpositions[k][i, 2] for k in keys]
        ax.plot(x, y, z, alpha=0.8, color="gray")

    positions = np.array(positions)
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    # ax.scatter(x, y, z, c="r", marker="o")

    ax.legend()
    plt.show()
    quit()

    # specify joint values (can do so in many forms)
    # do forward kinematics and get transform objects; end_only=False gives a dictionary of transforms for all links
    ret = serial_chain.forward_kinematics(th, end_only=False)
    # look up the transform for a specific link

    # numpy no scientific notation
    np.set_printoptions(suppress=True)

    pprint(ret)
    tg = ret["link_tcp"]
    # get transform matrix (1,4,4), then convert to separate position and unit quaternion
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])

    pprint(m.numpy())
