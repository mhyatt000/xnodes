from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading
import time

import jax
import numpy as np
from openpi_client import websocket_client_policy as wcp
import rclpy
from rich.pretty import pprint
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
import tyro
from xarm_msgs.msg import RobotMsg

from .base import Base

np.set_printoptions(suppress=True)  # no scientific notation


class ActionRepresentation(Enum):
    ABS = "absolute"
    REL = "relative"


@dataclass
class ModelClientConfig:
    host: str
    port: str

    rep: ActionRepresentation
    task: str
    # to use weigted average across time
    ensemble: bool = False
    # server: Server = Server.XGYM  # model server architecture


NOMODEL = ModelClientConfig(
    host="none",
    port=8000,
    rep=ActionRepresentation.REL,
    task="none",
    ensemble=False,
)


class GaussianConv:
    def __init__(self, kernel_size=5, std=1.0):
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric smoothing"
        self.kernel_size = kernel_size
        self.std = std
        self.half_k = kernel_size // 2

        # Create symmetric Gaussian kernel
        x = np.arange(kernel_size) - self.half_k
        self.kernel = np.exp(-0.5 * (x / std) ** 2)
        self.kernel /= self.kernel.sum()

        # Rolling buffer with symmetric size
        self.buffer = deque(maxlen=kernel_size)

    def __call__(self, x):
        x = np.atleast_1d(x).astype(float)
        result = []

        if x.shape[0] == 1:
            # Streaming mode: single value
            self.buffer.append(x.item())
            buffer_np = np.array(self.buffer)
            i = len(buffer_np) - 1
            start = max(0, i - self.half_k)
            end = i + self.half_k + 1
            k_start = self.half_k - (i - start)
            k_end = k_start + (end - start)
            values = buffer_np[start:end]
            weights = self.kernel[k_start:k_end]
            return np.average(values, weights=weights)

        else:
            # Batch mode: fill buffer and process entire sequence
            self.buffer.clear()
            self.buffer.extend(x.tolist())
            buffer_np = np.array(self.buffer)
            N = len(x)

            for i in range(N):
                start = max(0, i - self.half_k)
                end = min(N, i + self.half_k + 1)
                k_start = self.half_k - (i - start)
                k_end = k_start + (end - start)
                values = x[start:end]
                weights = self.kernel[k_start:k_end]
                result.append(np.average(values, weights=weights))

            return np.array(result)


class MyClient(wcp.WebsocketClientPolicy):
    def reset(self):
        self.infer({"reset": True})


class Model(Base):
    """Recieves action from model server"""

    def __init__(self, cfg: ModelClientConfig):
        super().__init__("model")
        self.cfg = cfg

        self.joints = None
        self.pose = None
        self.gripper = None
        self.g = 1
        self.gconv = GaussianConv(kernel_size=5, std=1.0)

        self.req_hz = 20  # request from server frequency
        self.cmd_hz = 100  # command frequency
        self.resolution = 1  # every 2 predicted steps

        self.data = {}
        cams = [x for x in self.list_camera_topics() if any(k in x for k in ["rs", "worm"])]
        self.build_cam_subs()

        self.policy = MyClient(host=cfg.host, port=cfg.port)
        self.policy.reset()
        self._reset = True

        self.targets = None

        self.get_logger().info("Model Client Initialized.")

        self.moveit_sub = self.create_subscription(JointState, "/xarm/joint_states", self.set_joints, 10)
        self.moveit_pose_sub = self.create_subscription(RobotMsg, "/xarm/robot_states", self.set_pose, 10)
        self.gripper_sub = self.create_subscription(Float32MultiArray, "/xgym/gripper", self.set_gripper, 10)
        # self.timer = self.create_timer(1, self.step)
        self.publisher = self.create_publisher(Float32MultiArray, "/gello/state", 10)
        self.timer = self.create_timer(1 / self.cmd_hz, self.command)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._command_loop, daemon=True)
        self._thread.start()
        # self.stepper = self.create_timer(1 / self.req_hz, self.step)

        # self.ds = tfds.load("xgym_duck_single", split="train")
        # self.episode = self.ds.take(1)

    def set_active(self, msg):
        super().set_active(msg)
        time.sleep(0.25)
        self.reset()
        time.sleep(0.25)

        self.joints = None
        self.poleaderse = None
        self.gripper = None
        self.targets = None

    def set_gripper(self, msg: Float32MultiArray):
        self.gripper = np.array(msg.data).astype(np.float32)

    def set_pose(self, msg: RobotMsg):
        """
        RobotMsg message has:
            header: std_msgs/Header header
            pose: List[float64]
            ...others...
        """
        self.pose = np.array(msg.pose).astype(np.float32)

    def set_joints(self, msg: JointState):
        """
        JointState message has:
            header: std_msgs/Header header
            name: List[string]
            position: List[float64]
            velocity: List[float64]
            effort: List[float64]
        """

        if len(msg.position) == 6:
            return
        self.joints = np.array(msg.position)
        # self.joint_names = msg.name

    def _command_loop(self):
        rate = 1.0 / self.req_hz
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
        if any(x is None for x in [self.joints, self.pose, self.gripper]):
            return
        if self.active is False:
            if not self._reset:
                self.policy.reset()
                self._reset = True
            # conat joints gripper
            target = np.concatenate([self.joints, self.gripper], axis=-1)
            self.targets = [target]
            return

        # example batch
        # image_left_wrist: '(1, 1, 224, 224, 224) != (1, 1, 224, 224, 3)',
        # image_primary: '(1, 1, 224, 224, 224) != (1, 1, 224, 224, 3)',
        # proprio_single: '(1, 1, 1, 6) != (1, 1, 6)',

        pose = self.pose

        prefix = "/xgym/camera/"
        imgs = {k: self.data.get(prefix + k) for k in ["low", "side", "wrist"]}

        spec = lambda arr: jax.tree.map(lambda x: x.shape, arr)
        # pprint(spec(imgs))

        rad2deg = lambda x: x * 180 / np.pi
        deg2rad = lambda x: x * np.pi / 180

        if any(x is None for k, x in imgs.items()):
            self.get_logger().info(f"Missing images: {spec(imgs)}")
            return

        pose[:3] /= 1000  # mm to m

        try:
            gripper = self.gripper
            # gripper = np.array([0.99]) if gripper > 0.8 else gripper
            assert gripper <= 1.0, "Gripper out of bounds"
            state = np.concatenate([self.joints, gripper], axis=-1)
        except Exception as e:
            self.get_logger().info(f"Gripper error: {e}")
            return

        # pprint(state)
        obs = {
            "pixels": jax.tree.map(lambda x: x.astype(np.uint8), imgs),
            "agent_pos": state,
        }
        obs = jax.tree.map(lambda x: x.reshape(1, *x.shape), obs)

        # pprint(spec(obs))
        payload = {
            "observation": {
                # 'image_high'
                "image_left_wrist": obs["pixels"]["wrist"][None],
                "image_primary": obs["pixels"]["low"][None],
                "image_side": obs["pixels"]["side"][None],
                # 'proprio_single_arm' : pose.reshape(1,1,-1).astype(np.float32),
                "proprio_single_arm": np.concatenate(
                    [pose.reshape(1, 1, -1).astype(np.float32), state.reshape(1, 1, -1).astype(np.float32)], axis=-1
                ),
            },
            "ensemble": False,
        }

        # pprint(payload['observation']['proprio_single_arm'])
        # actions : dict= self.policy.infer(obs)
        actions: dict = self.policy.infer(payload)
        # pprint(spec(actions))

        act = actions["action"].copy()

        np.set_printoptions(suppress=True, precision=2)
        # pprint(np.minimum(1.0, act[:,-1]))
        # pprint( act[:,-1])

        # g = act[:, -1]
        # act[:,-1] = np.where(g < 0.5, 0.3, g)
        self.targets = act[:: self.resolution].copy()

        # print( t.round(4))
        # print(state.round(4))
        # print()

        """
        pprint(
                jax.tree.map(
                lambda x: x.round(4),
                    {
                        "actions": actions,
                        "joints": self.joints,
                        "gripper": self.gripper,
                        },
                    )
                )
        """
        return

        quit()

        # else: # no more uvicorn/fastapi
        # observation = asdict(observation)
        # actions = self.client(**observation)

        ntile = 1
        expanded = np.tile(actions, (ntile, 1))  # Repeat 4 times to get (200, 8)
        # expanded[:,-1] *= 850
        # expanded = actions.copy()

        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        # gripper = np.clip(expanded[:, -1:], 0, 1)
        gripper = np.clip(expanded[:, -1:], 0, 1)
        # gripper = np.where(gripper > 0.5, 1, 0)

        # g = gripper.mean()
        # gripper = np.zeros_like(gripper) + g
        # gripper = np.cumsum(gripper, axis=0) / np.arange(1, len(gripper) + 1)[:, None]

        # print(gripper.flatten().round(2))
        _grip = []
        # print(gripper.flatten().tolist())
        for g in gripper.flatten().tolist():
            rate = 0.9
            self.g = rate * self.g + (1 - rate) * g
            _grip.append(self.g)

        # gripper = self.gconv(np.array(_grip))[:, None]
        gripper = np.array(_grip)[:, None]

        # print(gripper.flatten().round(2))
        # print()

        # gripper @ 5hz
        # n = len(gripper) // 3
        # gripper = sum([[g] * n for g in gripper.tolist()[::n]], [])
        # gripper = np.array(gripper).round(1)
        # gripper = np.zeros_like(gripper) + gripper[n]

        # gripper = np.ones_like(gripper)

        # if self.cfg.rep == ActionRepresentation.REL:
        expanded[:, :-1] /= ntile  # gripper stays the same
        cumulants = np.cumsum(expanded[:, :-1], axis=0)
        targets = cumulants + np.tile(self.joints, (len(cumulants), 1))
        # if self.cfg.rep == ActionRepresentation.ABS:
        # print("ABS")
        # targets = expanded[:, :-1]

        targets = np.concatenate((targets, gripper), axis=1)

        # print(self.joints.round(4))
        # print(rad2deg(targets).round(4))

        # from matplotlib import pyplot as plt
        # fig,axs = plt.subplots(1,1)
        # for i in range(targets.shape[1]):
        # axs.plot(targets[:,i], label=f"joint {i}")

        # plt.legend()
        # plt.show()
        # quit()

        self.targets = targets

    def reset(self):
        self._reset = False  # send reset signal to thread

    def command(self):
        """Publishes model action"""

        if self.targets is None or len(self.targets) == 0:
            if self.joints is None or self.gripper is None:
                # print("No joints or gripper")
                return
            target = np.concatenate([self.joints, self.gripper], axis=-1)
            # self.step()
            # return
        else:
            target, self.targets = self.targets[0], self.targets[1:]
            # print(len(self.targets))

        msg = Float32MultiArray()
        msg.data = target.tolist()
        self.publisher.publish(msg)


def main(cfg: ModelClientConfig):
    pprint(cfg)

    args = None
    rclpy.init(args=args)

    node = Model(cfg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Controller Node shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(tyro.parse(ModelClientConfig))
