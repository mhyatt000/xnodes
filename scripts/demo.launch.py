from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rich.pretty import pprint
import tyro
from xgym.nodes import Camera, FootPedal, Gello, Governor, Heleo, Model, SpaceMouse, Writer, Xarm
from xgym.nodes.model import ModelClientConfig, NOMODEL
from xgym.nodes.robot import ControlMode, InputMode, RobotConfig


def default(x):
    return field(default_factory=lambda: x)


@dataclass
class CameraConfig:
    side: int | None
    low: int | None
    wrist: int | None
    over: int | None = None  # optional

    def validate(self):
        assert all(x is not None for x in asdict(self).values()), "Camera indices must be set"

    def create(self):
        # self.validate()
        return {k: Camera(name=k, idx=v) for k, v in asdict(self).items() if v}

        # cameras = [
        # Camera(idx=0, name="side"),
        # Camera(idx=3, name="low"),
        # Camera(idx=8, name="wrist"),
        # Camera(idx=3, name="high"),
        # ]
        # cameras = {x.name: x for x in cameras}


@dataclass
class RunCFG:
    task: str = "demo"
    dir: Path = "."  # data directory

    seconds: int = 15
    nepisodes: int = 100
    use_gripper: bool = True

    cam: CameraConfig = tyro.MISSING  # default(CameraConfig())
    input: InputMode = InputMode.GELLO
    ctrl: ControlMode = ControlMode.JOINT

    # TODO make a factory bc task
    model: ModelClientConfig = default(NOMODEL)

    @property
    def robot(self):
        return RobotConfig(input=self.input, ctrl=self.ctrl)


"""
base_dir: str = osp.expanduser("~/data")
time: str = time.strftime("%Y%m%d-%H%M%S")
env_name: str = f"xgym-sandbox-{task}-v0-{time}"
data_dir: str = osp.join(base_dir, env_name)
"""


def main(cfg: RunCFG):
    """Main training loop with environment interaction."""

    import box

    box.build()

    pprint(cfg)
    print("[DEBUG] Task name:", cfg.task)
    print("[DEBUG] Model task:", cfg.model.task)
    print("[DEBUG] Model host:", cfg.model.host)
    print("[DEBUG] Input mode:", cfg.input)
    print("[DEBUG] Control mode:", cfg.ctrl)

    # Start environment-related scripts
    rclpy.init()
    cameras = cfg.cam.create()

    match cfg.input:
        case InputMode.GELLO:
            ctrl = Gello()
        case InputMode.SPACEMOUSE:
            ctrl = SpaceMouse()
        case InputMode.MODEL:
            ctrl = Model(cfg.model)
        case InputMode.HELEO:
            ctrl = Heleo(cfg.model)

    nodes = {
        "robot": Xarm(cfg.robot),
        # 'viewer': FastImageViewer(active=False),
        "ctrl": ctrl,
        "writer": Writer(seconds=cfg.seconds, dir=cfg.dir),  # writer spins after cameras init
        "pedal": FootPedal(),
        "gov": Governor(cfg),
    }
    nodes = cameras | nodes  # cameras spin before writer
    nodes = list(nodes.values())

    # import launch

    # bag = launch.actions.Node(
    # package="rosbag2_transport",
    # executable="record",
    # name="record",
    # )

    ex = MultiThreadedExecutor()
    for node in nodes:
        ex.add_node(node)
    ex.spin()

    # finally:
    # _ = [rclpy.spin(node) for node in nodes]
    # _ = [node.destroy_node() for node in nodes]
    # rclpy.shutdown()

    # for node in nodes:
    # node.destroy_node()

    rclpy.shutdown()
    quit()


if __name__ == "__main__":
    main(tyro.cli(RunCFG))
