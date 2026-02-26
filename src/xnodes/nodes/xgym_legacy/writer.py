from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray
from xarm_msgs.msg import RobotMsg
import xgym

from .base import Base


class Writer(Base):
    def __init__(self, seconds=15, dir="."):
        super().__init__("writer")

        self.t0 = time.time()
        self.recording = True  #  Flag to control recording
        self.dir = dir

        self.schema = {
            "xarm_joints": 7,
            "xarm_pose": 6,
            "xarm_gripper": 1,
            "gello_joints": 8,  # pad spacemouse 7->8 if used
        }
        self.data = dict.fromkeys(self.schema.keys())

        self.cwd = os.getcwd()
        self.hz = 50

        # QoS settings (ensures reliability)
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Matches publisher QoS
            durability=QoSDurabilityPolicy.VOLATILE,  # Allows receiving live messages
        )

        self.set_period()
        self.build_cam_subs()

        camspec = {k: self.data[k].shape for k in self.cams}
        self.schema = self.schema | camspec

        # old subscriber overwrote the moveit topic
        # self.create_subscription(
        # Float32MultiArray, "/robot_state", self.set_state, qos_profile
        # )

        # self.space_sub = self.create_subscription(
        # Float32MultiArray, "/robot_commands", self.set_cmd, qos_profile
        # )

        self.gello_sub = self.create_subscription(Float32MultiArray, "/gello/state", self.set_gello, 10)

        self.moveit_sub = self.create_subscription(JointState, "/xarm/joint_states", self.set_joints, 10)
        self.moveit_pose_sub = self.create_subscription(RobotMsg, "/xarm/robot_states", self.set_pose, 10)
        self.gripper_sub = self.create_subscription(Float32MultiArray, "/xgym/gripper", self.set_gripper, 10)

        self.subs = {
            "replay": self.create_subscription(Bool, "/xgym/gov/writer/replay", self.on_replay, 10),
        }

        self.timer = self.create_timer(1 / self.hz, self.write)

        self.get_logger().info("initialized")

        self.nrecord = self.hz * seconds  # 50 Hz for 5 seconds
        self.datai = 0
        self.build_memap()

        # # Start background thread to save data every 2 seconds
        # self.flush_thread = threading.Thread(target=self.periodic_save, daemon=True)
        # self.flush_thread.start()

    def set_active(self, msg: Bool):
        self.memap.flush()  # in case early stopping
        with open(str(self.path).replace(".dat", ".json"), "w") as f:
            data = {
                "schema": self.schema,
                "info": {
                    "hz": self.hz,
                    "len": self.datai,
                    "maxlen": self.nrecord,
                    "path": str(self.path),
                },
            }
            json.dump(data, f)

        active = msg.data
        if active:
            self.build_memap()
        super().set_active(msg)

    def build_memap(self):
        # images are uint8
        dtype = np.dtype(
            [("time", np.float32)]
            + [(k, np.float32 if isinstance(v, int) else np.uint8, v) for k, v in self.schema.items()]
        )

        dt = time.strftime("%Y%m%d-%H%M%S")
        self.path = Path(self.dir) / f"xgym_data_{dt}.dat"

        # dump schema and info to json
        with open(str(self.path).replace(".dat", ".json"), "w") as f:
            data = {
                "schema": self.schema,
                "info": {
                    "hz": self.hz,
                    "len": self.datai,
                    "maxlen": self.nrecord,
                    "path": str(self.path),
                },
            }
            json.dump(data, f)

        self.memap = np.memmap(self.path, dtype=dtype, mode="w+", shape=self.nrecord)
        self.datai = 0
        self.get_logger().info(f"Created memap at {self.path}")

    def write(self):
        logger = self.get_logger()
        if not self.active:
            # logger.info("Inactive")
            return
        if not self.recording:
            return

        self.logp(f"Writing data {self.datai}/{self.nrecord}")
        if (self.datai >= self.nrecord - 1) or (self.datai % self.hz == 0):
            self.memap.flush()
        spec = {k: v.shape if v is not None else None for k, v in self.data.items()}
        self.logp(f"Spec: {spec}")

        if self.datai >= self.nrecord - 1:
            self.active_pub.publish(Bool(data=False))
            return

        # self.memap[int(self.datai)] = [self.timestamp()]+[self.data[k] for k in sorted(self.schema.keys())]
        self.memap[int(self.datai)] = (self.timestamp(), *self.data.values())
        self.datai += 1

    def set_cmd(self, msg: Float32MultiArray):
        """Callback function for /robot_commands topic"""
        self.data["gello_joints"] = np.array(msg.data)

    def set_gello(self, msg: Float32MultiArray):
        raw = np.array(msg.data)
        self.data["gello_joints"] = raw  # TODO add more measurements later

    # old joint state subscriber
    # def set_state(self, msg: Float32MultiArray):
    # """Callback function for /robot_state topic"""
    # self.data["xarm_joints"] = np.array(msg.data)

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
        self.data["xarm_joints"] = self.joints

    def set_pose(self, msg: RobotMsg):
        """
        RobotMsg message has:
            header: std_msgs/Header header
            pose: List[float64]
            ...others...
        """
        self.pose = np.array(msg.pose).astype(np.float32)
        # self.pose[:3] /= 1000 # mm to m
        self.data["xarm_pose"] = self.pose

    def set_gripper(self, msg: Float32MultiArray):
        self.data["xarm_gripper"] = np.array(msg.data).astype(np.float32)

    def on_replay(self, msg: Bool):
        """Reads the memap file."""
        self.get_logger().info(f"Reading from {self.path}")
        if not msg.data:
            return

        _info, data = xgym.viz.memmap.read(self.path)
        xgym.viz.memmap.view(data)


def main(args=None):
    rclpy.init(args=args)
    node = Writer()

    try:
        rclpy.spin(node)  # Keeps the node running
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down XArm Listener Node")
    finally:
        node.save_data()  # Final save before exit
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
