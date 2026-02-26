from __future__ import annotations

from pathlib import Path
import time

from rich.pretty import pprint
from std_msgs.msg import Bool, Int32MultiArray
from xgym.nodes.base import Base


def delete_latest_files(path: Path):
    dat = sorted(path.glob("*.dat"), key=lambda p: p.stat().st_ctime)
    j = sorted(path.glob("*.json"), key=lambda p: p.stat().st_ctime)

    # dat_files = sorted(glob.glob(os.path.join(directory, "*.dat")), key=os.path.getctime)
    # json_files = sorted(glob.glob(os.path.join(directory, "*.json")), key=os.path.getctime)

    if dat and j:
        dat, j = dat[-1], j[-1]
        pprint({"deleting": (dat, j)})
        dat.unlink()
        j.unlink()
    else:
        pprint((False, "No .dat or .json files found."))
    time.sleep(0.5)  # ensure no double tap


class Governor(Base):
    def __init__(self, cfg):
        super().__init__("governor")
        self.cfg = cfg

        self.pedal_sub = self.create_subscription(Int32MultiArray, "/xgym/pedal", self.on_pedal, 10)

        self.pubs = {
            "replay": self.create_publisher(Bool, "/xgym/gov/writer/replay", 10),
            "del": self.create_publisher(Bool, "/xgym/gov/writer/del", 10),
        }

    def on_pedal(self, msg):
        data = {
            "ss": msg.data[0],  # start stop
            "fn1": msg.data[1],
            "fn2": msg.data[2],
        }

        if data["ss"] == 1:
            self.active_pub.publish(Bool(data=not self.active))
            self.get_logger().info("XGym Reactivated")

        # if data["fn1"] == 1:
        # if self.active is False:  # TODO add toggle replay
        # self.get_logger().info("Replay")
        # self.pubs["replay"].publish(Bool(data=True))
        if data["fn1"] == 2 and self.active is False:
            self.pubs["del"].publish(Bool(data=True))
        if data["fn2"] == 1:
            self.get_logger().info("Pedal 2 pressed deleting latest data files...")
            delete_latest_files(self.cfg.dir)


class ModelGovernor(Governor):
    def __init__(self):
        super().__init__("model_governor")


class ReplayGovernor(Governor):
    def __init__(self):
        super().__init__("model_governor")
        # TODO use space mouse to control video scrubbing ?
