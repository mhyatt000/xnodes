from __future__ import annotations

from pydantic import BaseModel, Field
from sensor_msgs.msg import CameraInfo, RegionOfInterest
from std_msgs.msg import Header
import yaml

# ---------------------------
# ROI + Header helpers
# ---------------------------


class ROIModel(BaseModel):
    x_offset: int = 0
    y_offset: int = 0
    height: int = 0
    width: int = 0
    do_rectify: bool = False

    def to_msg(self) -> RegionOfInterest:
        return RegionOfInterest(
            x_offset=self.x_offset,
            y_offset=self.y_offset,
            height=self.height,
            width=self.width,
            do_rectify=self.do_rectify,
        )


class HeaderModel(BaseModel):
    stamp: float = 0.0
    frame_id: str = ""

    def to_msg(self) -> Header:
        h = Header()
        h.stamp = h.stamp.from_sec(self.stamp)
        h.frame_id = self.frame_id
        return h


# ---------------------------
# CameraInfoModel
# ---------------------------


class CameraInfoModel(BaseModel):
    height: int
    width: int
    distortion_model: str = "plumb_bob"

    D: list[float] = Field(default_factory=list)
    K: list[float] = Field(default_factory=lambda: [0.0] * 9)
    R: list[float] = Field(default_factory=lambda: [0.0] * 9)
    P: list[float] = Field(default_factory=lambda: [0.0] * 12)

    binning_x: int = 0
    binning_y: int = 0

    roi: ROIModel = Field(default_factory=ROIModel)
    header: HeaderModel | None = None

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
        validate_assignment = True

    # convert back to ROS msg
    def info(self) -> CameraInfo:
        msg = CameraInfo()
        if self.header is not None:
            msg.header = self.header.to_msg()

        msg.height = self.height
        msg.width = self.width
        msg.distortion_model = self.distortion_model
        msg.D = list(self.D)
        msg.K = list(self.K)
        msg.R = list(self.R)
        msg.P = list(self.P)
        msg.binning_x = self.binning_x
        msg.binning_y = self.binning_y
        msg.roi = self.roi.to_msg()
        return msg

    # ---------------------------
    # YAML loader
    # ---------------------------
    @classmethod
    def from_yaml(cls, path: str) -> CameraInfoModel:
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # OpenCV YAML → CameraInfoModel normalization
        def _extract(mat):
            if isinstance(mat, dict) and "data" in mat:
                return mat["data"]
            return mat

        return cls(
            height=data.get("image_height") or data.get("height"),
            width=data.get("image_width") or data.get("width"),
            distortion_model=data.get("distortion_model", "plumb_bob"),
            D=_extract(data.get("distortion_coefficients", {})),
            K=_extract(data.get("camera_matrix", {})),
            R=_extract(data.get("rectification_matrix", {})),
            P=_extract(data.get("projection_matrix", {})),
            binning_x=data.get("binning_x", 0),
            binning_y=data.get("binning_y", 0),
        )
