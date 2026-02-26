from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml


class CalibrationTarget(BaseModel):
    # tell pydantic this is a tagged union base
    model_config = ConfigDict(discriminator="target_type")

    target_type: str  # overridden with Literals below

    @classmethod
    def load(cls, path: Path | str) -> CalibrationTarget:
        with open(str(path), "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)  # returns a subclass instance


class CheckerboardConfig(CalibrationTarget):
    target_type: Literal["checkerboard"]
    targetRows: int = Field(ge=3)
    targetCols: int = Field(ge=3)
    rowSpacingMeters: float = Field(gt=0)
    colSpacingMeters: float = Field(gt=0)


class CircleGridConfig(CalibrationTarget):
    target_type: Literal["circlegrid"]
    targetRows: int = Field(ge=3)
    targetCols: int = Field(ge=3)
    spacingMeters: float = Field(gt=0)
    asymmetricGrid: bool


class AprilGridConfig(CalibrationTarget):
    target_type: Literal["aprilgrid"]
    tagRows: int = Field(ge=3)
    tagCols: int = Field(ge=3)
    tagSize: float = Field(gt=0)
    tagSpacing: float = Field(ge=0)
    codeOffset: int = Field(ge=0)
    families: str = "tag36h11"

    @property
    def num_tags(self) -> int:
        return self.tagCols * self.tagRows

    @property
    def center_spacing(self) -> float:
        """Distance between neighboring tag centers [m]."""
        return self.tagSize * (1.0 + self.tagSpacing)

    @classmethod
    def create(cls):
        data = {
            "target_type": "aprilgrid",  # gridtype
            "tagCols": 6,  # number of apriltags
            "tagRows": 4,  # number of apriltags
            "tagSize": 0.03,  # size of apriltag, edge to edge [m]
            "tagSpacing": 0.0,  # ratio of space between tags to tagSize
            "codeOffset": 0,  # code offset for the first tag in the aprilboard
        }
        return cls.model_validate(data)
