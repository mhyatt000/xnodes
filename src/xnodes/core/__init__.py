"""Core utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "NOMODEL",
    "ActionRepresentation",
    "ModelClientConfig",
    "build_idle_target",
    "build_policy_payload",
    "corner_grid_rms",
    "extract_action_targets",
    "extract_camera_images",
    "missing_camera_images",
]


def __getattr__(name: str) -> Any:
    if name == "corner_grid_rms":
        module = import_module(".metrics", __name__)
        return module.corner_grid_rms

    model_exports = {
        "ActionRepresentation",
        "ModelClientConfig",
        "NOMODEL",
        "build_idle_target",
        "build_policy_payload",
        "extract_action_targets",
        "extract_camera_images",
        "missing_camera_images",
    }
    if name in model_exports:
        module = import_module(".model_client", __name__)
        return getattr(module, name)

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
