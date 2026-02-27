from __future__ import annotations

from glob import glob
import os
from pathlib import Path

from setuptools import find_packages, setup

package_name = "xnodes"


def files(*patterns):
    out = []
    for pat in patterns:
        out.extend(glob(pat, recursive=True))
    return [p for p in out if os.path.isfile(p)]


launch_files = files(
    "launch/*",  # catchall
    "launch/*.launch.py",
    "launch/**/*.launch.py",
    "launch/*.yaml",
    "launch/**/*.yaml",
    "launch/*.xml",
    "launch/**/*.xml",
)

# rglob all config files
configs = Path("config").rglob("*.yaml")
config_files = [str(p) for p in configs if p.is_file()]

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "transforms3d",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/xnodes"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/launch", launch_files),
        (f"share/{package_name}/config", config_files),
    ],
    entry_points={
        "console_scripts": [
            # add your nodes here
            # "apriltag_node = xnodes.apriltag:main",
            "stereo = xnodes.nodes.stereo_anchor_node:run",
            "cam = xnodes.nodes.opencv_camera:run",
            "intr = xnodes.nodes.chess:run",
            "april = xnodes.nodes.april_grid:run",
            "keyboard = xnodes.nodes.ctrl.keyboard:run",
            "model = xnodes.nodes.legacy.model:run",
        ],
    },
)
