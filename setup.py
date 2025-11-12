from setuptools import setup, find_packages
from glob import glob
import os

package_name = "xnodes"

def files(*patterns):
    out = []
    for pat in patterns:
        out.extend(glob(pat, recursive=True))
    return [p for p in out if os.path.isfile(p)]

launch_files = files(
    "launch/*.launch.py",
    "launch/**/*.launch.py",
    "launch/*.yaml", "launch/**/*.yaml",
    "launch/*.xml",  "launch/**/*.xml",
)

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    data_files=[
        (f"share/ament_index/resource_index/packages", ["resource/xnodes"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/launch", launch_files),
    ],
    entry_points={
        "console_scripts": [
            # add your nodes here
            # "apriltag_node = xnodes.apriltag:main",
        ],
    },
)

