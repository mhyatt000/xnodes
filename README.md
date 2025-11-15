# xnodes

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Runtime Targets](#runtime-targets)
- [Data](#data)
- [Contributing](#contributing)

## Overview
xnodes is a collection of ROS 2 (Python) nodes focused on spatial perception and
calibration workflows. It provides ready-to-run executables for tasks such as
AprilTag detection, stereo camera anchoring, fiducial visualization, and chess
pattern helpers. The package is designed to be dropped into an existing ROS 2
workspace so robotics teams can bootstrap localization pipelines or integrate
calibration utilities without rebuilding core infrastructure.

Highlights:
- **AprilTag detector** (`xnodes.apriltag.AprilTagNode`) publishes tag poses,
  marker arrays, and transforms from monocular camera streams.
- **Stereo anchor node** (`xnodes.nodes.stereo_anchor_node.StereoAnchorNode`)
  converts live pose estimates into globally referenced TF frames for multi-
  camera rigs.
- **Utility nodes** in `src/xnodes/nodes/` (e.g. `marker.py`, `chess.py`) offer
  building blocks for visualization and calibration tasks.
- Launch files in `launch/` wire common camera and visualization setups
  together, while YAML configs under `src/xnodes/config/` document expected
  parameters.

## Installation
The package targets ROS 2 Humble (or newer) on Python 3.11. You will need a ROS
2 environment sourced prior to running the nodes so TF, message definitions, and
`rclpy` are available.

### Quick start inside an existing ROS workspace
1. Clone the repository into your ROS workspace `src` directory:
   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/your-org/xnodes.git
   ```
2. Install Python dependencies into your environment (virtualenv or system)
   using `pip`:
   ```bash
   cd ~/ros2_ws/src/xnodes
   pip install -U pip
   pip install -e .
   ```
3. Build the workspace and source the overlay:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select xnodes
   source install/setup.bash
   ```
4. Launch any of the included launch files, e.g. the multicam setup:
   ```bash
   ros2 launch xnodes multicam.launch.py
   ```

### Standalone development (without colcon)
If you only need the Python modules (e.g. for unit testing), create a virtual
environment and install locally:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```
You can then run nodes directly with `ros2 run xnodes <entrypoint>` as long as
your ROS 2 environment is sourced.

## Runtime Targets
Use the following commands to invoke the nodes and launch files that ship with
the package:

### `ros2 run`
- `ros2 run xnodes stereo_anchor_node` &mdash; publishes TF frames for stereo
  camera rigs using live pose estimates.

### `ros2 launch`
- `ros2 launch xnodes multicam.launch.py` &mdash; brings up the stereo anchor node
  alongside RViz helpers for multi-camera calibration workflows.
- `ros2 launch xnodes rqt.launch.py` &mdash; starts the RQt visualization bundle
  preconfigured for xnodes debugging.

## Data
The nodes operate on live ROS topics rather than static datasets. Ensure the
following streams are available when launching the package:

- **Camera topics**: Image (`sensor_msgs/msg/Image`) and camera info
  (`sensor_msgs/msg/CameraInfo`) topics for each camera feeding the AprilTag
  detector. The default expectation is `<topic>/image_raw` and
  `<topic>/camera_info`, configurable through Tyro CLI arguments.
- **Pose topics**: `geometry_msgs/msg/PoseStamped` streams that describe the
  transforms between anchor/trackers and floaters for the stereo anchor node.
  Configure topic names and frame IDs via parameters or the
  `src/xnodes/config/stereo_anchor.yaml` example file.
- **Frame tree**: A TF tree providing the world/anchor relationship so derived
  transforms can be broadcast.

If you rely on recorded data, play back a ROS 2 bag that exposes the same topic
layout prior to launching xnodes.

## Contributing
We welcome contributions that expand node coverage or improve usability.

- **Branching**: Develop on feature branches (`feature/<short-description>`) and
  open pull requests against `main`.
- **Coding style**: Follow the concise docstring style in `AGENTS.md`, favouring
  dataclasses for configuration and keeping functions short (cyclomatic
  complexity ≤ 8). Prefer feature-oriented module organization under
  `src/xnodes/`.
- **Testing**: Run `colcon test --packages-select xnodes` (or relevant unit
  tests) before submitting a PR. Provide launch-level smoke tests where
  applicable.
- **Pull requests**: Keep diffs focused, include context on the intended robot
  setup, and update documentation/launch files when adding new topics or
  parameters.

For onboarding questions, file an issue or reach out to the maintainers listed
in `package.xml`.
