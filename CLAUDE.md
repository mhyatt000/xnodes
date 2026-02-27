# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## docs

Use `tree -a docs` for the local docs map:

```text
docs
├── DESIGN.md
├── PIXI.md
├── PIXI_GH.md
├── TYRO.md
├── ZEN.md
├── antipattern
│   ├── broad_except.md
│   ├── import.md
│   └── node_business_logic_coupling.md
├── api
├── batch.md
├── install.md
└── test
    ├── ml
    │   ├── integration.md
    │   └── unit.md
    └── ros
        ├── integration.md
        └── unit.md
```

One-liners for each docs file:
- `docs/DESIGN.md` - Entry point for design decisions; links to antipattern guidance.
- `docs/PIXI.md` - How to define and run Pixi tasks, with syntax, examples, and best practices.
- `docs/PIXI_GH.md` - How to add GitHub-hosted Python dependencies to Pixi using PEP 508 refs.
- `docs/TYRO.md` - Dataclass + Tyro CLI style guide (field comments inline, `main(tyro.cli(Config))` pattern).
- `docs/ZEN.md` - The Zen of Python principles used as style and design heuristics.
- `docs/batch.md` - Canonical nested batch spec example (action/observation/task shapes and pad masks).
- `docs/install.md` - Installation policy: build via `colcon` in `~/ws`; never use pip/uv/conda.
- `docs/antipattern/broad_except.md` - Why broad `except` is harmful and how to catch specific exceptions.
- `docs/antipattern/import.md` - Avoid hiding import errors with try/except fallback stubs; fail explicitly on missing deps.
- `docs/antipattern/node_business_logic_coupling.md` - Keep ROS nodes thin; move decision logic into testable services/policies.
- `docs/test/ros/unit.md` - ROS unit test spec: pure-Python helpers only, no rclpy, run with pytest.
- `docs/test/ros/integration.md` - ROS integration test spec: launch_testing, real nodes, topic/TF assertions, colcon test.

## What is xnodes

A ROS 2 Python package for spatial perception and camera calibration. Provides nodes for AprilTag detection, stereo camera anchoring, checkerboard calibration, and fiducial visualization. Targets ROS 2 Humble+ on Python 3.11.

This package lives at `~/ws/src/xnodes` and is built inside the ROS 2 workspace at `~/ws`. Dependencies are declared in `package.xml` and resolved by the workspace — do not use `pip`, `uv`, or `conda` to manage dependencies.

## Build & Run

```bash
# ROS workspace build (from ~/ws)
colcon build --packages-select xnodes
source install/setup.bash

# Launch
ros2 launch xnodes multicam.launch.py
ros2 launch xnodes rqt.launch.py

# Run individual nodes
ros2 run xnodes stereo    # stereo anchor TF publisher
ros2 run xnodes cam       # OpenCV camera capture
ros2 run xnodes intr      # checkerboard calibration
ros2 run xnodes april     # AprilTag grid detector
```

## Lint & Test

```bash
# Pre-commit (ruff lint + format, hooks)
pre-commit run --all-files

# Manual
ruff check --fix src/
ruff format src/

# Tests
pytest tests/
colcon test --packages-select xnodes
```

Ruff config is in `pyproject.toml`: line-length 120, target py311, `from __future__ import annotations` required in all files.

## Architecture

### Package layout
- `src/xnodes/nodes/` — Standalone ROS 2 nodes (each with `run()` for ros2 entrypoint + `main(tyro.cli(Config))` for direct Python execution)
- `src/xnodes/april/` — AprilTag grid detection, config parsing, detector functions
- `src/xnodes/core/` — Shared utilities: camera info models (Pydantic), quaternion math, calibration metrics
- `src/xnodes/apriltag.py` — AprilTag detection node (OpenCV ArUco-based)
- `launch/` — ROS 2 launch files
- `config/` — YAML configs for stereo rigs, camera intrinsics, AprilTag grids

### Node pattern
Every node follows this structure:
```python
@dataclass
class Config:
    param: str = "default"

class MyNode(Node):
    def __init__(self, cfg: Config | None = None):
        super().__init__("my_node")
        if cfg is None:
            cfg = Config(param=self.declare_parameter("param", "default").value)
        # setup publishers, subscribers, timers

def run(cfg: Config | None = None):
    rclpy.init(); node = MyNode(cfg); rclpy.spin(node)

def main(cfg: Config):
    run(cfg)

if __name__ == "__main__":
    main(tyro.cli(Config))
```

### Topic naming
Nodes scrape the topic prefix from subscriber arguments and republish derived topics under that prefix (e.g. `--sub /video0` → publishes to `/video0/depth/image_raw`).

### Key data flows
- **AprilTag pipeline**: Camera Image → AprilTagDetector → PoseStamped + MarkerArray + TF
- **Stereo anchor**: Pose topics → StereoAnchorNode → TF chain (world → anchor → atf → floater) at 30 Hz, saves transforms to `~/.xnodes/t_world_cam/`
- **Calibration**: Camera Image → chess/AprilGrid detector → corner poses → PCA analysis

## Coding conventions (from AGENTS.md)

- Concise code and docstrings; describe dataclass fields with inline comments, not in docstring
- OOP for components; `config.create()` pattern for building from configs
- Feature-based module organization, shared/core layer for reusable primitives
- `pathlib.Path` over `os.path`, f-strings over `.format()`, `tyro` over `argparse`
- Cyclomatic complexity ≤ 8 (prefer 4-5); keep variable names short
- Avoid excessive nesting and try/except blocks
- QoS: sensor data uses `qos_profile_sensor_data` (best effort, depth=5)
