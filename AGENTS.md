# AGENTS.md

This file provides guidance when working with code in this repository.
If you find anything surprising or confusion, please raise this with your developer ask to add a note here
so future agent readers can benefit from your experience.

me proactive about raising surprises and antipatterns, even though we might not accept all proposed
changes to the AGENTS.md

# Surprises

## Pixi
this repo uses `colcon` to build the ROS package, not `pip` or `uv` or `conda`. The ROS workspace is
at `~/ws`, and this package lives at `~/ws/src/xnodes`. Dependencies are declared in `package.xml`
and resolved by the workspace build.

look in `./envrc` and note that we use pixi to manage ros and other packages.
any command that uses ros should be prepended with `source .envrc` and `pixi run ros...` to ensure the correct environment is loaded. For example, to run a node:

```bash
source .envrc
pixi run ros2 run xnodes model
```

to see the defined pixi tasks use `pixi run` with no args

## ruff
Prefer `uvx ruff ...` over other invocations of ruff

## Workspace sync
this repo is often synced or symlinked into the ROS workspace install/use path, and commands may import the
workspace-installed `xnodes` package instead of the checkout you are editing. Do not set `PYTHONPATH`
manually to work around that. Prefer rebuilding/resyncing the workspace so runtime verification uses the
intended code.

## ros bags

many nodes in the `legacy` folder are getting transitioned to production. Writer node truly is
legacy and should eventually be replaced with ros bagging as soon as is convenient.

# main

we use ros2 humble with python 3.11.

## docs

Use `tree -a docs` for the local docs map:
since it is dynamic and growing, it is not listed here.
also use `tree -i __pycache__ -P "*.pyc" src` to see the code structure without pyc files.

for intelligent traversal, sometimes `find src -type f -name "*.py"` is better.

it has notes on design decisions, coding style, ROS testing, and anti-patterns to avoid.

notably: `docs/PIXI.md` - How to define and run Pixi tasks, with syntax, examples, and best practices.

## What is xnodes

A ROS 2 Python package for robot learning experiments.
includes nodes and utilities for
spatial perception and camera calibration. stereo camera anchoring,

Targets ROS 2 Humble+ on Python 3.11.

This package lives at `~/ws/src/xnodes` and is built inside the ROS 2 workspace at `~/ws`. Dependencies are declared in `package.xml` and resolved by the workspace — do not use `pip`, `uv`, or `conda` to manage dependencies.

## Environment

A `.envrc` (direnv) in this repo sets `PIXI_PROJECT_MANIFEST=~/ws/pixi.toml`, so `pixi shell` works from this directory even though `pixi.toml` lives at `~/ws/`.

## debugging

- a good agent who is debugging ros should be using `ros2 topic echo/list/info`
- write logging where appropriate. it will help the dev.

- Always debug in the repo’s ROS environment: source .envrc and then run pixi run ros2 ..., since the workspace/pixi setup is required
- Start with ROS graph inspection, not code edits: ros2 node list, ros2 node info, ros2 topic list, ros2 topic info -v, ros2 topic echo, and ros2 topic hz.
- Check QoS compatibility when image/sensor topics look “dead”; this code uses mixed qos. sometimes best-effort sometimes others.
- Verify topic-prefix contracts before assuming a publisher is broken.
- Debug TF explicitly because it is a common failure point: ros2 run tf2_tools, ros2 run tf2_ros tf2_echo, and ros2 run tf2_ros view_frames.
- Check parameter semantics early with ros2 param list/get
- When relevant, suggest that we reproduce bugs with bag playback or launch tests instead of ad-hoc manual steps.
- Add targeted logging at callback boundaries: topic name, frame_id, timestamp, message sizes, and explicit drop reasons. That fits the repo’s existing logging style better than silent failures.

- Check /xgym/active first. A lot of the stack is gated on that flag. Base defines and owns it. base.py:23
- `ros2 node info` is one of the most useful single command because it shows pubs/subs per node, which lets you walk the chain: /a/b -> /c/d -> /c/e
- Document startup-order debugging for nodes.
- Add hardware-device checks as first-class debugging steps. some nodes expect a fixed /dev/input/by-id/... path and grabs the device. verify those device paths and permissions before blaming ROS.
- Prefer topic-rate and payload-shape checks for legacy control loops. recommend using logs to confirm joint counts, names, and command magnitudes instead of blind prints.

legacy topic tracing, active-state debugging, startup-order/camera discovery, device-path verification, model input completeness, and writer artifact inspection.

- use subagents proactively if you need to debug or inspect multiple streams at a time

## Lint & Test

```bash
# Pre-commit (ruff lint + format, hooks)
pre-commit run --all-files

# Manual
ruff check --fix src/
ruff format src/

# Tests
pytest tests/
# colcon test --packages-select xnodes # TODO it is not working yet...
```

### Topic naming
Nodes scrape the topic prefix from subscriber arguments and republish derived topics under that prefix (e.g. `--sub /video0` → publishes to `/video0/depth/image_raw`).

### Key data flows
use `ros2 topic echo` and `ros2 node info` to inspect these streams at runtime:

## Coding conventions (from AGENTS.md)

- Concise code and docstrings; describe dataclass fields with inline comments, not in docstring
- OOP for components; `config.create()` pattern for building from configs
- Feature-based module organization, shared/core layer for reusable primitives
- `pathlib.Path` over `os.path`, f-strings over `.format()`
- Cyclomatic complexity ≤ 8 (prefer 4-5); keep variable names short
- Avoid excessive nesting and try/except blocks
- QoS: sensor data uses `qos_profile_sensor_data` (best effort, depth=5)
