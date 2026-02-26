# Installation

This package is part of the ROS 2 workspace at `~/ws`. Build and install via colcon:

```bash
cd ~/ws
colcon build --packages-select xnodes
source install/setup.bash
```

Dependencies are declared in `package.xml` and resolved by the workspace (rosdep / apt / existing workspace packages). Do not use `pip`, `uv`, or `conda` to manage dependencies for this package.
