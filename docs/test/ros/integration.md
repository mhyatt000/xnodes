# Integration Test Spec

Integration tests validate end-to-end behavior across ROS nodes. They use
`launch_testing` to start real nodes over the ROS middleware and assert on
topics, TF frames, or node lifecycle — with minimal stubs and realistic message
flow.

## Scope

- Publisher → subscriber data flow (image in, detection out).
- TF frame publication and correctness (stereo anchor chain).
- Node startup, parameter binding, and clean shutdown.
- Config-driven launch of a full pipeline (e.g. `multicam.launch.py`).

## ROS Assumptions

- Tests require a sourced ROS 2 environment; fail fast if `rclpy` is unavailable.
- Run via `colcon test --packages-select xnodes` (uses CTest + launch_testing).
- Use small timeouts and synthetic image frames to limit runtime.

## What To Assert

- Expected topics are published within a timeout.
- TF transforms exist and are geometrically valid (finite, unit quaternion).
- Node shuts down cleanly (exit code 0, no ERROR logs).
- Output messages have the correct `frame_id` and header timestamps.

## Data

- Publish synthetic images (e.g. NumPy checkerboard → `sensor_msgs/Image`).
- No mocks, no stubs — use real nodes and real middleware.
- Minimal launch fixture: start only the nodes under test.

## Required Coverage

- At least one topic round-trip test per node entry point.
- At least one TF frame check for nodes that publish transforms.
- At least one shutdown / lifecycle test confirming clean exit.

## Example Structure

```
tests/integration/test_april_pipeline.py
tests/integration/test_stereo_anchor_tf.py
tests/integration/test_cam_node_lifecycle.py
```

## Example Assertions

- `/detections` receives at least one message within 5 s of image publish.
- `tf_buffer.lookup_transform("world", "anchor", ...)` returns finite translation.
- Node process exits with code 0 after `rclpy.shutdown()`.

## Running Tests

```bash
# Build first
cd ~/ws && colcon build --packages-select xnodes && source install/setup.bash

# Run all tests (CTest dispatches launch_testing)
colcon test --packages-select xnodes
colcon test-result --all --verbose
```

## Performance Budget

- Each integration test < 30s wall time.
- Total integration suite < 5 minutes.
