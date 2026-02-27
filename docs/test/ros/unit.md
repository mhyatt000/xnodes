# Unit Test Spec

Unit tests cover pure Python logic extracted from ROS nodes — no middleware, no
spinning, no topics. Keep tests fast (< 1s each), deterministic, and focused on
behavioral contracts that are stable across refactors.

## Scope

- Pure math and geometry helpers (quaternion ops, PCA, calibration metrics).
- Config parsing and object construction (`Config` dataclasses, Pydantic models).
- Service/policy classes that hold node business logic but have no ROS deps.
- Utility helpers (coordinate transforms, corner extraction, pad masks).

## ROS Assumptions

- No `rclpy.init()` or node spinning. Tests must run without a ROS install on
  `PYTHONPATH` — only pure-Python xnodes modules are imported.
- If a helper transitively imports `rclpy`, refactor it into a ROS-free module
  first (see `docs/antipattern/node_business_logic_coupling.md`).

## What To Assert

- Output shapes and dtypes for array-returning functions.
- Determinism with fixed seeds or fixed inputs.
- Numerical properties (bounds, symmetry, monotonicity).
- Error handling on invalid inputs (wrong shape, out-of-range values).

## Data

- Use real NumPy/OpenCV arrays and tiny synthetic inputs.
- No mocks, no stubs — ever.

## Required Coverage

- Every public function or class method has at least one unit test.
- Every `Config` dataclass builds without error from defaults.
- Every geometry util has a shape test and a known-answer sanity check.

## Example Structure

```
tests/unit/test_quaternion.py
tests/unit/test_calibration_metrics.py
tests/unit/test_config.py
```

## Example Assertions

- `quat_multiply(q, q_inv)` is close to identity.
- `reprojection_error(pts, pts)` is zero.
- `Config()` constructs the expected field types and default values.

## Performance Budget

- Total unit suite < 120s with `pytest tests/unit/`.
- Avoid loops over large parameter grids; use one representative case.
