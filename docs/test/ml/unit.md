# Unit Test Spec

Unit tests cover single functions or methods with no IO, minimal mocking, and
real tensors on GPU. Keep tests fast (< 1s each), deterministic, and focused on
behavioral contracts that are stable across refactors.

## Scope

- Pure math and tensor ops (losses, metrics, normalization, masking).
- Model components in isolation (single module forward, shape/ dtype rules).
- Config parsing and object construction.
- Utility helpers (schedulers, samplers, data transforms without disk access).

## GPU Assumptions

- All tensor tests run on CUDA by default.
- Use `torch.device("cuda")` and guard with `torch.cuda.is_available()`.
- Prefer small tensor sizes that still exercise kernels (e.g. 8-64 batch).

## What To Assert

- Output shapes and dtypes.
- Determinism with fixed seeds.
- Numerical properties (monotonicity, bounds, invariants).
- Error handling on invalid inputs.

## Data And Mocking

- Use real tensors and minimal stubs only for IO or external services.
- Avoid mocking core ML logic (modules, losses, optimizers).
- Prefer tiny synthetic data sampled from known distributions.

## Required Coverage

- Every public function or class method has at least one unit test.
- Each module forward has a shape test and a basic sanity check.
- Config objects build the expected component type and device placement.

## Example Structure

- `tests/unit/test_losses.py`
- `tests/unit/test_modules_attention.py`
- `tests/unit/test_config.py`

## Example Assertions

- Loss is near zero for identical inputs.
- Masked positions do not affect output.
- Output preserves batch/sequence dimensions.

## Performance Budget

- Total unit test suite < 2 minutes on a single GPU.
- Avoid parameter sweeps; use one representative case.
