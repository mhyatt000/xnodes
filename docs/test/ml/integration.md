# Integration Test Spec

Integration tests validate end-to-end behavior across components.
They exercise real data flow, minimal stubs, and ensure training and evaluation
pipelines run in a constrained but realistic setting.

# Scope

- Dataset -> dataloader -> model -> loss -> optimizer -> step.
- Checkpoint save/load and resume.
- Evaluation loop with metrics aggregation.
- Config-driven instantiation of a full experiment.

## GPU Assumptions

- Tests require CUDA and should fail fast if unavailable.
- Use small batch sizes and short sequences to limit runtime.

## What To Assert

- Gradients are finite and optimizer updates weights.
- Checkpoint round-trip preserves model outputs.
- Evaluation metrics are measurable and within expected ranges.

## Data And Mocking

- Use a tiny synthetic dataset (e.g. random tokens or gaussian inputs).
- No other mocking . Avoid mocking the training loop, model, or optimizer.

## Required Coverage

- At least one full training step per model family.
- At least one checkpoint resume path.
- At least one evaluation-only run from config.

## Example Structure

- `tests/integration/test_train_step.py`
- `tests/integration/test_checkpoint.py`
- `tests/integration/test_eval.py`

## Example Assertions

- Loss at step 1 < loss at step 0 for fixed seed.
- `state_dict` reload yields identical outputs within tolerance.
- Metrics keys match the configured list.

## Performance Budget

- Each integration test < 30s on a single GPU.
- Total integration suite < 10 minutes.

# Integration test plan (JAX, 4 GPU, 1–2 nodes, Grain + NumPy)

test_train_20_steps_single_process_1gpu
test_checkpoint_roundtrip
test_grain_pipeline_contract
test_pmap_collectives_smoke
test_multinode_2proc_smoke
