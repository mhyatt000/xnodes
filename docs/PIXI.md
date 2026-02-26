# Pixi Tasks Documentation

This document describes how to use pixi tasks in the TabPi project. Pixi tasks provide a simple, cross-platform way to define and run common project commands.

## What is a Pixi Task?

A pixi task is a named command that can be executed via `pixi run <task-name>`. Tasks are defined in `pixi.toml` and provide:

- **Cross-platform shell support** via `deno_task_shell`
- **Dependency management** (tasks can depend on other tasks)
- **Environment isolation** with custom env vars
- **Working directory control** (cwd)
- **Input/output tracking** for caching

## Task Definition Syntax

All tasks are defined under the `[tasks]` section in `pixi.toml`:

```toml
[tasks.task-name]
cmd = "echo 'Hello, World!'"
depends-on = []        # (optional) list of tasks to run first
cwd = "./"             # (optional) working directory
args = []              # (optional) pass args to the command
env = {}               # (optional) environment variables
inputs = []            # (optional) input files for caching
outputs = []           # (optional) output files/dirs for caching
clean-env = false      # (optional) use clean environment (no inherited vars)
```

## Examples

### Simple Task: Run Tests

```toml
[tasks.test]
cmd = "pytest -vvrs"
```

Run with:
```bash
pixi run test
```

### Task with Dependencies

Create a task that depends on another task (always runs dependencies first):

```toml
[tasks.check]
cmd = "ruff check ."
depends-on = ["test"]  # runs 'test' first, then this task
```

### Task with Custom Environment

```toml
[tasks.libero]
cmd = "uv run TabPFN_Libero.py"
env = { MUJOCO_GL = "osmesa" }
```

The `MUJOCO_GL` var is already set globally via `[activation.env]` in pixi.toml, but task-level env vars override global settings.

### Task with Working Directory Control

```toml
[tasks.src-test]
cmd = "pytest tests/"
cwd = "src/tabpi"      # run from src/tabpi directory
```

### Task with Arguments

Pass static args to the command:

```toml
[tasks.lint]
cmd = "ruff"
args = ["check", "src/", "--select=E,W"]
```

Run with custom args (appended to the task's args):
```bash
pixi run lint -- --ignore=F401
```

### Task with Input/Output Tracking

For long-running tasks, specify inputs and outputs for caching:

```toml
[tasks.build]
cmd = "python scripts/build.py"
inputs = ["src/**/*.py", "pyproject.toml"]
outputs = ["build/"]
```

Pixi tracks these files and can skip the task if inputs haven't changed.

## Running Tasks

### Run a single task

```bash
pixi run test
```

### Run a task with arguments

```bash
pixi run test -- -k integration
```

Everything after `--` is passed to the underlying command.

### List all available tasks

```bash
pixi task list
```

## Best Practices

1. **Keep tasks small and focused** — one job per task
2. **Use depends-on for task composition** — chain simple tasks into workflows
3. **Set cwd only when necessary** — prefer absolute paths in scripts
4. **Document task purpose** — add comments above complex tasks in pixi.toml
5. **Use inputs/outputs for long tasks** — enables caching and incremental builds
6. **Cross-platform compatibility** — use `deno_task_shell` syntax (pixi default)

## Cross-Platform Shell Considerations

Pixi uses `deno_task_shell` by default, which provides cross-platform compatibility:

- Use `/` for paths (works on Windows, Linux, macOS)
- Variables: `$VAR` (not `%VAR%` on Windows)
- Command separator: `&&` for sequential commands
- Pipes: `|` work as expected

Example cross-platform task:

```toml
[tasks.build-and-test]
cmd = "python scripts/build.py && pytest tests/"
```

This runs on Linux, macOS, and Windows without modification.

## TabPi-Specific Tasks

See the active tasks defined in `pixi.toml` for project-specific examples:

- `libero` — Run LIBERO environment tests with MuJoCo

Add new tasks as the project evolves!
