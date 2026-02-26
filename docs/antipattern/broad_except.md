# Antipatterns

## Broad `except` blocks

Bad pattern:

```python
try:
    update_state(msg)
except Exception:
    return
```

Why this is harmful:

- It swallows programmer errors (`TypeError`, `AttributeError`, bad assumptions).
- It masks protocol and schema drift in ROS messages.
- It makes incidents non-actionable because logs do not show root cause.
- It can leave node state partially updated and inconsistent.

Preferred pattern:

```python
try:
    payload = parse_msg(msg)
except ValueError as exc:
    self.get_logger().warn(f"Invalid payload: {exc}")
    return
```

Rules for this repo:

- Catch only expected exceptions close to the failing boundary.
- Log enough context (`topic`, message field, key id) before returning.
- Let unexpected exceptions fail fast during development and tests.
- If you need a top-level safety net in a loop, log and re-raise in debug/test paths.

Review checklist:

- Replace `except:` and `except Exception:` unless there is a documented boundary reason.
- Verify the caught exception types are specific and intentional.
- Verify failure paths preserve valid state and observability.
