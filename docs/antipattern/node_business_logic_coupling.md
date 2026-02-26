# Antipatterns

## Node and business-logic coupling

Bad pattern:

```python
class MyNode(Node):
    def _tick(self):
        # ROS I/O + parsing + policy + transforms + control math in one method
        ...
```

Why this is harmful:

- Hard to test without ROS runtime, timers, and message stubs.
- Business rules become tied to callback timing and transport details.
- Reuse drops: logic cannot be shared by CLI tools, batch jobs, or other nodes.
- Changes become risky because one class owns too many concerns.

Preferred pattern:

- Keep ROS nodes thin: subscriptions, publications, parameters, timers, and wiring.
- Put decision logic in policy/service classes with plain Python inputs/outputs.
- Keep message conversion at boundaries (Node <-> domain objects).
- Unit test policy/service behavior independently from ROS.

Applied example:

- `KeyboardNode` should own `KeyboardPolicy`.
- `KeyboardPolicy.step()` returns a plain dict of pressed keys.
- Node maps that output to ROS `Twist` publishing and shutdown behavior.

Review checklist:

- Can core logic be imported and tested without `rclpy`?
- Are callbacks mostly glue code rather than decision-heavy code?
- Are parameter parsing, domain logic, and transport concerns separated?
