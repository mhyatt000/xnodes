Below is a compact map of what people test in **ROS 2 launch_testing** and the common helpers they use.

---

# 1️⃣ Process Lifecycle

### What people test

* Node starts successfully
* Node does not crash
* Clean shutdown
* Correct exit code
* Restart behavior (rare but done)

### Tools

```python
proc_info.assertWaitForStartup(process="my_node", timeout=5)
proc_info.assertWaitForShutdown(process="my_node", timeout=5)
proc_info.assertExitCodes(process="my_node")
```

Decorators:

```python
@pytest.mark.launch_test
@launch_testing.post_shutdown_test()
```

---

# 2️⃣ Exit Codes

### What people test

* Node exits with 0
* Node exits with specific error code
* Node crashes when expected

### Tools

```python
proc_info.assertExitCodes(process="my_node")
proc_info.assertExitCodes(process="my_node", allowable_exit_codes=[1])
```

Usually placed in:

```python
@launch_testing.post_shutdown_test()
class TestAfterShutdown:
```

---

# 3️⃣ Log / Output Validation

### What people test

* Startup message printed
* Warning/error logged
* “Ready” message appears
* No unexpected traceback

### Tools

```python
proc_output.assertWaitFor("Ready", timeout=5)
proc_output.assertWaitFor("Error", timeout=5)
proc_output.assertWaitFor("Initialized")
```

You can also search stderr/stdout explicitly.

---

# 4️⃣ Topic Behavior

### What people test

* Node publishes expected topic
* Message contents correct
* Rate correct
* QoS compatibility

### Tools (manual rclpy test node)

```python
rclpy.init()
node = rclpy.create_node("test_node")

# subscribe and assert
self.assertTrue(received_message)
```

Decorators:

```python
@pytest.mark.launch_test
```

---

# 5️⃣ Service Behavior

### What people test

* Service exists
* Service responds correctly
* Timeout handling

Pattern:

```python
client = node.create_client(MySrv, "/my_service")
self.assertTrue(client.wait_for_service(timeout_sec=5))
```

---

# 6️⃣ Action Server Behavior

### What people test

* Action server available
* Goal accepted/rejected
* Feedback published
* Result returned

Uses normal rclpy action client inside test.

---

# 7️⃣ Parameter Configuration

### What people test

* Parameter override works
* Node reads launch parameters correctly
* Dynamic parameter update

Typically:

```python
self.assertEqual(node.get_parameter("rate").value, 10)
```

---

# 8️⃣ Multi-Node Interaction

### What people test

* Node A triggers Node B
* End-to-end pipeline works
* Transform tree available
* Full graph integration

Usually:

* Launch multiple nodes
* Assert final observable behavior

---

# 9️⃣ ROS Graph Properties

Less common but useful:

* Topic exists
* Service exists
* Action exists

You manually inspect:

```python
node.get_topic_names_and_types()
```

---

# 🔟 After-Shutdown Assertions

Used when you want:

* Exit codes checked
* No crash
* Log inspection
* Cleanup verification

Decorator:

```python
@launch_testing.post_shutdown_test()
class TestAfterShutdown(unittest.TestCase):
```

---

# Decorator Summary

| Decorator                              | Purpose                      |
| -------------------------------------- | ---------------------------- |
| `@pytest.mark.launch_test`             | Marks test as launch test    |
| `@launch_testing.post_shutdown_test()` | Run after ROS graph shutdown |

---

# Helper Objects Summary

| Object            | Used For                      |
| ----------------- | ----------------------------- |
| `proc_info`       | Startup, shutdown, exit codes |
| `proc_output`     | Inspect stdout/stderr         |
| `self.assertTrue` | Generic unittest assertion    |
| `pytest` asserts  | Preferred modern style        |

---

# Most Common Real-World Combo

In practice most integration tests contain:

```python
proc_info.assertWaitForStartup(...)
proc_output.assertWaitFor("Ready")
self.assertTrue(received_msg)
proc_info.assertExitCodes(...)
```

---

# Practical Observation (from real repos)

80% of ROS2 launch tests check:

1. Node starts
2. “Ready” printed
3. Topic message received
4. Clean exit

Everything else is specialized.
