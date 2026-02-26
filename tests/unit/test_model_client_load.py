from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types


def _stub_dependencies(monkeypatch):
    cv_bridge = types.ModuleType("cv_bridge")

    class CvBridge:
        pass

    cv_bridge.CvBridge = CvBridge
    monkeypatch.setitem(sys.modules, "cv_bridge", cv_bridge)

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *args, **kwargs: None
    rclpy.spin = lambda *args, **kwargs: None
    rclpy.shutdown = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)

    rclpy_node = types.ModuleType("rclpy.node")

    class Node:
        pass

    rclpy_node.Node = Node
    monkeypatch.setitem(sys.modules, "rclpy.node", rclpy_node)

    rclpy_qos = types.ModuleType("rclpy.qos")

    class QoSProfile:
        def __init__(self, *args, **kwargs):
            pass

    class ReliabilityPolicy:
        BEST_EFFORT = "BEST_EFFORT"

    rclpy_qos.QoSProfile = QoSProfile
    rclpy_qos.ReliabilityPolicy = ReliabilityPolicy
    monkeypatch.setitem(sys.modules, "rclpy.qos", rclpy_qos)

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")

    class CompressedImage:
        pass

    class JointState:
        pass

    sensor_msgs_msg.CompressedImage = CompressedImage
    sensor_msgs_msg.JointState = JointState
    sensor_msgs.msg = sensor_msgs_msg
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")

    class Bool:
        pass

    class Float32MultiArray:
        pass

    std_msgs_msg.Bool = Bool
    std_msgs_msg.Float32MultiArray = Float32MultiArray
    std_msgs.msg = std_msgs_msg
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg)

    xarm_msgs = types.ModuleType("xarm_msgs")
    xarm_msgs_msg = types.ModuleType("xarm_msgs.msg")

    class RobotMsg:
        pass

    xarm_msgs_msg.RobotMsg = RobotMsg
    xarm_msgs.msg = xarm_msgs_msg
    monkeypatch.setitem(sys.modules, "xarm_msgs", xarm_msgs)
    monkeypatch.setitem(sys.modules, "xarm_msgs.msg", xarm_msgs_msg)

    jax = types.ModuleType("jax")
    jax.tree = types.SimpleNamespace(map=lambda fn, x: x)
    monkeypatch.setitem(sys.modules, "jax", jax)

    openpi_client = types.ModuleType("openpi_client")
    wcp = types.ModuleType("openpi_client.websocket_client_policy")

    class WebsocketClientPolicy:
        def __init__(self, *args, **kwargs):
            pass

        def infer(self, _payload):
            return {}

    wcp.WebsocketClientPolicy = WebsocketClientPolicy
    openpi_client.websocket_client_policy = wcp
    monkeypatch.setitem(sys.modules, "openpi_client", openpi_client)
    monkeypatch.setitem(sys.modules, "openpi_client.websocket_client_policy", wcp)


def test_model_client_module_loads(monkeypatch):
    _stub_dependencies(monkeypatch)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "src"))
    monkeypatch.delitem(sys.modules, "xnodes.nodes.xgym_legacy.base", raising=False)
    monkeypatch.delitem(sys.modules, "xnodes.nodes.xgym_legacy.model", raising=False)

    model = importlib.import_module("xnodes.nodes.xgym_legacy.model")

    assert isinstance(model.NOMODEL, model.ModelClientConfig)
    assert issubclass(model.MyClient, model.wcp.WebsocketClientPolicy)
