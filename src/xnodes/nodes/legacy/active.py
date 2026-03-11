from __future__ import annotations

from collections.abc import Callable

from rclpy.node import Node
from std_msgs.msg import Bool


class ActiveFlag:
    """ROS-backed active flag that can be injected into legacy nodes."""

    def __init__(
        self,
        node: Node,
        topic: str = "/xgym/active",
        initial: bool = False,
        toggle: bool = False,
        on_change: Callable[[bool], None] | None = None,
    ):
        self._active = initial
        # toggle_default = False if toggle is None else toggle
        # self._toggle = bool(node.declare_parameter("toggle", toggle_default).value)
        self._toggle = toggle
        self._last_flag = initial
        self._hooks: list[Callable[[bool], None]] = []
        if on_change is not None:
            self._hooks.append(on_change)
        self.publisher = node.create_publisher(Bool, topic, 10)
        self.subscription = node.create_subscription(Bool, topic, self._on_active, 10)

    @property
    def active(self) -> bool:
        return self._active

    def add_hook(self, hook: Callable[[bool], None]) -> None:
        self._hooks.append(hook)

    def add_listener(self, listener: Callable[[bool], None]) -> None:
        self.add_hook(listener)

    def publish(self, active: bool) -> None:
        self.publisher.publish(Bool(data=active))

    def set(self, active: bool) -> None:
        self._active = active
        for hook in self._hooks:
            hook(active)

    def _on_active(self, msg: Bool) -> None:
        if self._toggle:
            if msg.data and not self._last_flag:
                self.set(not self.active)
        else:
            self.set(msg.data)
        self._last_flag = msg.data
