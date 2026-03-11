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
        on_change: Callable[[bool], None] | None = None,
    ):
        self._active = initial
        self._listeners: list[Callable[[bool], None]] = []
        if on_change is not None:
            self._listeners.append(on_change)
        self.publisher = node.create_publisher(Bool, topic, 10)
        self.subscription = node.create_subscription(Bool, topic, self._on_active, 10)

    @property
    def active(self) -> bool:
        return self._active

    def add_listener(self, listener: Callable[[bool], None]) -> None:
        self._listeners.append(listener)

    def publish(self, active: bool) -> None:
        self.publisher.publish(Bool(data=active))

    def set(self, active: bool) -> None:
        self._active = active
        for listener in self._listeners:
            listener(active)

    def _on_active(self, msg: Bool) -> None:
        self.set(msg.data)
