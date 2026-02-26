from __future__ import annotations

from xnodes.client.keyboard import KeyboardPolicy


class FakeKeyboardPolicy(KeyboardPolicy):
    def __init__(self, keys: list[str | None]) -> None:
        self._keys = iter(keys)
        self.fd = None
        self.term_state = None

    def _read_key(self) -> str | None:
        return next(self._keys, None)


def test_step_returns_pressed_keys():
    policy = FakeKeyboardPolicy(["up", "w", "up", None])
    assert policy.step() == {"up": True, "w": True}


def test_step_returns_empty_dict():
    policy = FakeKeyboardPolicy([None])
    assert policy.step() == {}
