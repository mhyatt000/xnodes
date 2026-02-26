from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
import threading
import time

# import keyboard
import numpy as np
import pynput
from pynput.keyboard import Key

# spacemouse imports
import pyspacemouse
from pyspacemouse.pyspacemouse import SpaceNavigator


def default(x):
    return field(default_factory=lambda: x)


@dataclass
class SpaceMouseConfig:
    # sensitivity
    scale = default(
        np.array(
            [
                5.0,
                5.0,
                5.0,
                0.02,
                0.02,
                0.02,
                100,
            ]
        )
    )
    flip = default([1, 1, 1, -1, -1, -1, 1])
    order = default([0, 1, 2, 3, 5, 4, 6])


class VelocitySMC(SpaceMouseConfig):
    # xyz, rpy, gripper
    # yxz, rpy, gripper
    # rpy is rotation along x, y, z
    scale = default(np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 0.1 * 850]))

    order = default([1, 0, 2, 5, 3, 4, 6])
    # applied after order
    flip = default([-1, 1, 1, -1, -1, -1, 1])
    sensitivity = 1.5  # from 1 -> inf


class PositionSMC(SpaceMouseConfig):
    # xyz, rpy, gripper
    # yxz, rpy, gripper
    # rpy is rotation along x, y, z
    scale = default(np.array([1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 0.1]))

    order = default([1, 0, 2, 5, 3, 4, 6])
    # applied after order
    flip = default([-1, 1, 1, -1, -1, -1, 1])
    sensitivity = 1.5  # from 1 -> inf


def ema_2d(data, window):
    """Calculate EMA for 2D array"""

    alpha = 2 / (window + 1)  # Smoothing factor
    ema = np.empty_like(data)
    ema[0, :] = data[0, :]  # Start with first row

    for i in range(1, data.shape[0]):
        ema[i, :] = alpha * data[i, :] + (1 - alpha) * ema[i - 1, :]

    return ema


class SpaceMouseController:  # from 1 -> inf
    def __init__(self, cfg: SpaceMouseConfig = VelocitySMC()):
        self.cfg = cfg
        self.state: SpaceNavigator | None = SpaceNavigator(
            t=0.0,
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            buttons=(0, 0),
        )

        self._running = False
        self.thread = None

        self.freq = 300  # Frequency Hz
        self.dt = 1.0 / self.freq

        self.hist = np.zeros((self.freq, 7))

        self.success = pyspacemouse.open(
            dof_callback=self.set_state,
            button_callback=None,
            button_callback_arr=[],
        )

        if self.success:
            print("SpaceMouse connected.")
        else:
            print("Failed to connect to SpaceMouse.")

        self.start()

    def start(self):
        """Start reading the SpaceMouse in the background."""
        if not self._running and self.success:
            self._running = True
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
        time.sleep(0.1)

    def _read(self):
        """Read input in a loop and update the last state."""
        while self._running and self.success:
            pyspacemouse.read()
            time.sleep(self.dt)

    def stop(self):
        """Stop reading and close the SpaceMouse."""
        if self._running:
            self._running = False
            if self.thread is not None:
                self.thread.join()  # Wait for the thread to finish
                del self.thread
            pyspacemouse.close()  # Close the SpaceMouse device when stopping
            print("SpaceMouse disconnected.")

    def set_state(self, state):
        self.state = state

    def read(self, as_dict=False):
        if as_dict:
            print(type(self.state))
            return self.state._asdict()

        else:  # x,y,z,roll,pitch,yaw,gripper
            gripper = self.state.buttons
            gripper = -1 * gripper[0] + gripper[1]
            out = np.array(
                [
                    self.state.x,
                    self.state.y,
                    self.state.z,
                    self.state.pitch,
                    self.state.yaw,
                    self.state.roll,
                    gripper,
                ]
            )
            out = out[self.cfg.order]
            # out = out ** self.cfg.sensitivity * np.sign(out) # make it less sensitive
            out = out * self.cfg.scale * self.cfg.flip

            self.hist = np.roll(self.hist, -1, axis=0)  # smooth
            self.hist[-1] = out
            # out = ema_2d(self.hist, 10)[-1]

            # out = np.where(out > self.hist[-2], out, 0)
            return out


class Controller(ABC):
    def __init__(self):
        pass


class KeyboardController:
    def __init__(self):
        self.vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.running = True
        self.keys_pressed = set()
        self.listener = pynput.keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)

        self.size = 10.0
        self.rsize = 0.05
        self.gsize = 50.0

        self.funcs = {}

        self.run()  # Begin the run method at the end of initialization

    def on_key_press(self, key):
        if isinstance(key, pynput.keyboard._xorg.KeyCode):
            key = key.char

        if key in self.funcs:
            self.funcs[key]()
        self.keys_pressed.add(key)
        self.update_vector()

    def on_key_release(self, key):
        if isinstance(key, pynput.keyboard._xorg.KeyCode):
            key = key.char
        self.keys_pressed.discard(key)
        self.update_vector()

    def update_vector(self):
        self.vec = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]  # Reset vec to 0 at the beginning

        shift_pressed = Key.shift in self.keys_pressed or Key.shift_r in self.keys_pressed

        if Key.up in self.keys_pressed:
            if shift_pressed:
                self.vec[4] += self.rsize  # Increase ry value
            else:
                self.vec[1] += self.size  # Increase y value
        if Key.down in self.keys_pressed:
            if shift_pressed:
                self.vec[4] -= self.rsize  # Decrease ry value
            else:
                self.vec[1] -= self.size  # Decrease y value
        if Key.left in self.keys_pressed:
            if shift_pressed:
                self.vec[3] -= self.rsize  # Decrease rx value
            else:
                self.vec[0] -= self.size  # Decrease x value
        if Key.right in self.keys_pressed:
            if shift_pressed:
                self.vec[3] += self.rsize  # Increase rx value
            else:
                self.vec[0] += self.size  # Increase x value

        if any(hasattr(k, "char") and k.char == "w" for k in self.keys_pressed):
            self.vec[2] += self.size  # Increase z value
        if any(hasattr(k, "char") and k.char == "s" for k in self.keys_pressed):
            self.vec[2] -= self.size  # Decrease z value
        if any(hasattr(k, "char") and k.char == "e" for k in self.keys_pressed):
            self.vec[5] += self.rsize  # Increase rz value
        if any(hasattr(k, "char") and k.char == "d" for k in self.keys_pressed):
            self.vec[5] -= self.rsize  # Decrease rz value

        if any(hasattr(k, "char") and k.char == "m" for k in self.keys_pressed):
            self.vec[6] += self.gsize  # Increase gripper value (open)
        if any(hasattr(k, "char") and k.char == "n" for k in self.keys_pressed):
            self.vec[6] -= self.gsize  # Decrease gripper value (close)

        self.vec = np.array(self.vec)

    def run(self):
        print("running")
        thread = threading.Thread(target=self.listener.start, daemon=True)
        thread.start()
        """
        try:
            while True:
                # print(self.vec)  # Continuously print the vector to see the changes
                time.sleep(0.1)  # Print every 0.1 second
        except KeyboardInterrupt:
            self.running = False
            self.listener.stop()
        """

    def close(self):
        self.running = False
        self.listener.stop()

    def __call__(self, *args, **kwargs):
        return self.vec

    def register(self, key, func):
        self.funcs[key] = func


# kb = KeyboardController()
# while True:
# time.sleep(0.1)


class ScriptedController(Controller):
    def __init__(self):
        self.action = None

    def __call__(self, obs):
        return self.action

    def update(self, obs, reward, truncated, terminated, info):
        self.action = 1


def main():
    sm = SpaceMouseController()
    while True:
        print(sm.read().round(4))
        time.sleep(0.1)


if __name__ == "__main__":
    main()
