from __future__ import annotations

import argparse

import numpy as np
from webpolicy import base_policy
from webpolicy.server import Server


class _DummyPolicy(base_policy.BasePolicy):
    def step(self, obs: dict) -> dict:
        del obs
        return {"action": np.zeros((1, 8), dtype=np.float32)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    server = Server(
        policy=_DummyPolicy(),
        host=args.host,
        port=args.port,
        metadata={"server": "policy_stub"},
    )
    server.start()


if __name__ == "__main__":
    main()
