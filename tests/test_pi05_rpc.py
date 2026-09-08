"""Opt-in loopback-only tests; never import a hardware controller or model weights."""

import asyncio
import os
import threading
import time

import numpy as np
import pytest

from ur5e_real.adapters.robotwin_pi05.client import PolicyClient
from ur5e_real.adapters.robotwin_pi05.native import REPOSITORY, add_native_paths
from test_pi05 import contract

pytestmark = pytest.mark.skipif(
    os.environ.get("UR5E_TEST_LOCAL_RPC") != "1" or not (REPOSITORY / ".third_party/RoboTwin/policy/pi05").is_dir(),
    reason="opt-in loopback native protocol check",
)


class LocalServer:
    def __init__(self, metadata, *, delay=0):
        self.metadata, self.delay = metadata, delay
        self.ready, self.stop = threading.Event(), threading.Event()
        self.error = None

    def __enter__(self):
        add_native_paths(model=False)
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        from websockets.asyncio.server import serve

        class FakePolicy:
            def infer(policy, obs):
                time.sleep(self.delay)
                return {"actions": np.tile(obs["state"], (50, 1))}

        server = WebsocketPolicyServer(FakePolicy(), metadata=self.metadata)

        async def run():
            async with serve(server._handler, "127.0.0.1", 0, compression=None) as websocket:
                self.port = websocket.sockets[0].getsockname()[1]
                self.ready.set()
                while not self.stop.is_set():
                    await asyncio.sleep(0.01)

        def runner():
            try:
                asyncio.run(run())
            except BaseException as exc:
                self.error = exc
                self.ready.set()

        self.thread = threading.Thread(target=runner, daemon=True)
        self.thread.start()
        assert self.ready.wait(2)
        if self.error:
            raise self.error
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(2)
        assert not self.thread.is_alive()


def test_real_native_server_and_robot_client_round_trip():
    with LocalServer({"ur5e_contract": contract()}) as server:
        with PolicyClient(contract(), port=server.port) as client:
            result = client.infer({"state": np.zeros(14, dtype=np.float32)})
            assert result["actions"].shape == (50, 14)
            assert result["client_elapsed_s"] < 1.5


def test_wrong_action_space_rejected_during_handshake():
    with LocalServer({"ur5e_contract": contract() | {"action_space": "tcp_pose"}}) as server:
        with pytest.raises(ValueError, match="action_space"):
            PolicyClient(contract(), port=server.port)


def test_slow_response_closes_client_instead_of_using_late_actions():
    with LocalServer({"ur5e_contract": contract()}, delay=0.2) as server:
        with PolicyClient(contract(), port=server.port, timeout_s=0.05) as client:
            with pytest.raises(TimeoutError):
                client.infer({"state": np.zeros(14, dtype=np.float32)})
            assert client.socket is None
