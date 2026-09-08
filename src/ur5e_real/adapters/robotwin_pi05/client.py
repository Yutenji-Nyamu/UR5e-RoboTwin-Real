"""Bounded local native-protocol RPC client. No JAX/PyTorch imports on the robot side."""

from __future__ import annotations

import inspect
import socket
import threading
import time

import numpy as np

from .contract import decode_actions, require_same_contract
from .native import add_native_paths


class PolicyClient:
    def __init__(self, contract: dict, *, port=8005, timeout_s=1.5):
        if not np.isfinite(timeout_s) or not 0 < timeout_s <= 1.5:
            raise ValueError("RPC timeout must be in (0, 1.5] seconds, below the robot command deadline")
        if not 1 <= port <= 65535:
            raise ValueError("invalid local server port")
        add_native_paths(model=False)
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        self.timeout_s = timeout_s
        self.codec = msgpack_numpy
        self.packer = msgpack_numpy.Packer()
        options = {"proxy": None} if "proxy" in inspect.signature(connect).parameters else {}
        self.socket = connect(
            f"ws://127.0.0.1:{port}",
            open_timeout=timeout_s,
            close_timeout=0.2,
            compression=None,
            max_size=16 * 1024 * 1024,
            **options,
        )
        try:
            self.metadata = self._receive()
            require_same_contract(contract, self.metadata.get("ur5e_contract", {}))
        except BaseException:
            self.close()
            raise

    def _receive(self):
        response = self.socket.recv(timeout=self.timeout_s)
        if isinstance(response, str):
            raise RuntimeError("native policy server returned an error; inspect the model server log")
        result = self.codec.unpackb(response)
        if not isinstance(result, dict):
            raise ValueError("policy server response is not a mapping")
        return result

    def infer(self, observation):
        started = time.monotonic()
        completed = threading.Event()
        reply = {}

        def exchange():
            try:
                self.socket.send(self.packer.pack(observation))
                reply["result"] = self._receive()
            except BaseException as exc:
                reply["error"] = exc
            finally:
                completed.set()

        try:
            worker = threading.Thread(target=exchange, name="pi05-rpc", daemon=True)
            worker.start()
            if not completed.wait(self.timeout_s):
                # recv(timeout) alone does not bound a blocked send of camera images.
                try:
                    self.socket.socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                raise TimeoutError("policy request exceeded its end-to-end deadline")
            if "error" in reply:
                raise reply["error"]
            result = reply["result"]
            decode_actions(result["actions"])
            result["client_elapsed_s"] = time.monotonic() - started
            if result["client_elapsed_s"] > self.timeout_s:
                raise TimeoutError("policy result exceeded its end-to-end deadline")
            return result
        except BaseException:
            self.close()  # A late response must never be mistaken for the next request.
            raise

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
