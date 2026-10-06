"""Opt-in real loopback tests of the Metis service and bounded robot client."""

import os
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from ur5e_real.adapters.metiswam4d.client import PolicyClient
from ur5e_real.adapters.metiswam4d.data import prepare, read_dataset
from ur5e_real.adapters.metiswam4d.server import ModelService
from test_metiswam4d import episode

pytestmark = pytest.mark.skipif(os.environ.get("UR5E_TEST_LOCAL_RPC") != "1", reason="opt-in loopback check")


@pytest.fixture
def contract(tmp_path):
    run, _ = episode(tmp_path)
    prepare(tmp_path, [run], "drawer", tmp_path / "dataset")
    return read_dataset(tmp_path / "dataset")[0]


def observation():
    return {"state": np.zeros(7, np.float32), "head": np.zeros((480, 640, 3), np.uint8),
            "wrist": np.full((480, 640, 3), 255, np.uint8)}


class LocalServer:
    def __init__(self, contract, *, delay=0, bad_shape=False, bad_id=False):
        self.delay, self.bad_shape, self.bad_id = delay, bad_shape, bad_id
        self.service = ModelService(self.predict, contract, status="diagnostic", source_sha256="test")

    def predict(self, obs):
        time.sleep(self.delay)
        assert obs["head"].shape == (480, 640, 3) and obs["wrist"].mean() == 255
        if self.bad_id:
            obs["request_id"] += 1
        return np.tile(obs["state"], (49 if self.bad_shape else 50, 1))

    def __enter__(self):
        from websockets.sync.server import serve

        self.server = serve(self.service.handler, "127.0.0.1", 0, compression=None,
                            max_size=16 * 1024 * 1024, close_timeout=0.1)
        self.port = self.server.socket.getsockname()[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.thread.join(2)
        assert not self.thread.is_alive()


def test_real_round_trip_with_two_rgb_frames(contract):
    with LocalServer(contract) as server, PolicyClient(contract, port=server.port) as client:
        for i in (1, 2):
            result = client.infer(observation())
            assert result["request_id"] == i and result["actions"].shape == (50, 7)
            assert result["client_elapsed_s"] < 1.5


def test_native_tiny_action_model_through_same_service_and_client(contract):
    torch = pytest.importorskip("torch")
    from ur5e_real.adapters.metiswam4d.__main__ import _diagnostic_latent
    from ur5e_real.adapters.metiswam4d.native import build_model, load_source, model_config
    from ur5e_real.adapters.metiswam4d.policy import ActionPolicy

    load_source(Path(__file__).resolve().parents[1] / ".third_party/MetisWAM4D")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    policy = ActionPolicy(build_model(model_config(tiny=True)), contract["stats"])
    def predict(obs):
        latent = _diagnostic_latent(obs["head"], obs["wrist"], "cpu")
        return policy.predict(latent, obs["state"][None], rounds=4)[0]
    server = LocalServer(contract)
    server.service.predictor = predict
    with server, PolicyClient(contract, port=server.port) as client:
        first = client.infer(observation())["actions"]
        changed = client.infer(dict(observation(), head=np.full((480, 640, 3), 255, np.uint8)))["actions"]
        assert first.shape == (50, 7) and np.isfinite(first).all()
        assert np.max(np.abs(first[:, :6] - changed[:, :6])) > 0


def test_mismatched_dataset_rejected_at_handshake(contract):
    with LocalServer(dict(contract, task="another task")) as server:
        with pytest.raises(ValueError, match="contracts differ"):
            PolicyClient(contract, port=server.port)


@pytest.mark.parametrize("option,exception", [("delay", TimeoutError), ("bad_shape", RuntimeError), ("bad_id", ValueError)])
def test_late_or_invalid_predictions_close_connection(contract, option, exception):
    with LocalServer(contract, **{option: 0.25 if option == "delay" else True}) as server:
        with PolicyClient(contract, port=server.port, timeout_s=0.1) as client:
            with pytest.raises(exception):
                client.infer(observation())
            assert client.socket is None
