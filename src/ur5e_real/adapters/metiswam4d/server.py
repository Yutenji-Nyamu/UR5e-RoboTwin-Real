"""Local RGB/joint7 model service. Model code never imports a robot controller."""

import logging
import threading
import time
import uuid

import numpy as np

from ..robotwin_pi05.native import add_native_paths
from .contract import decode_actions, finite, validate_runtime_contract


def validate_observation(value):
    state = finite(value["state"], 7)
    if state.shape != (7,) or not 0 <= state[6] <= 1:
        raise ValueError("observation state must be joint7 with commanded gripper in [0, 1]")
    for role in ("head", "wrist"):
        view = np.asarray(value[role])
        if (view.ndim != 3 or view.dtype != np.uint8 or view.shape[-1] != 3
                or min(view.shape[:2]) < 1 or max(view.shape[:2]) > 1920):
            raise ValueError("head/wrist observations must be HWC uint8 RGB, up to 1920 pixels per side")
    return state


class ModelService:
    def __init__(self, predictor, contract, *, status, source_sha256, instance_id=None):
        add_native_paths(model=False)  # Only the existing NumPy/msgpack wire codec is imported.
        from openpi_client import msgpack_numpy

        self.codec = msgpack_numpy
        self.predictor = predictor
        self.lock = threading.Lock()
        self.metadata = {"metis_contract": validate_runtime_contract(contract), "training_status": status,
                         "source_sha256": source_sha256, "instance_id": instance_id or uuid.uuid4().hex}

    def handler(self, socket):
        from websockets.exceptions import ConnectionClosed

        packer = self.codec.Packer()
        try:
            socket.send(packer.pack(self.metadata))
            for message in socket:
                obs = self.codec.unpackb(message)
                if not isinstance(obs, dict) or type(obs.get("request_id")) is not int:
                    raise ValueError("observation request ID is required")
                validate_observation(obs)
                if not self.lock.acquire(blocking=False):
                    raise RuntimeError("model service is busy")
                try:
                    started = time.monotonic()
                    actions = np.asarray(self.predictor(obs), dtype=np.float32)
                    decode_actions(actions)
                    response = {"actions": actions, "request_id": obs["request_id"],
                                "server_elapsed_s": time.monotonic() - started}
                finally:
                    self.lock.release()
                socket.send(packer.pack(response))
        except ConnectionClosed:
            pass
        except Exception:
            logging.exception("Metis request failed")
            try:
                socket.send("Metis request failed; inspect the model server log")
            except ConnectionClosed:
                pass
            socket.close(code=1011)


def serve(args, source):
    from websockets.sync.server import serve as listen
    from .checkpoint import load, verify_vae
    from .policy import WanRgbEncoder

    if not 1 <= args.port <= 65535 or not 1 <= args.rounds <= 1000:
        raise ValueError("invalid port or sampling rounds")
    policy, saved = load(args.checkpoint, source, device=args.device)
    contract = validate_runtime_contract(saved["contract"])
    encoder = WanRgbEncoder(verify_vae(saved, args.vae), device=args.device)

    def predict(obs):
        latent = encoder.encode(obs["head"], obs["wrist"])
        return policy.predict(latent, np.asarray(obs["state"], dtype=np.float32)[None], rounds=args.rounds)[0]

    # Initialize CUDA/VAE/sampling before advertising readiness to the robot client.
    warm = {"head": np.zeros((480, 640, 3), np.uint8), "wrist": np.zeros((480, 640, 3), np.uint8),
            "state": np.asarray([*contract["home_q"], contract["initial_gripper"]], dtype=np.float32)}
    started = time.monotonic()
    decode_actions(predict(warm))
    service = ModelService(predict, contract, status="SFT_not_physical_validation", source_sha256=source)
    with listen(service.handler, "127.0.0.1", args.port, compression=None,
                max_size=16 * 1024 * 1024, close_timeout=0.2) as server:
        print(f"[METIS READY] localhost:{args.port}; warmup={time.monotonic() - started:.3f}s; "
              f"instance={service.metadata['instance_id']}", flush=True)
        server.serve_forever()
