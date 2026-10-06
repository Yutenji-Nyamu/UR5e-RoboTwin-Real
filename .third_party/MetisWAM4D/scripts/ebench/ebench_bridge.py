"""OpenWAM EBench bridge: policy servers shared between workers, camera rendering only where the policy looks.

Equivalent to benchmarks/ebench/openwam2ebench_interface.py (same observation / action conversion, prompt and
raw-23 proprio via its EBenchOpenWAMDriver) with two changes:

* The action chunk lives in the bridge. The OpenWAM sync executor generates a chunk from the observation seen
  when its buffer is empty and pops one action per request; the server reset only clears that buffer. At each
  replan a worker takes a per-server file lock, resets the server, sends the current observation once and pops
  the whole chunk, so several workers can share one server (~25 GB each) and every replan computes exactly what
  a dedicated server would.
* Steps between replans are sent through GenManip's /step_chunk without camera rendering. The policy ignores
  observations inside a chunk; the only per-step quantity it uses is the base pose of the step before a replan
  (proprio base delta), which the chunk call returns. Per chunk of N actions: N-1 actions via /step_chunk
  (render at its last step only), then the last action via /step, whose full observation drives the next replan.

    PYTHONPATH=<OpenWAM> python scripts/ebench/ebench_bridge.py --url http://127.0.0.1:8087 --run-id X \
        --worker-id 3 --south-port 8851 --ckpt-config <ckpt>/config.yaml [--chunk 32]
"""

import logging
import sys
import time
from collections import deque

import numpy as np
from filelock import FileLock

import benchmarks.ebench.openwam2ebench_interface as iface

logger = logging.getLogger("ebench_bridge")


class ChunkedSouth:
    """WSPolicyClient stand-in that fetches whole chunks atomically and serves them locally."""

    def __init__(self, client, lock_path: str, chunk: int):
        self._client = client
        self._lock = FileLock(lock_path)
        self._chunk = chunk
        self._buffer: deque = deque()

    def reset(self) -> dict:
        self._buffer.clear()
        return {"type": "reset_ack"}

    def predict(self, payload: dict) -> dict:
        if not self._buffer:
            with self._lock:
                ack = self._client.reset()
                if ack.get("type") != "reset_ack":
                    raise RuntimeError(f"policy server reset not acknowledged: {ack}")
                self._buffer.extend(self._client.predict(payload) for _ in range(self._chunk))
        return self._buffer.popleft()

    def __getattr__(self, name):
        return getattr(self._client, name)


def run_worker(args, chunk: int) -> None:
    from genmanip_client import EvalClient

    wid = str(args.worker_id)
    south = ChunkedSouth(
        iface.WSPolicyClient(f"ws://{args.south_host}:{args.south_port}", timeout=float(args.request_timeout)),
        f"/tmp/ebench_policy_{args.south_port}.lock",
        chunk,
    )
    iface.wait_until_healthy(south)
    driver = iface.EBenchOpenWAMDriver(south, send_state=not args.no_send_state)

    def make_client():
        client = EvalClient(args.url, worker_ids=[wid], token=args.token or None, run_id=args.run_id,
                            save_process=args.save_process, verbose=False)
        base_params = client._build_params
        # One render at the end of each /step_chunk: path tracing without accumulation needs no catch-up frames.
        client._build_params = lambda *a, **k: {**base_params(*a, **k), "render_mode": "lite", "subframes": 0,
                                                "render_suffix": 0}
        return client

    def inner_of(obs):
        return ((obs or {}).get(wid) or {}).get("obs")

    client = make_client()
    failures = 0
    obs, done = client.reset(), False
    try:
        while not done:
            inner = inner_of(obs)
            if inner is None:
                logger.warning("no actionable obs for worker %s; leaving the loop", wid)
                break
            try:
                # The first call replans (and starts the episode on a reset obs); the rest pop the chunk.
                cont = dict(inner, reset=False)
                actions = [driver.act(inner)] + [driver.act(cont) for _ in range(chunk - 1)]
                mid, done = client.step({wid: actions[:-1]})
                mid_inner = inner_of(mid)
                if done or mid_inner is None or mid_inner.get("reset", False):
                    obs = mid  # episode ended inside the chunk; replan from the new episode
                    continue
                driver._prev_base = np.asarray(mid_inner[iface.STATE_BASE_KEY], dtype=np.float64).reshape(3)
                obs, done = client.step({wid: actions[-1]})
            except Exception as e:  # noqa: BLE001
                failures += 1
                if failures > getattr(args, "max_reconnects", 20):
                    raise RuntimeError(f"worker {wid}: {failures} step failures, last: {e}") from e
                logger.warning("worker %s step failed (%s); rebuilding client (%d)", wid, e, failures)
                time.sleep(args.client_reinit_backoff * failures)
                try:
                    client.close()
                except Exception:  # noqa: BLE001
                    pass
                client = make_client()
                obs = client.reset()
                driver.invalidate_episode()
                south.reset()
        logger.info("run complete: %d episodes, %d policy steps", driver.episodes, driver.steps)
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
        south.close()


def main() -> None:
    chunk = 32
    if "--chunk" in sys.argv:
        i = sys.argv.index("--chunk")
        chunk = int(sys.argv[i + 1])
        del sys.argv[i : i + 2]
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    args = iface.parse_args_with_config()
    iface.verify_ckpt_config(args)
    run_worker(args, chunk)


if __name__ == "__main__":
    main()
