"""Small CPU models and fake episodes only: these tests never connect robot hardware."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from ur5e_real.adapters.rlinf_rlt.actions import ActionCodec, proprio
from ur5e_real.adapters.rlinf_rlt.cache import build_cache, load_cache
from ur5e_real.adapters.rlinf_rlt.checkpoint import load, restore_rng, rng_state
from ur5e_real.adapters.rlinf_rlt.config import configuration, FEATURE_VERSION
from ur5e_real.adapters.rlinf_rlt.learner import RLTTrainer, masked_bc, td_target
from ur5e_real.adapters.rlinf_rlt.replay import load_replay, label_episode, summarize
from ur5e_real.adapters.rlinf_rlt.rounds import collect_round
from ur5e_real.adapters.rlinf_rlt.run import begin_round, decide, load_run
from ur5e_real.adapters.rlinf_rlt.serve import RLTPolicy
from ur5e_real.adapters.rlinf_rlt.storage import atomic_json, atomic_npz, digest, read_json, lease
from ur5e_real.adapters.rlinf_rlt.train import train_heads, train_token, load_encoder
from ur5e_real.adapters.rlinf_rlt.vendor.rlt_token_transformer import RLTTokenTransformer
from ur5e_real.adapters.robotwin_pi05.contract import STATE_LAYOUT


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    return configuration(
        {
            "token": {
                "input_dim": 8,
                "embed_dim": 8,
                "prefix_seq_len": 4,
                "num_layers": 1,
                "num_heads": 2,
                "mlp_ratio": 1.0,
            },
            "token_train": {
                "steps": 2,
                "diagnostic_interval": 1,
                "save_interval": 1,
                "warmup_steps": 0,
                "batch_size": 2,
            },
            "algorithm": {"hidden_dim": 16, "batch_size": 2},
            "real": {"action_steps": 2, "max_chunks": 2},
            "save": {"keep_last": 2},
        }
    )


def obs(value=0.0):
    state = np.zeros(14, dtype=np.float32)
    state[:6] = state[7:13] = value
    return {
        "state": state,
        "prompt": "test cube",
        "executor_aux": np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32),
        "images": {"cam_high": np.zeros((3, 4, 4), np.uint8), "cam_right_wrist": np.zeros((3, 4, 4), np.uint8)},
    }


class FakeFeatures:
    def infer(self, observation, *, noise=None):
        raw = np.zeros((50, 32), dtype=np.float32) + 0.05
        raw[:, 13] = -0.5
        state = np.zeros(32, dtype=np.float32)
        physical = raw[:, :14].copy() * 1.0000005 + 0.0000005
        physical[:, :6] += observation["state"][:6]
        prefix = np.arange(32, dtype=np.float32).reshape(4, 8) * 0.01 + observation["state"][0]
        return {
            "actions": physical,
            "rlt_features": {
                "version": FEATURE_VERSION,
                "prefix": prefix,
                "mask": np.asarray([True, True, False, True]),
                "normalized_state": state,
                "raw_reference": raw,
            },
        }


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    atomic_npz(dataset / "ur5e_adapter" / "demo.npz", vectors=np.zeros((5, 14)))
    atomic_json(
        dataset / "ur5e_adapter" / "assets" / STATE_LAYOUT / "norm_stats.json",
        {"norm_stats": {"actions": {"q01": [-1.0] * 32, "q99": [1.0] * 32}}},
    )
    run = {
        "version": 1,
        "config": tiny_config(),
        "base_id": "fake-base",
        "dataset": str(dataset),
        "checkpoint": str(tmp_path / "base"),
        "contract": {"run_ids": ["demo"]},
        "lab_config": "unused",
    }
    run["run_id"] = digest(run)
    directory = tmp_path / "run"
    atomic_json(directory / "run.json", run)
    atomic_json(
        directory / "state.json",
        {
            "stage": "cache",
            "selected_token": None,
            "selected_head": None,
            "latest_token": None,
            "latest_head": None,
            "round": None,
            "review_pending": False,
        },
    )
    monkeypatch.setattr(
        "ur5e_real.adapters.rlinf_rlt.cache.offline_observation", lambda _d, _r, i: (obs(i * 0.01), None)
    )
    return directory


@pytest.mark.parametrize(
    "overrides",
    [
        {"real": {"episodes_per_round": 8}},
        {"token": {"num_heads": 0}},
        {"real": {"action_steps": 2.5}},
        {"algorithm": {"gamma": float("nan")}},
        {"save": {"every_update": False}},
        {"token_train": {"batch_size": 0}},
        {"algorithm": {"learning_rate": -1}},
        {"real": {"release_hold_s": 0.2}},
        {"unknown": 1},
    ],
)
def test_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        configuration(overrides)


def test_exact_donor_token_source():
    root = Path(__file__).resolve().parents[1]
    path = root / "src/ur5e_real/adapters/rlinf_rlt/vendor/rlt_token_transformer.py"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "d72364f58436af345164d9bb574d7f7edf054e5c9d1e94d367ae075b53779332"
    )


def test_action_decode_once_and_out_of_domain():
    raw = np.zeros((50, 32), dtype=np.float32) + 0.25
    codec = ActionCodec.fit(raw, {"actions": {"q01": [-1.0] * 32, "q99": [1.0] * 32}})
    canonical = codec.encode(raw)
    q = np.arange(6) * 0.1
    np.testing.assert_allclose(codec.decode(canonical, q)[:, :6], np.tile(q + 0.250000625, (50, 1)), atol=1e-6)
    assert codec.decode(canonical, q).shape == (50, 7)
    np.testing.assert_array_equal(ActionCodec.from_dict(codec.as_dict()).scale, codec.scale)
    with pytest.raises(ValueError):
        codec.encode(raw * 100)
    with pytest.raises(ValueError):
        codec.decode(canonical * 100, q)
    assert proprio(np.arange(32, dtype=np.float32)).shape == (13,)


def test_partial_td_terminal_timeout_and_bc():
    rewards = torch.tensor([[0.0, 1.0, 99.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    actual = td_target(
        rewards,
        torch.tensor([2, 2, 2]),
        torch.tensor([True, False, False]),
        torch.tensor([False, True, False]),
        torch.tensor([10.0, 10.0, 10.0]),
        0.9,
    )
    torch.testing.assert_close(actual, torch.tensor([0.9, 8.1, 8.1]))
    no_bootstrap = td_target(
        rewards[1:2], torch.tensor([2]), torch.tensor([False]), torch.tensor([True]), torch.tensor([10.0]), 0.9, False
    )
    assert no_bootstrap.item() == 0
    prediction = torch.ones(1, 3, 7)
    prediction[:, 2] = 100
    assert masked_bc(prediction, torch.zeros_like(prediction), torch.tensor([2])).item() == 1


def test_optimizer_ratio_target_ema_and_rng_resume():
    cfg = tiny_config()
    torch.manual_seed(10)
    trainer = RLTTrainer(cfg)
    batch = {
        "z": np.zeros((2, 8), np.float32),
        "proprio": np.zeros((2, 13), np.float32),
        "reference": np.zeros((2, 2, 7), np.float32),
        "action": np.zeros((2, 2, 7), np.float32),
        "next_z": np.zeros((2, 8), np.float32),
        "next_proprio": np.zeros((2, 13), np.float32),
        "next_reference": np.zeros((2, 2, 7), np.float32),
        "lengths": np.asarray([2, 1]),
        "rewards": np.asarray([[0, 1], [0, 0]], np.float32),
        "terminated": np.asarray([True, False]),
        "truncated": np.asarray([False, True]),
    }
    before = [p.clone() for p in trainer.target.parameters()]
    m1 = trainer.update(batch)
    for old, target, current in zip(before, trainer.target.parameters(), trainer.critic.parameters(), strict=True):
        torch.testing.assert_close(target, old.lerp(current, cfg["algorithm"]["tau"]))
    m2, m3 = trainer.update(batch), trainer.update(batch)
    assert [m["actor_updated"] for m in (m1, m2, m3)] == [True, False, True]
    assert trainer.actor_updates == 2 and trainer.critic_updates == 3
    assert isinstance(trainer.actor_optimizer, torch.optim.Adam)
    saved, rng = copy.deepcopy(trainer.state_dict()), rng_state(np.random.default_rng(1))
    expected = trainer.update(batch)
    other = RLTTrainer(cfg)
    other.load_state_dict(saved)
    restore_rng(rng, np.random.default_rng())
    actual = other.update(batch)
    assert actual == expected
    for left, right in zip(trainer.actor.parameters(), other.actor.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_token_mask_training_and_checkpoint_restore(run_dir):
    cache = build_cache(run_dir, policy=FakeFeatures())
    assert cache["full_dataset"] and cache["decode_parity_max"] < 1e-6
    with pytest.raises(ValueError):
        build_cache(run_dir, policy=FakeFeatures())
    first = train_token(run_dir, steps=1, device="cpu")
    second = train_token(run_dir, steps=1, device="cpu")
    assert first != second
    _, run, state = load_run(run_dir)
    saved = load(state["latest_token"], run, load_cache(run_dir), kind="token")
    assert saved["metadata"]["step"] == 2
    model = RLTTokenTransformer(**run["config"]["token"])
    model.load_state_dict(saved["model"])
    assert read_json(run_dir / "token_diagnostics.json")["diagnostic_samples"] == 4
    decide(run_dir, "bc", second, "CPU test, not a convergence claim", run_dir / "token_diagnostics.json")
    with pytest.raises(ValueError):
        train_token(run_dir, steps=1, device="cpu")


class FakeEnvironment:
    def __init__(self):
        self.resets = self.starts = self.steps = self.closes = 0
        self.last_trace = {}

    def reset(self):
        self.resets += 1

    def start(self):
        self.starts += 1
        return obs()

    def step(self, response, *, cancelled):
        assert not cancelled()
        self.steps += 1
        return obs(0.01), 1, True, {"requested_q": np.zeros((2, 6))}

    def close(self):
        self.closes += 1


def prepare_heads(run_dir):
    build_cache(run_dir, policy=FakeFeatures())
    token = train_token(run_dir, steps=1, device="cpu")
    decide(run_dir, "bc", token, "test token accepted", run_dir / "token_diagnostics.json")
    head = train_heads(run_dir, mode="bc", steps=2, device="cpu")
    decide(run_dir, "reference", head, "test BC accepted", run_dir / "heads_diagnostics.json")


def fake_policy(directory, spec):
    _, run, _ = load_run(directory)
    cache = load_cache(directory)
    encoder = load_encoder(directory, run, cache, spec["token"], "cpu")
    trainer = RLTTrainer(run["config"])
    saved = load(spec["heads"], run, cache, kind="heads", token_sha256=spec["token"]["sha256"])
    trainer.load_state_dict(saved["learner"])
    policy = RLTPolicy(run, cache, spec, FakeFeatures(), encoder, trainer.actor, device="cpu")
    policy.metadata = {
        "run_id": run["run_id"],
        "round_id": spec["round_id"],
        "token_sha256": spec["token"]["sha256"],
        "head_sha256": spec["heads"]["sha256"],
        "kind": spec["kind"],
        "spec_sha256": digest(spec),
    }
    return policy


def test_complete_staged_fake_pipeline(run_dir):
    prepare_heads(run_dir)
    for kind in ("reference", "actor_probe", "online"):
        rd, spec = begin_round(run_dir, kind)
        policy = fake_policy(run_dir, spec)
        environment = FakeEnvironment()
        replies = iter(["r", "", "s", "r", "", "f", "r", "", "s", "r", "", "a"])
        summary = collect_round(
            run_dir,
            rd,
            spec,
            policy,
            environment,
            ask=lambda _: next(replies),
            keys_context=lambda: nullcontext(SimpleNamespace(poll=lambda: None)),
        )
        assert summary["attempts"] == 4 and summary["success"] == 2 and summary["aborted"] == 1
        assert environment.starts == environment.resets == environment.steps == 4
        with pytest.raises(ValueError, match="review"):
            begin_round(run_dir, kind)
        replay, identity = load_replay(run_dir)
        assert len(replay["z"]) == 3 * (1 + ("reference", "actor_probe", "online").index(kind))
        assert replay["rewards"][0, 0] == 1 and replay["lengths"][0] == 1
        assert replay["action"].shape[1:] == (2, 7)  # Full request retained despite early release.
        assert identity["episodes"] == len(replay["z"])
        if kind != "actor_probe":
            head = train_heads(run_dir, mode="warmup" if kind == "reference" else "online", steps=2, device="cpu")
        else:
            head = Path(spec["heads"]["path"])
        next_stage = {"reference": "actor_probe", "actor_probe": "online", "online": "complete"}[kind]
        decide(run_dir, next_stage, head, "fake pipeline gate; no real performance claim", rd / "summary.json")
    _, _, state = load_run(run_dir)
    assert state["stage"] == "complete"
    assert len((run_dir / "decisions.jsonl").read_text().splitlines()) == 5
    with pytest.raises(FileExistsError):
        label_episode(rd / "episode_01", "failure", reason="must not overwrite a committed result")


def test_pause_resume_counts_and_abort_exclusion(run_dir):
    prepare_heads(run_dir)
    rd, spec = begin_round(run_dir, "reference")
    environment = FakeEnvironment()
    replies = iter(["r", "", "s", "q"])
    collect_round(
        run_dir,
        rd,
        spec,
        fake_policy(run_dir, spec),
        environment,
        ask=lambda _: next(replies),
        keys_context=lambda: nullcontext(SimpleNamespace(poll=lambda: None)),
    )
    assert summarize(rd)["attempts"] == 1
    with pytest.raises(ValueError):
        train_heads(run_dir, mode="warmup", steps=1, device="cpu")
    with pytest.raises(ValueError):
        decide(run_dir, "actor_probe", spec["heads"]["path"], "incomplete", rd / "summary.json")
    rd2, spec2 = begin_round(run_dir, "reference", resume=rd.name)
    assert rd == rd2 and spec == spec2
    replies = iter(["r", "", "s"] * 3)
    collect_round(
        run_dir,
        rd,
        spec,
        fake_policy(run_dir, spec),
        environment,
        ask=lambda _: next(replies),
        keys_context=lambda: nullcontext(SimpleNamespace(poll=lambda: None)),
    )
    assert summarize(rd)["attempts"] == 4 and environment.starts == 4


def test_no_import_time_hardware_or_model_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ur5e_real.rlt_operator; "
            "assert not any(k in sys.modules for k in ('torch', 'jax', 'rtde.rtde', 'pyrealsense2'))",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_run_recipe_and_cache_tampering(run_dir):
    cache = build_cache(run_dir, policy=FakeFeatures())
    path = run_dir / "cache" / cache["files"][0]["file"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        load_cache(run_dir)
    run = read_json(run_dir / "run.json")
    run["config"]["real"]["action_steps"] = 3
    atomic_json(run_dir / "run.json", run)
    with pytest.raises(ValueError, match="modified"):
        load_run(run_dir)


def test_single_device_lease(tmp_path):
    with lease(tmp_path / ".device.lock"):
        with pytest.raises(RuntimeError, match="another process"):
            with lease(tmp_path / ".device.lock"):
                pytest.fail("double owner")


def test_real_environment_joint_trace_and_release_hold(monkeypatch):
    from ur5e_real.adapters.rlinf_rlt.environment import RealEnvironment
    from ur5e_real.control.gripper_policy import GripperPolicy, GripperCommandConfig

    controller = SimpleNamespace(
        set_target_joints=lambda q: None,
        get_commanded_joints=lambda: [0.0] * 6,
        get_latest_state=lambda: SimpleNamespace(actual_q=[0.0] * 6, tcp_pose=[0.0] * 6),
    )
    env = RealEnvironment({"config": configuration(), "contract": {}})
    env.controller, env.motion = controller, SimpleNamespace(policy_hz=10)
    events = []
    env.gripper = GripperPolicy(
        SimpleNamespace(close=lambda: events.append("close"), open=lambda: events.append("open")),
        GripperCommandConfig(stable_count=2, minimum_command_interval_s=0.5, maximum_cycles=1),
    )
    original = env.gripper.step
    times = iter([10.0, 10.1, 11.0, 11.1])
    env.gripper.step = lambda value: original(value, now=next(times))
    monkeypatch.setattr(env, "observe", lambda: obs())
    calls = []

    def stream(controller, targets, _motion, **kwargs):
        calls.append(len(targets))
        for i, q in enumerate(targets):
            controller.set_target_joints(q)
            if "on_waypoint" in kwargs:
                kwargs["on_waypoint"](i, q)
            if kwargs.get("finish_after_waypoint", lambda _i: False)(i):
                return {"executed_waypoints": i + 1}
        return {"executed_waypoints": len(targets)}

    monkeypatch.setattr("ur5e_real.control.joint.stream_joint_chunk", stream)
    actions = np.zeros((50, 14), dtype=np.float32)
    actions[:2, 13] = 1
    _, k, released, trace = env.step({"actions": actions}, cancelled=lambda: False)
    assert k == 4 and released and events == ["close", "open"]
    assert calls == [20, 10]  # K20 request, early release then independent one-second hold.
    assert len(trace["command_t_q"]) == 14
    assert len(trace["waypoint_t_q_tcp_grip"]) == 4
    assert trace["release_hold_s"] == 1


def test_interrupted_label_recovery_never_reexecutes(run_dir):
    prepare_heads(run_dir)
    rd, spec = begin_round(run_dir, "reference")
    environment = FakeEnvironment()
    replies = iter(["r", ""])

    def interrupted(_prompt):
        try:
            return next(replies)
        except StopIteration:
            raise EOFError()

    with pytest.raises(EOFError):
        collect_round(
            run_dir,
            rd,
            spec,
            fake_policy(run_dir, spec),
            environment,
            ask=interrupted,
            keys_context=lambda: nullcontext(SimpleNamespace(poll=lambda: None)),
        )
    assert summarize(rd)["pending"] == 1
    rd, spec = begin_round(run_dir, "reference", resume=rd.name)
    replies = iter(["s", "q"])
    collect_round(
        run_dir,
        rd,
        spec,
        fake_policy(run_dir, spec),
        environment,
        ask=lambda _: next(replies),
        keys_context=lambda: nullcontext(SimpleNamespace(poll=lambda: None)),
    )
    assert summarize(rd)["pending"] == 0 and environment.starts == 1
    assert len(load_replay(run_dir)[0]["z"]) == 1
