"""Four serial episodes, one frozen policy, no learner updates inside a round."""

from __future__ import annotations

from contextlib import nullcontext
import time

from .replay import EpisodeWriter, label_episode, summarize
from .run import load_run, update_state
from .storage import atomic_json, atomic_npz, read_json, digest


def ask_result(instruction, *, budget, ask=input):
    print(instruction)
    while True:
        value = ask("结果 [s 成功 / f 失败 / t 时间到但未判失败 / a 中止不入RL]：").strip().lower()
        choices = {"s": "success", "f": "failure", "a": "aborted"}
        if budget:
            choices["t"] = "timeout"
        if value in choices:
            return choices[value]
        print("请输入有效标注；t 只适用于执行预算耗尽。")


def collect_round(directory, round_dir, spec, policy, environment, *, ask=input, keys_context=None):
    directory, run, _ = load_run(directory)
    cfg = run["config"]["real"]
    expected = {
        "run_id": run["run_id"],
        "round_id": spec["round_id"],
        "token_sha256": spec["token"]["sha256"],
        "head_sha256": spec["heads"]["sha256"],
        "kind": spec["kind"],
        "spec_sha256": digest(spec),
    }
    if any(policy.metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("round policy does not match its immutable checkpoint specification")
    if keys_context is None:
        from ...collection.terminal import TerminalKeyPoller

        keys_context = TerminalKeyPoller
    # Recover human labels without starting any device or rolling out the episode again.
    for episode in sorted(round_dir.glob("episode_*")):
        if (episode / "result.json").exists():
            continue
        if (episode / "capture_complete.json").exists():
            capture = read_json(episode / "capture_complete.json")
            label = ask_result(cfg["success_instruction"], budget=capture["stop_reason"] == "budget", ask=ask)
            label_episode(episode, label, reason="terminal label recovered after interruption")
        else:
            label_episode(episode, "aborted", reason="interrupted capture; incomplete final state excluded from RL")
    try:
        for slot in range(len(list(round_dir.glob("episode_*"))), 4):
            answer = ask(f"[局 {slot + 1}/4] r 回关节初始位并开爪；q 暂停本轮：").strip().lower()
            if answer != "r":
                break
            environment.reset()
            if ask("请复位场景；Enter 开始本局，q 暂停：").strip().lower() == "q":
                break
            writer = EpisodeWriter(round_dir / f"episode_{slot + 1:02d}", spec, run["config"])
            started = time.monotonic()
            writer.event("start", policy=expected)
            try:
                obs = environment.start()
                response = policy.infer(obs)
                with keys_context() if keys_context else nullcontext() as keys:
                    for chunk in range(cfg["max_chunks"]):

                        def cancelled():
                            return keys is not None and keys.poll() in ("q", "\x03")

                        if cancelled():
                            raise InterruptedError("operator stopped the episode")
                        writer.event("action_requested", chunk=chunk, action=response["rlt"]["action"].tolist())
                        writer.event(
                            "inference",
                            chunk=chunk,
                            policy_call=response["rlt"].get("call"),
                            observation_age_s=response["rlt"].get("observation_age_s"),
                            policy_timing=response.get("policy_timing"),
                            client_elapsed_s=response.get("client_elapsed_s"),
                        )
                        following_obs, executed_k, released, trace = environment.step(response, cancelled=cancelled)
                        finished = released or chunk + 1 == cfg["max_chunks"]
                        if finished:
                            # The final observation is captured before stopping/resetting; labels never move the robot.
                            environment.close()
                        following = policy.infer(following_obs)
                        writer.transition(
                            response["rlt"],
                            following["rlt"],
                            executed_k,
                            observation=obs,
                            next_observation=following_obs,
                            trace=trace,
                        )
                        obs, response = following_obs, following
                        if finished:
                            break
                environment.close()
                stop_reason = "release" if released else "budget"
                rollout_elapsed_s = time.monotonic() - started
                atomic_json(
                    writer.path / "capture_complete.json",
                    {
                        "transitions": len(writer.entries),
                        "stop_reason": stop_reason,
                        "elapsed_s": rollout_elapsed_s,
                    },
                    replace=False,
                )
                label = ask_result(cfg["success_instruction"], budget=not released, ask=ask)
                writer.finish(label, reason=f"terminal result after {stop_reason}", elapsed_s=rollout_elapsed_s)
                writer.event("label_wait", elapsed_s=time.monotonic() - started - rollout_elapsed_s)
            except (EOFError, KeyboardInterrupt):
                environment.close()
                if environment.last_trace:
                    atomic_npz(writer.path / "interrupted_trace.npz", **environment.last_trace)
                writer.event("interrupted", final_capture_complete=(writer.path / "capture_complete.json").exists())
                raise
            except Exception as exc:
                environment.close()
                if environment.last_trace:
                    atomic_npz(writer.path / "interrupted_trace.npz", **environment.last_trace)
                writer.event("aborted", error=f"{type(exc).__name__}: {exc}")
                writer.finish("aborted", reason=f"{type(exc).__name__}: {exc}", elapsed_s=time.monotonic() - started)
                raise
    finally:
        environment.close()
        summary = summarize(round_dir)
        update_state(directory, review_pending=True)
        print(
            f"[ROUND] {spec['round_id']}: attempts={summary['attempts']}/4 success={summary['success']} "
            f"failure={summary['failure']} timeout={summary['timeout']} aborted={summary['aborted']} "
            f"pending={summary['pending']}; details={round_dir / 'summary.json'}",
            flush=True,
        )
    return summary
