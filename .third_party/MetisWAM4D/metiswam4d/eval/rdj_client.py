"""Official RoboDojo evaluation client (Isaac Sim, the RoboDojo Python 3.11 runtime) with the MetisWAM4D policy hook.

Runs ``src/eval_client/main.py`` in-process after installing:
  * the ``XPolicyLab.policy.MetisWAM4D.deploy`` module (``rdj_deploy.eval_one_episode``);
  * head / wrist camera intrinsic + extrinsic matrices in the observation (the policy renders the robot geometry
    itself; no depth / instance annotators are requested);
  * a video limit (only the first ``METIS_VIDEO_EPISODES`` episodes of a task are encoded);
  * OptiX warm-up before cuRobo captures its CUDA graphs, and the ``METIS_DISABLE_CUROBO_GRAPHS`` fallback;
  * ``os._exit`` on failure (Kit's non-daemon threads otherwise keep a failed process alive forever).

Environment: ``ROBODOJO_ROOT``, ``ROBODOJO_EVAL_ROOT``, ``ROBODOJO_RUN_ID``, ``EVAL_NUM``, ``METIS_VARIANT``.
"""
import builtins
import os
from pathlib import Path
import sys
import types

POLICY_NAME = "MetisWAM4D"


def video_limit() -> int:
    return int(os.environ.get("METIS_VIDEO_EPISODES", "3"))


def current_episode_index(env) -> int:
    return int(env.success_nums) + int(env.fail_nums)


def install_env_wrapper():
    from src.eval_client import eval_env as module
    original = module.create_eval_env
    limit = video_limit()

    def create_eval_env(*args, **kwargs):
        import torch
        app = args[1] if len(args) > 1 else kwargs["app"]
        for _ in range(3):           # lazy OptiX initialisation before cuRobo's CUDA graph capture
            app.update()
        torch.cuda.synchronize()
        if os.environ.get("METIS_DISABLE_CUROBO_GRAPHS") == "1":
            from curobo import runtime
            import curobo._src.runtime as internal_runtime
            runtime.cuda_graphs = internal_runtime.cuda_graphs = False
            print("cuRobo CUDA graphs disabled", flush=True)
        env = original(*args, **kwargs)
        stream, save = env._stream_vision, env.save_video

        def _stream_vision(env_idx, frame):
            if current_episode_index(env) >= limit:
                return
            return stream(env_idx, frame)

        def save_video(env_idx, video_path, tag):
            if current_episode_index(env) >= limit:
                env._abort_video_writers([env_idx])
                return
            return save(env_idx, video_path, tag)

        env._stream_vision, env.save_video = _stream_vision, save_video
        return env

    module.create_eval_env = create_eval_env


def install_hooks(root: Path):
    from utils import load_file
    from metiswam4d.eval import rdj_deploy
    package = types.ModuleType(f"XPolicyLab.policy.{POLICY_NAME}")
    package.__path__ = []
    sys.modules[package.__name__] = package
    sys.modules[package.__name__ + ".deploy"] = rdj_deploy
    original_import = builtins.__import__
    original_load = load_file.load_yaml
    state = {"wrapped": False}

    def import_hook(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if not state["wrapped"] and name == "src.eval_client.eval_env":
            install_env_wrapper()
            state["wrapped"] = True
        return module

    def load_yaml(path, *args, **kwargs):
        value = original_load(path, *args, **kwargs)
        if Path(path).resolve() == root / "env_cfg/arx_x5.yml":
            value["observation"]["vision"].update(intrinsic_matrix=True, extrinsic_matrix=True)
        if Path(path).name == "_task.yml" and os.environ.get("METIS_UNCAP_EVAL_NUM") == "1":
            # the benchmark caps episodes per task (eval_nums 25 / 50); self-play collection uses every layout
            value.setdefault("common", {})["eval_nums"] = 10 ** 6
            for task in value.get("tasks", {}).values():
                task.pop("eval_nums", None)
        return value

    builtins.__import__ = import_hook
    load_file.load_yaml = load_yaml
    from utils import pipeline_utils          # ``from utils.load_file import *`` bound the original name
    pipeline_utils.load_yaml = load_yaml


def run_entry(entry: Path):
    try:
        exec(compile(entry.read_text(), str(entry), "exec"), {"__name__": "__main__", "__file__": str(entry)})
    except SystemExit as request:
        code = request.code if isinstance(request.code, int) else 0
    except BaseException:
        import traceback
        traceback.print_exc()
        code = 1
    else:
        code = 0
    if code:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


def main():
    root = Path(os.environ["ROBODOJO_ROOT"]).resolve()
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "XPolicyLab"), str(root / "src/eval_client")]
    import faulthandler
    faulthandler.enable()
    install_hooks(root)
    sys.argv[0] = str(Path(__file__).resolve())
    run_entry(root / "src/eval_client/main.py")


if __name__ == "__main__":
    main()
