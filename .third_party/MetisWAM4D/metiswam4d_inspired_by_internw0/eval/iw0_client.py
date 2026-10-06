"""Official RoboDojo client (Isaac, Python 3.11) with the InternW0-Delta deploy hook.

Same in-process wrapper as ``metiswam4d.eval.rdj_client`` (env wrapper, video limit, cuRobo fallback, ``os._exit`` on
failure); the deploy module is ``iw0_deploy`` under ``XPolicyLab.policy.<POLICY_NAME>``.  ``METIS_RECORD_4D=1``
additionally attaches head-camera depth and instance-id annotators (in memory only; the evaluation configs on disk
are untouched) for the ground-truth Track recording.
"""
import builtins
import os
from pathlib import Path
import sys
import types

from metiswam4d.eval import rdj_client

POLICY_NAME = os.environ.get("METIS_POLICY_NAME", "InternW0_delta")


def install_hooks(root: Path):
    from utils import load_file
    if os.environ.get("METIS_IW0_OFFICIAL_DEPLOY") == "1":   # InternW0's own deploy.py (official server protocol)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "iw0_official_deploy", "/m2v_intern_v3/danglingwei/files/InternW0_delta_src/XPolicyLab/policy/"
                                   "InternW0_delta/deploy.py")
        deploy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deploy)
    else:
        from metiswam4d_inspired_by_internw0.eval import iw0_deploy as deploy
    package = types.ModuleType(f"XPolicyLab.policy.{POLICY_NAME}")
    package.__path__ = []
    sys.modules[package.__name__] = package
    sys.modules[package.__name__ + ".deploy"] = deploy
    original_import = builtins.__import__
    original_load = load_file.load_yaml
    state = {"wrapped": False, "obs_patched": False}
    record_4d = os.environ.get("METIS_RECORD_4D") == "1"

    def import_hook(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if not state["wrapped"] and name == "src.eval_client.eval_env":
            rdj_client.install_env_wrapper()
            state["wrapped"] = True
        if record_4d and not state["obs_patched"]:
            target = sys.modules.get("env.observation_manager.obs_manager")
            if target is not None and hasattr(target, "ObsManager"):
                state["obs_patched"] = True
                target.ObsManager.ANNOTATORS_TO_COLLECT["instance_id_segmentation_fast"] = "instance_id"
                print("[iw0_client] ObsManager collects instance_id_segmentation_fast", flush=True)
        return module

    def load_yaml(path, *args, **kwargs):
        value = original_load(path, *args, **kwargs)
        if Path(path).resolve() == root / "env_cfg/arx_x5.yml":
            value["observation"]["vision"].update(intrinsic_matrix=True, extrinsic_matrix=True)
            if record_4d:
                value["observation"]["vision"].update(depth=True, approximate_depth=False)
        if record_4d and isinstance(value, dict) and "cam_head" in (value.get("annotator") or {}):
            head = value["annotator"]["cam_head"]
            head["depth_capture"] = {"type": "distance_to_image_plane", "device": "cpu"}
            head["instance_capture"] = {"type": "instance_id_segmentation_fast", "device": "cpu"}
            print(f"[iw0_client] head depth + instance-id annotators enabled ({path})", flush=True)
        if Path(path).name == "_task.yml" and os.environ.get("METIS_UNCAP_EVAL_NUM") == "1":
            value.setdefault("common", {})["eval_nums"] = 10 ** 6
            for task in value.get("tasks", {}).values():
                task.pop("eval_nums", None)
        return value

    builtins.__import__ = import_hook
    load_file.load_yaml = load_yaml
    from utils import pipeline_utils
    pipeline_utils.load_yaml = load_yaml


def main():
    root = Path(os.environ["ROBODOJO_ROOT"]).resolve()
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "XPolicyLab"), str(root / "src/eval_client")]
    import faulthandler
    faulthandler.enable()
    install_hooks(root)
    sys.argv[0] = str(Path(__file__).resolve())
    rdj_client.run_entry(root / "src/eval_client/main.py")


if __name__ == "__main__":
    main()
