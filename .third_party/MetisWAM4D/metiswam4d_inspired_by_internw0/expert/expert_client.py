"""Official RoboDojo client (Isaac, Python 3.11) driven by the scripted expert.

``iw0_client``'s hooks (env wrapper, video limit, cuRobo fallback, head depth / instance-id annotators with
``METIS_RECORD_4D=1``, uncapped episode count) with the deploy module ``expert_deploy`` registered under
``XPolicyLab.policy.<METIS_POLICY_NAME>``.  The expert needs no policy server: the env's websocket model client is
replaced by an in-process stand-in.
"""
import os
from pathlib import Path
import sys

os.environ.setdefault("METIS_POLICY_NAME", "ScriptedExpert")

from metiswam4d.eval import rdj_client  # noqa: E402
from metiswam4d_inspired_by_internw0.eval import iw0_client  # noqa: E402


class LocalModelClient:
    def __init__(self, **kwargs):
        pass

    def call(self, func_name=None, obs=None, **kwargs):
        return None

    def close(self):
        pass


_install_env_wrapper = rdj_client.install_env_wrapper


def install_env_wrapper():
    from src.eval_client import eval_env
    eval_env.WsModelClient = LocalModelClient
    _install_env_wrapper()


def main():
    root = Path(os.environ["ROBODOJO_ROOT"]).resolve()
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "XPolicyLab"), str(root / "src/eval_client")]
    import faulthandler
    faulthandler.enable()
    rdj_client.install_env_wrapper = install_env_wrapper
    iw0_client.install_hooks(root)
    from metiswam4d_inspired_by_internw0.expert import expert_deploy
    sys.modules[f"XPolicyLab.policy.{iw0_client.POLICY_NAME}.deploy"] = expert_deploy
    sys.argv[0] = str(Path(__file__).resolve())
    rdj_client.run_entry(root / "src/eval_client/main.py")


if __name__ == "__main__":
    main()
