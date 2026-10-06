from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

from ...vendor import verify_source
from .contract import ROBOTWIN_COMMIT

REPOSITORY = Path(__file__).resolve().parents[4]
CHECKPOINT_COMPAT = "ur5e_orbax_0111_callback_v1"


def native_root(robotwin_root: Path | None = None) -> Path:
    root = (robotwin_root or REPOSITORY / ".third_party" / "RoboTwin").resolve()
    verified = verify_source("RoboTwin", root=root, prefix="policy/pi05")
    if verified["upstream_commit"] != ROBOTWIN_COMMIT:
        raise RuntimeError("RoboTwin commit differs from the pi05 integration lock")
    return root / "policy" / "pi05"


def add_native_paths(robotwin_root: Path | None = None, *, model: bool = True) -> Path:
    if model and sys.version_info < (3, 11):
        raise RuntimeError("pi05 model/data runtime requires .venv/pi05/bin/python (Python 3.11+)")
    root = native_root(robotwin_root)
    existing = sys.modules.get("openpi")
    if existing is not None and getattr(existing, "__file__", None):
        if not Path(existing.__file__).resolve().is_relative_to(root / "src"):
            raise RuntimeError("another openpi implementation is already imported in this process")
    for path in (root / "src", root / "packages" / "openpi-client" / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if model:
        os.environ.setdefault("OPENPI_DATA_HOME", str(REPOSITORY / ".venv" / "openpi-assets"))
        os.environ.setdefault("HF_HOME", str(REPOSITORY / ".venv" / "huggingface"))
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(REPOSITORY / ".venv" / "jax-cache"))
    return root


def load_native_train(robotwin_root: Path | None = None):
    root = add_native_paths(robotwin_root)
    spec = importlib.util.spec_from_file_location("ur5e_native_pi05_train", root / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
