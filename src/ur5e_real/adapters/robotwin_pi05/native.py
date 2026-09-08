from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from .contract import ROBOTWIN_COMMIT

REPOSITORY = Path(__file__).resolve().parents[4]
CHECKPOINT_COMPAT = "ur5e_orbax_0111_callback_v1"


def native_root(robotwin_root: Path | None = None) -> Path:
    root = (robotwin_root or REPOSITORY / ".third_party" / "RoboTwin").resolve()
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True, check=True)
    if result.stdout.strip() != ROBOTWIN_COMMIT:
        raise RuntimeError("RoboTwin commit differs from the pi05 integration lock")
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", "policy/pi05"],
        text=True,
        capture_output=True,
        check=True,
    )
    patch_dir = REPOSITORY / "integrations/pi05/patches"
    expected_diff = "".join(
        (patch_dir / name).read_text()
        for name in ("0003-propagate-download-errors.patch", "0002-orbax-callback.patch", "0001-config-import.patch")
    )
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--no-color", "--no-ext-diff", "HEAD", "--", "policy/pi05"],
        text=True,
        capture_output=True,
        check=True,
    )
    expected_status = (
        " M policy/pi05/src/openpi/shared/download.py\n"
        " M policy/pi05/src/openpi/training/checkpoints.py\n M policy/pi05/src/openpi/training/config.py\n"
    )
    if dirty.stdout != expected_status or diff.stdout != expected_diff:
        raise RuntimeError(
            "native pi05 must match the locked runtime-compatibility patches exactly; "
            "preserve other changes and inspect scripts/bootstrap_robotwin.sh"
        )
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
