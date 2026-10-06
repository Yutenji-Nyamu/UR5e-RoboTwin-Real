import hashlib
import json

import pytest

from ur5e_real.vendor import LOCK, REPOSITORY, verify_source


def test_current_source_snapshots_and_pi05_root():
    from ur5e_real.adapters.robotwin_pi05.native import native_root

    for name in ("RoboTwin", "MetisWAM4D"):
        assert verify_source(name)["status"] == "verified"
    assert native_root() == REPOSITORY / ".third_party/RoboTwin/policy/pi05"


@pytest.mark.parametrize("mutation", ["edit", "missing", "extra_module"])
def test_vendor_detects_source_drift_without_git(tmp_path, mutation):
    root = tmp_path / "source"
    root.mkdir()
    file = root / "module.py"
    file.write_text("x = 1\n")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"version": 1, "sources": {"test": {"files": {
        "module.py": {"bytes": file.stat().st_size, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    }}}}))
    assert verify_source("test", root=root, lock=lock)["files"] == 1
    if mutation == "edit":
        file.write_text("x = 2\n")
    elif mutation == "missing":
        file.unlink()
    else:
        (root / "unexpected.py").touch()
    with pytest.raises(RuntimeError, match="differs from vendor lock"):
        verify_source("test", root=root, lock=lock)


def test_vendor_manifest_contains_no_cache_or_model_checkpoints():
    manifest = json.loads(LOCK.read_text())
    for source in manifest["sources"].values():
        for name in source["files"]:
            assert "__pycache__" not in name
            assert not name.endswith((".pyc", ".safetensors", ".ckpt", ".zip"))
