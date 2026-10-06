"""Code snapshot next to the outputs: every (re)start zips the project sources so a run can be reproduced
from its own directory even after the working tree moves on.

``<output_dir>/code/<UTC time>_step<start>.zip`` contains ``metiswam4d/ configs/ scripts/ tests/`` and the top-level
``*.py / *.md / *.yaml / *.toml`` files (no ``__pycache__``, no ``third_party``), plus ``GIT_STATE.txt`` with the
commit, ``git status --short`` and the full diff against HEAD (untracked files are in the zip itself).
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import time
import zipfile

SNAPSHOT_DIRS = ("metiswam4d", "configs", "scripts", "tests")
SNAPSHOT_TOP_SUFFIXES = (".py", ".md", ".yaml", ".toml", ".txt")
EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", "third_party"}


def _git(project: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=project, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def snapshot_code(project: str | Path, output_dir: str | Path, *, step: int = 0, extra: dict | None = None) -> Path:
    project, output_dir = Path(project), Path(output_dir)
    folder = output_dir / "code"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_step{step}.zip"
    git_state = "\n".join([
        "commit: " + _git(project, "rev-parse", "HEAD").strip(),
        "branch: " + _git(project, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "", "# git status --short", _git(project, "status", "--short"),
        "# git diff HEAD", _git(project, "diff", "HEAD"),
    ])
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in SNAPSHOT_DIRS:
            root = project / name
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and not (EXCLUDE_DIR_NAMES & set(path.relative_to(project).parts)):
                    zf.write(path, path.relative_to(project).as_posix())
        for path in sorted(project.iterdir()):
            if path.is_file() and path.suffix in SNAPSHOT_TOP_SUFFIXES:
                zf.write(path, path.name)
        zf.writestr("GIT_STATE.txt", git_state)
        if extra:
            zf.writestr("RUN_INFO.txt", "\n".join(f"{k}: {v}" for k, v in extra.items()))
    return target


__all__ = ["snapshot_code"]
