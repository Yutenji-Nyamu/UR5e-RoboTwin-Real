import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "archive_evidence", Path(__file__).resolve().parents[1] / "scripts/archive_experiment_evidence.py"
)
archive_evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive_evidence)


def fixture_file(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_small_evidence_is_copied_exactly_and_bulk_data_stays_out(tmp_path):
    repo, data, out = [tmp_path / name for name in ("repo", "data", "archive")]
    fixture_file(repo, "logs/run/metrics.jsonl", b'{"step": 1}\n')
    fixture_file(repo, "outputs/review.png", b"png-preview")
    fixture_file(repo, "checkpoints/run/params/_METADATA", b"parameter metadata")
    fixture_file(repo, "checkpoints/run/params/d/model-shard", b"tiny fragment of a large model")
    fixture_file(repo, "configs/lab.yaml", b"robot: {host: 192.168.0.4}\n")
    fixture_file(data, "raw/action/session_1.json", b'{"outcome":"failure"}')
    fixture_file(data, "raw/action/trajectory.csv", b"q\n0\n")
    fixture_file(data, "raw/camera/frame.png", b"part of bulk image dataset")
    fixture_file(data, "pi05/demo/ur5e_adapter/episode.npz", b"small diagnostic array")
    fixture_file(data, "checkpoints/dp/.hydra/config.yaml", b"seed: 0\n")
    fixture_file(repo, "logs/.env", b"do not publish")
    fixture_file(repo, "logs/large.log", b"x" * 129)
    result = archive_evidence.archive(repo, data, out, max_bytes=128)
    assert result["file_count"] == 8
    assert result["excluded_counts"]["over_size_limit"] == 1
    assert not (out / "data_root/raw/camera/frame.png").exists()
    assert not (out / "repository/checkpoints/run/params/d/model-shard").exists()
    assert not (out / "repository/logs/.env").exists()
    for record in result["files"]:
        archived = (out / record["archive"]).read_bytes()
        assert Path(record["source"]).read_bytes() == archived
        assert hashlib.sha256(archived).hexdigest() == record["archived_sha256"]


def test_live_tail_and_repeated_snapshot_preserve_source_and_old_evidence(tmp_path):
    repo, data, out = [tmp_path / name for name in ("repo", "data", "archive")]
    original = b'{"step":1}\n{"step":'
    source = fixture_file(repo, "logs/metrics.jsonl", original)
    fixture_file(repo, "logs/incomplete.json", b'{"still":')
    first = archive_evidence.archive(repo, data, out)
    assert source.read_bytes() == original
    assert (out / "repository/logs/metrics.jsonl").read_bytes() == b'{"step":1}\n'
    assert first["files"][0]["incomplete_tail_bytes_omitted"] == len(b'{"step":')
    second = archive_evidence.archive(repo, data, out)
    assert first["files"] == second["files"]
    source.unlink()
    third = archive_evidence.archive(repo, data, out)
    assert third["retained_from_earlier_snapshots"] == ["repository/logs/metrics.jsonl"]
    assert third["file_count"] == 1
    assert json.loads((out / "MANIFEST.json").read_text())["file_count"] == 1


def test_no_symlink_traversal_or_recursive_output(tmp_path):
    repo, data, out = [tmp_path / name for name in ("repo", "data", "archive")]
    external = fixture_file(tmp_path, "external/result.json", b"{}")
    (repo / "logs").mkdir(parents=True)
    (repo / "logs/linked.json").symlink_to(external)
    (repo / "logs/linked_dir").symlink_to(external.parent, target_is_directory=True)
    assert archive_evidence.archive(repo, data, out)["file_count"] == 0
    with pytest.raises(ValueError, match="inside an evidence source"):
        archive_evidence.archive(repo, data, repo / "logs/archive")


def test_versioned_archive_matches_manifest():
    output = Path(__file__).resolve().parents[1] / "docs/experiments/evidence"
    manifest = json.loads((output / "MANIFEST.json").read_text())
    names = set()
    total = 0
    for record in manifest["files"]:
        relative = Path(record["archive"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert str(relative) not in names
        names.add(str(relative))
        content = (output / relative).read_bytes()
        assert len(content) == record["archived_bytes"]
        assert hashlib.sha256(content).hexdigest() == record["archived_sha256"]
        total += len(content)
    assert manifest["file_count"] == len(names)
    assert manifest["total_archived_bytes"] == total
    actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    assert actual == names | {"MANIFEST.json"}
