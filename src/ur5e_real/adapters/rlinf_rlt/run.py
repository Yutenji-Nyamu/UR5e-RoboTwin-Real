"""Run identity, explicit stage decisions, and immutable four-episode round specs."""

from __future__ import annotations

from pathlib import Path
import uuid

from ..robotwin_pi05.contract import contract_digest, STATE_LAYOUT
from ..robotwin_pi05.dataset import validate_dataset
from ..robotwin_pi05.native import REPOSITORY
from .config import configuration
from .storage import atomic_json, read_json, digest, file_digest, now, identifier, append_json

STAGES = ("cache", "token", "bc", "reference", "actor_probe", "online", "complete")


def parameter_inventory(checkpoint):
    root = Path(checkpoint) / "params"
    return [
        {
            "file": str(p.relative_to(root)),
            "size": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
            "sha256": file_digest(p),
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def validate_base(run):
    """A frozen parameter content inventory is taken once; detect local replacement before use."""
    identity, checkpoint = run["base_identity"], Path(run["checkpoint"])
    current = [
        {"file": str(p.relative_to(checkpoint / "params")), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in sorted((checkpoint / "params").rglob("*"))
        if p.is_file()
    ]
    expected = [{k: v for k, v in item.items() if k != "sha256"} for item in identity["params"]]
    if (
        current != expected
        or file_digest(checkpoint / "assets" / STATE_LAYOUT / "norm_stats.json") != identity["norm_sha256"]
    ):
        raise ValueError("frozen base parameters or normalization were replaced; create a new versioned run")


def resolve_run(value):
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    return REPOSITORY / "logs" / "rlt" / identifier(str(value))


def initialize(directory, dataset, checkpoint, *, overrides=None, lab_config=None):
    directory, dataset, checkpoint = map(lambda p: Path(p).resolve(), (directory, dataset, checkpoint))
    contract, ready = validate_dataset(dataset)
    from ..robotwin_pi05.contract import read_contract, require_same_contract

    require_same_contract(contract, read_contract(checkpoint / "ur5e_contract.json"))
    norm = checkpoint / "assets" / STATE_LAYOUT / "norm_stats.json"
    if file_digest(norm) != ready["norm_sha256"]:
        raise ValueError("checkpoint and dataset normalization differ")
    if not (checkpoint / "params").is_dir():
        raise ValueError("native JAX checkpoint params are missing")
    identity = {
        "contract_sha256": contract_digest(contract),
        "norm_sha256": ready["norm_sha256"],
        "checkpoint": str(checkpoint),
        "recipe_sha256": file_digest(checkpoint / "ur5e_recipe.json"),
        "verification_sha256": file_digest(checkpoint / "ur5e_verification.json"),
        "params": parameter_inventory(checkpoint),
    }
    manifest = {
        "version": 1,
        "created_at": now(),
        "config": configuration(overrides),
        "base_identity": identity,
        "base_id": digest(identity),
        "dataset": str(dataset),
        "checkpoint": str(checkpoint),
        "contract": contract,
        "lab_config": str(lab_config or REPOSITORY / "configs/lab.yaml"),
    }
    if not identity["params"]:
        raise ValueError("native JAX parameter directory is empty")
    manifest["run_id"] = digest(manifest)
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "run.json", manifest, replace=False)
    atomic_json(
        directory / "state.json",
        {
            "stage": "cache",
            "selected_token": None,
            "selected_head": None,
            "latest_head": None,
            "latest_token": None,
            "round": None,
            "review_pending": False,
        },
        replace=False,
    )
    return directory


def load_run(directory):
    directory = resolve_run(directory)
    manifest = read_json(directory / "run.json")
    expected = manifest["run_id"]
    if digest({k: v for k, v in manifest.items() if k != "run_id"}) != expected:
        raise ValueError("run configuration was modified; use an explicit new versioned run")
    configuration(manifest["config"])
    return directory, manifest, read_json(directory / "state.json")


def update_state(directory, **changes):
    state = read_json(Path(directory) / "state.json")
    state.update(changes)
    atomic_json(Path(directory) / "state.json", state)
    return state


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_digest(path)}


def verify_artifact(reference):
    if not reference or file_digest(reference["path"]) != reference["sha256"]:
        raise ValueError("selected artifact is absent or changed")
    return Path(reference["path"])


def decide(directory, stage, checkpoint, reason, evidence):
    directory, run, state = load_run(directory)
    if not reason.strip() or not evidence:
        raise ValueError("a stage/round decision requires a reason and a diagnostic/round evidence file")
    if stage not in STAGES or abs(STAGES.index(stage) - STAGES.index(state["stage"])) > 1:
        raise ValueError("advance one stage at a time; same-stage review or one-stage rollback is allowed")
    if state["stage"] == "cache":
        raise ValueError("complete the feature cache before reviewing training stages")
    evidence = artifact(evidence)
    selected = artifact(checkpoint) if checkpoint else None
    if selected:
        metadata = read_json(str(checkpoint) + ".json")
        if metadata["run_id"] != run["run_id"] or metadata["file_sha256"] != selected["sha256"]:
            raise ValueError("checkpoint does not belong to this run")
        expected_kind = "token" if stage in ("token", "bc") else "heads"
        if metadata["kind"] != expected_kind:
            raise ValueError(f"stage {stage} needs a {expected_kind} checkpoint")
        if expected_kind == "token":
            if state["selected_token"] and state["selected_token"] != selected:
                raise ValueError("token representation is frozen for this run; fork a new run to change it")
            state["selected_token"] = selected
        else:
            if metadata.get("token_sha256") != (state["selected_token"] or {}).get("sha256"):
                raise ValueError("head checkpoint has a different token encoder")
            if metadata.get("counters", {}).get("bc_updates", 0) < 1:
                raise ValueError("head has not been BC initialized")
            if (
                stage in ("actor_probe", "online", "complete")
                and metadata.get("counters", {}).get("critic_updates", 0) < 1
            ):
                raise ValueError("warm up Q/actor before the actor probe")
            state["selected_head"] = selected
            state["latest_head"] = selected  # an explicit rollback also selects its paired learner resume state
    if STAGES.index(stage) >= STAGES.index("bc") and not state["selected_token"]:
        raise ValueError("select an accepted token checkpoint first")
    if STAGES.index(stage) >= STAGES.index("reference") and not state["selected_head"]:
        raise ValueError("select a BC-initialized head checkpoint first")
    if stage in ("actor_probe", "online", "complete"):
        accepted = read_json(state["selected_head"]["path"] + ".json")
        if accepted.get("counters", {}).get("critic_updates", 0) < 1:
            raise ValueError("select a Q/actor warmup checkpoint before actor execution")
        if stage == "complete" and accepted.get("online_start_update") is None:
            raise ValueError("no online A/C update has been accepted")
    if stage in ("actor_probe", "online") and stage != state["stage"]:
        previous = "reference" if stage == "actor_probe" else "actor_probe"
        completed = [read_json(p) for p in (directory / "rounds").glob("*/summary.json")]
        if not any(r["kind"] == previous and r["attempts"] == 4 and r["pending"] == 0 for r in completed):
            raise ValueError(f"complete and label a four-episode {previous} round first")
    if state["round"]:
        summary = read_json(directory / "rounds" / state["round"] / "summary.json")
        if summary["pending"] or summary["attempts"] != 4:
            raise ValueError("finish and label all four attempts before reviewing a round")
    entry = {
        "from_stage": state["stage"],
        "to_stage": stage,
        "round": state["round"],
        "checkpoint": selected,
        "reason": reason,
        "evidence": evidence,
    }
    append_json(directory / "decisions.jsonl", entry)
    with (directory / "DECISIONS.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## {now()} — {state['stage']} → {stage}\n\n{reason}\n\n"
            f"Round: {state['round']}; checkpoint: {selected}; evidence: {evidence}\n"
        )
    state.update(stage=stage, review_pending=False)
    atomic_json(directory / "state.json", state)
    return entry


def begin_round(directory, kind, *, resume=None):
    directory, run, state = load_run(directory)
    if kind not in ("reference", "actor_probe", "online") or state["stage"] != kind:
        raise ValueError("round kind must match the accepted run stage")
    if state["review_pending"] and not resume:
        raise ValueError("review the previous round and record a decision before starting another")
    if resume:
        round_dir = directory / "rounds" / identifier(resume)
        spec = read_json(round_dir / "spec.json")
        if spec["kind"] != kind or spec["run_id"] != run["run_id"]:
            raise ValueError("round identity mismatch")
        if state["round"] != resume:
            raise ValueError("only the active unfinished round can be resumed")
        if state.get("round_spec_sha256") != digest(spec):
            raise ValueError("the frozen round specification changed")
        summary_path = round_dir / "summary.json"
        if summary_path.is_file():
            summary = read_json(summary_path)
            if summary["attempts"] == 4 and summary["pending"] == 0:
                raise ValueError("round already completed; review it before creating the next round")
    else:
        if state["round"]:
            previous = directory / "rounds" / state["round"] / "summary.json"
            if not previous.is_file() or read_json(previous)["attempts"] != 4 or read_json(previous)["pending"]:
                raise ValueError("resume the unfinished round first")
        name = f"round_{len(list((directory / 'rounds').glob('round_*'))):04d}_{uuid.uuid4().hex[:6]}"
        round_dir = directory / "rounds" / name
        spec = {
            "run_id": run["run_id"],
            "round_id": name,
            "kind": kind,
            "episodes": 4,
            "token": state["selected_token"],
            "heads": state["selected_head"],
            "created_at": now(),
            "seed": len(list((directory / "rounds").glob("round_*"))),
        }
        atomic_json(round_dir / "spec.json", spec, replace=False)
        update_state(directory, round=name, round_spec_sha256=digest(spec))
    verify_artifact(spec["token"])
    if spec["heads"]:
        verify_artifact(spec["heads"])
    return round_dir, spec
