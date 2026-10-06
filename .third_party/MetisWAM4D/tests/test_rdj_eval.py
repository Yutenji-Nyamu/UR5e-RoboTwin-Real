"""RoboDojo closed-loop evaluation: round scheduling, official resume manifest, aggregation, EEF conversions."""
import json
from pathlib import Path

import numpy as np
import pytest

from metiswam4d.eval import rdj_campaign as C
from metiswam4d.eval.rdj_observation import eef20_from_observation, eef20_to_action_dicts, pose_wxyz_to_arm10


def campaign(tmp_path: Path) -> dict:
    variants = [
        {"name": "stack_bowls", "base_task": "stack_bowls", "dimension": "Generalization", "split": "standard", "episodes": 5, "step_limit": 500},
        {"name": "stack_bowls_random", "base_task": "stack_bowls", "dimension": "Generalization", "split": "random", "episodes": 5, "step_limit": 500},
        {"name": "insert_key", "base_task": "insert_key", "dimension": "Precision", "split": "standalone", "episodes": 10, "step_limit": 900},
    ]
    data = {"variants": variants, "dimensions": {"Generalization": ["stack_bowls"], "Precision": ["insert_key"]}, "robodojo_root": "/tmp",
            "protocol": {"episodes_per_round": 2, "video_episodes": 3, "execute_steps": 32}, "policy_name": "MetisWAM4D", "config_name": "arx_x5", "env_seed": 0,
            "additional_info": "t", "model_file": "m.pt", "source_checkpoint": "/x/step_0000001"}
    C.write_json(tmp_path / "campaign.json", data)
    return data


def write_result(out: Path, variant: str, flags: list[bool]) -> Path:
    save_dir = out / "results/RoboDojo" / variant / "MetisWAM4D/arx_x5/0_t" / variant
    save_dir.mkdir(parents=True)
    details = {str(i): {"layout_id": i, "success": bool(f), "score": 1.0 if f else 0.1} for i, f in enumerate(flags)}
    path = save_dir / "_result.json"
    path.write_text(json.dumps({"success_rate": 0, "eval_time": len(flags), "score": 0, "details": details}))
    return path


def test_round_targets_and_next_round():
    paired = {"episodes": 5}
    assert [C.round_target(paired, r, 2) for r in (1, 2, 3)] == [2, 4, 5]
    assert C.next_round(paired, 0, 2) == 1 and C.next_round(paired, 2, 2) == 2 and C.next_round(paired, 4, 2) == 3
    assert C.next_round(paired, 5, 2) is None
    standalone = {"episodes": 10}
    assert C.next_round(standalone, 3, 2) == 2  # an interrupted round continues at its own target


def test_manifest_from_result(tmp_path):
    data = campaign(tmp_path)
    path = write_result(tmp_path, "insert_key", [True, False, True])
    manifest = C.synthesize_resume_manifest(path, data, "insert_key")
    payload = json.loads(manifest.read_text())
    assert manifest.name == "_resume_insert_key.json" and payload["save_dir"] == str(path.parent)
    assert payload["success_nums"] == 2 and payload["fail_nums"] == 1
    assert payload["completed_layout_ids"] == [0, 1, 2] and abs(payload["total_score"] - 2.1) < 1e-9
    assert list(payload["details"]) == ["0", "1", "2"]
    # an existing manifest (crashed client) is kept
    manifest.write_text("{\"marker\": 1}")
    assert json.loads(C.synthesize_resume_manifest(path, data, "insert_key").read_text()) == {"marker": 1}


def test_claim_order_and_summary(tmp_path):
    data = campaign(tmp_path)
    write_result(tmp_path, "stack_bowls", [True, True])
    write_result(tmp_path, "insert_key", [False, True, True])
    details = {v["name"]: C.read_details(tmp_path, v["name"]) for v in data["variants"]}
    assert C.complete_rounds(data, details) == 0            # stack_bowls_random has nothing yet
    write_result(tmp_path, "stack_bowls_random", [False, False])
    details = {v["name"]: C.read_details(tmp_path, v["name"]) for v in data["variants"]}
    assert C.complete_rounds(data, details) == 1
    cut = {v["name"]: C.round_target(v, 1, 2) for v in data["variants"]}
    agg = C.aggregate(data, details, cut)
    # stack_bowls merged 2/4 = 50 %, insert_key first two episodes 1/2 = 50 %
    assert agg["dimensions"]["Generalization"]["sr"] == 50.0 and agg["dimensions"]["Precision"]["sr"] == 50.0
    assert agg["dimensions"]["Gen-Std"]["sr"] == 100.0 and agg["dimensions"]["Gen-Random"]["sr"] == 0.0
    assert agg["overall"]["sr"] == 50.0 and agg["episodes"] == 6
    result = C.summary(tmp_path)
    assert result["complete_rounds"] == 1 and (tmp_path / "summary.md").exists()
    # next work: the lowest pending round first (both bowls configs are at round 2, insert_key at round 2 too;
    # ties go to the most remaining simulation steps -> insert_key)
    manager = C.Manager.__new__(C.Manager)
    manager.out, manager.campaign, manager.per_round = tmp_path, data, 2
    manager.variants = {v["name"]: v for v in data["variants"]}
    variant, target = manager.claim_next("test")
    assert variant["name"] == "insert_key" and target == 4
    variant, target = manager.claim_next("test")
    assert variant["name"] in ("stack_bowls", "stack_bowls_random") and target == 4
    assert manager.pending_anywhere()


def test_manifest_with_layout_offset_and_policy_variants(tmp_path):
    data = campaign(tmp_path)
    data["protocol"]["layout_offset"] = 10
    variant = {"name": "insert_key@hide", "task": "insert_key", "policy": {"hide_track": True}, "episodes": 6}
    directory = C.save_dir(tmp_path, data, variant)
    assert directory == tmp_path / "results/RoboDojo/insert_key/MetisWAM4D/arx_x5/0_t/insert_key@hide"
    manifest = C.synthesize_resume_manifest(None, data, variant, 10, directory)
    payload = json.loads(manifest.read_text())
    assert payload["abandoned_layout_ids"] == list(range(10)) and payload["completed_layout_ids"] == []
    assert payload["task_name"] == "insert_key" and payload["save_dir"] == str(directory)
    assert C.task_of("insert_key@hide") == "insert_key" and C.task_of(variant) == "insert_key"
    env = C.client_env(0, data, tmp_path, variant, 2)
    assert json.loads(env["METIS_POLICY"]) == {"hide_track": True} and env["ROBODOJO_RUN_ID"] == "insert_key@hide"
    assert C.client_command(1, data, tmp_path, variant)[4] == "insert_key"


def test_manifest_with_excluded_layouts(tmp_path):
    data = campaign(tmp_path)
    variant = {"name": "build_tower@e8", "task": "build_tower", "policy": {"execute_steps": 8}, "episodes": 6,
               "exclude_layouts": [0, 3, 4]}
    directory = C.save_dir(tmp_path, data, variant)
    payload = json.loads(C.synthesize_resume_manifest(None, data, variant, 2, directory).read_text())
    assert payload["abandoned_layout_ids"] == [0, 1, 3, 4] and payload["completed_layout_ids"] == []


def test_blend_chunks():
    from metiswam4d.eval.rdj_policy import blend_chunks
    prev = np.ones((32, 20), np.float32)
    cur = np.zeros((32, 20), np.float32)
    out = blend_chunks(prev, 16, cur, 0.5)
    assert out.shape == (32, 20) and abs(out[0, 0] - 0.5) < 1e-6 and out[15, 0] < 0.04 and out[16:].max() == 0
    assert np.array_equal(blend_chunks(prev, 32, cur, 0.5), cur)


def test_eef_round_trip():
    rng = np.random.default_rng(0)
    from scipy.spatial.transform import Rotation
    obs = {"state": {}}
    for side in ("left", "right"):
        q = Rotation.random(random_state=int(rng.integers(1 << 30))).as_quat()      # xyzw
        obs["state"][f"{side}_ee_pose"] = np.concatenate((rng.normal(size=3), q[[3, 0, 1, 2]])).astype(np.float32)
        obs["state"][f"{side}_ee_joint_state"] = [float(rng.uniform())]
    eef = eef20_from_observation(obs)
    assert eef.shape == (20,)
    back = eef20_to_action_dicts(eef[None])[0]
    for side in ("left", "right"):
        pose, ref = back[f"{side}_ee_pose"], obs["state"][f"{side}_ee_pose"]
        assert np.allclose(pose[:3], ref[:3], atol=1e-5)
        assert min(np.abs(pose[3:] - ref[3:]).max(), np.abs(pose[3:] + ref[3:]).max()) < 1e-5
        assert abs(float(back[f"{side}_ee_joint_state"][0]) - obs["state"][f"{side}_ee_joint_state"][0]) < 1e-6
    again = np.concatenate([pose_wxyz_to_arm10(back[f"{s}_ee_pose"], back[f"{s}_ee_joint_state"][0]) for s in ("left", "right")])
    assert np.allclose(again, eef, atol=1e-5)
def test_policy_labels_share_reference_noise_seed():
    from metiswam4d.eval.rdj_policy import episode_seed
    import hashlib
    ref=int(hashlib.sha256(b"metiswam4d-rdj:42:fold_clothes:3").hexdigest()[:16],16)
    assert episode_seed(42,"fold_clothes",3)==ref
    assert episode_seed(42,"fold_clothes@base",3)==ref
    assert episode_seed(42,"fold_clothes@hide",3)==ref
    assert episode_seed(42,"fold_clothes_random@hide",3)!=ref
