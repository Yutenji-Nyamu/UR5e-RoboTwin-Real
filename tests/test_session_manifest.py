import tempfile
import unittest
from pathlib import Path

from ur5e_real.data.session_manifest import (
    load_manifest,
    recorded_state_label,
    review_session,
    session_summaries,
    write_manifest,
)


class SessionManifestTest(unittest.TestCase):
    def test_state_labels_distinguish_old_new_empty_and_unknown_sessions(self):
        self.assertEqual(recorded_state_label({"schema_version": 2}), "tcp")
        self.assertEqual(recorded_state_label({"schema_version": 1}), "tcp")
        self.assertEqual(recorded_state_label({"schema_version": 3}), "unknown")
        state_recording = {"joint_source": "actual_q", "tcp_source": "actual_TCP_pose"}
        self.assertEqual(recorded_state_label({"schema_version": 3, "state_recording": state_recording}), "joint+tcp")
        self.assertEqual(
            recorded_state_label({"state_recording": state_recording, "counts": {"rtde_samples": 0}}), "none"
        )

    def test_session_listing_includes_recorded_state_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_manifest(root / "session_old.json", {"run_id": "old", "schema_version": 2})
            write_manifest(
                root / "session_new.json",
                {
                    "run_id": "new",
                    "schema_version": 3,
                    "counts": {"rtde_samples": 5},
                    "state_recording": {"joint_source": "actual_q", "tcp_source": "actual_TCP_pose"},
                },
            )
            rows = {row["run_id"]: row for row in session_summaries(root)}
            self.assertEqual(rows["old"]["recorded_state"], "tcp")
            self.assertEqual(rows["new"]["recorded_state"], "joint+tcp")
            self.assertEqual(rows["new"]["schema_version"], 3)

    def test_review_is_appended_and_raw_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session_test.json"
            write_manifest(
                path,
                {
                    "schema_version": 2,
                    "run_id": "test",
                    "task": "pick_block",
                    "outcome": "unreviewed",
                    "reviews": [],
                    "paths": {"rtde": "/data/raw/actions.csv"},
                },
            )
            review_session(path, "success", "clean demonstration")
            manifest = load_manifest(path)
            self.assertEqual(manifest["outcome"], "success")
            self.assertEqual(manifest["reviews"][0]["note"], "clean demonstration")
            self.assertEqual(manifest["paths"]["rtde"], "/data/raw/actions.csv")


if __name__ == "__main__":
    unittest.main()
