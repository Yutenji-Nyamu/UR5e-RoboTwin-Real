import tempfile
import unittest
from pathlib import Path

from ur5e_real.data.session_manifest import load_manifest, review_session, write_manifest


class SessionManifestTest(unittest.TestCase):
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
