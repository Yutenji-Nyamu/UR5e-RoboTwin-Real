import tempfile
import unittest
from pathlib import Path

from ur5e_real.config import load_config


CONFIG = """
robot:
  host: robot.local
  home_tcp_pose: [1, 2, 3, 4, 5, 6]
cameras:
  head_serial: head
  wrist_serial: wrist
collection:
  data_root: ../runtime-data
servoj:
  config_xml: ../robot_programs/control_loop_configuration.xml
  program_script: ../robot_programs/servoj_control_loop.script
"""


class ConfigTest(unittest.TestCase):
    def test_relative_paths_are_anchored_to_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "configs"
            config_dir.mkdir()
            path = config_dir / "lab.yaml"
            path.write_text(CONFIG, encoding="utf-8")
            cfg = load_config(path)
            self.assertEqual(cfg.robot.host, "robot.local")
            self.assertEqual(cfg.robot.home_tcp_pose, (1, 2, 3, 4, 5, 6))
            self.assertEqual(cfg.cameras.warmup_frames, 60)
            self.assertEqual(cfg.collection.data_root, Path(directory) / "runtime-data")
            self.assertEqual(cfg.servoj.config_xml, Path(directory) / "robot_programs/control_loop_configuration.xml")
            self.assertEqual(cfg.servoj.program_script, Path(directory) / "robot_programs/servoj_control_loop.script")

    def test_rejects_same_camera_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lab.yaml"
            path.write_text(CONFIG.replace("wrist_serial: wrist", "wrist_serial: head"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must differ"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
