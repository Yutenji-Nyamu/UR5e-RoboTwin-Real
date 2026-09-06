import unittest
from pathlib import Path

from ur5e_real.storage import _device_path


class StorageTest(unittest.TestCase):
    def test_uuid_source_maps_to_stable_device_path(self):
        self.assertEqual(_device_path("UUID=1234-ABCD"), Path("/dev/disk/by-uuid/1234-ABCD"))

    def test_absolute_device_source_is_preserved(self):
        self.assertEqual(_device_path("/dev/sda2"), Path("/dev/sda2"))

    def test_relative_source_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported fstab source"):
            _device_path("sda2")


if __name__ == "__main__":
    unittest.main()
