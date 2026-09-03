import unittest

from ur5e_real.control.prepare import pose_error


class PrepareTest(unittest.TestCase):
    def test_pose_error_separates_translation_and_rotation(self):
        translation, rotation = pose_error([0, 0, 0, 0, 0, 0], [0.003, 0.004, 0, 0, 0, 0.1])
        self.assertAlmostEqual(translation, 0.005)
        self.assertAlmostEqual(rotation, 0.1)

    def test_equivalent_rotation_vector_branch_has_zero_error(self):
        _, rotation = pose_error([0, 0, 0, 0, 0, 3.0], [0, 0, 0, 0, 0, 3.0 - 2.0 * 3.141592653589793])
        self.assertAlmostEqual(rotation, 0.0)


if __name__ == "__main__":
    unittest.main()
