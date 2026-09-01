import unittest

from ur5e_real.control.trajectory import interpolate_pose, minimum_jerk


class TrajectoryTest(unittest.TestCase):
    def test_minimum_jerk_endpoints(self):
        self.assertEqual(minimum_jerk(-1), 0.0)
        self.assertEqual(minimum_jerk(0), 0.0)
        self.assertEqual(minimum_jerk(1), 1.0)
        self.assertEqual(minimum_jerk(2), 1.0)
        self.assertAlmostEqual(minimum_jerk(0.5), 0.5)

    def test_pose_interpolation(self):
        start = [0, 0, 0, 0, 0, 0]
        target = [1, 2, 3, 4, 5, 6]
        self.assertEqual(interpolate_pose(start, target, 0, 2), start)
        self.assertEqual(interpolate_pose(start, target, 2, 2), target)
        self.assertEqual(interpolate_pose(start, target, 1, 2), [0.5, 1, 1.5, 2, 2.5, 3])


if __name__ == "__main__":
    unittest.main()
