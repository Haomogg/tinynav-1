import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from tool.frame_quality import measure_sharpness, select_sharp_timestamps


def _mock_video_db(images):
    db = mock.MagicMock()
    db.read.side_effect = lambda timestamp: images[timestamp]
    return db


class FrameQualityTest(unittest.TestCase):
    def test_sharpness_ranks_blurred_images_lower(self):
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(sharp, (9, 9), 4.0)
        self.assertGreater(measure_sharpness(sharp), measure_sharpness(blurred))

    def test_drops_the_blurriest_fraction(self):
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        images = {
            1: sharp,
            2: cv2.GaussianBlur(sharp, (9, 9), 4.0),  # blurriest
            3: cv2.GaussianBlur(sharp, (3, 3), 1.0),
            4: sharp,
        }
        with mock.patch("tool.video_db.VideoDB", return_value=_mock_video_db(images)):
            self.assertEqual(
                select_sharp_timestamps(Path("/x"), [1, 2, 3, 4], drop_blurriest=0.5),
                [1, 4],
            )
            # No thresholds -> input order returned untouched.
            self.assertEqual(
                select_sharp_timestamps(Path("/x"), [1, 2, 3, 4]), [1, 2, 3, 4]
            )

    def test_refuses_to_keep_almost_nothing(self):
        images = {1: np.zeros((32, 32, 3), np.uint8), 2: np.zeros((32, 32, 3), np.uint8)}
        with mock.patch("tool.video_db.VideoDB", return_value=_mock_video_db(images)):
            with self.assertRaisesRegex(ValueError, "would keep only"):
                select_sharp_timestamps(Path("/x"), [1, 2], min_sharpness=1e6)

    def test_rejects_out_of_range_fraction(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            select_sharp_timestamps(Path("/x"), [1, 2], drop_blurriest=1.0)


class PoseOutlierTest(unittest.TestCase):
    def _trajectory(self, count=12):
        """A straight walk, 10 cm per keyframe."""
        poses = {}
        for index in range(count):
            pose = np.eye(4)
            pose[:3, 3] = [0.1 * index, 0.0, 0.0]
            poses[index] = pose
        return poses

    def test_drops_a_single_jumped_keyframe(self):
        from tool.convert_to_nerf_format import drop_pose_outliers

        poses = self._trajectory()
        poses[6][:3, 3] = [4.0, 0.0, 0.0]  # the VO glitch: out and straight back
        kept = drop_pose_outliers(poses, sorted(poses), max_deviation=0.5)
        self.assertEqual(kept, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11])

    def test_keeps_honest_motion(self):
        from tool.convert_to_nerf_format import drop_pose_outliers

        poses = self._trajectory()
        kept = drop_pose_outliers(poses, sorted(poses), max_deviation=0.5)
        self.assertEqual(kept, sorted(poses))

    def test_keeps_the_first_and_last_keyframe_of_a_fast_walk(self):
        """The edges only have neighbours on one side; motion must not read as an outlier."""
        from tool.convert_to_nerf_format import drop_pose_outliers

        poses = self._trajectory(count=12)
        for index in range(12):
            poses[index][:3, 3] = [0.4 * index, 0.0, 0.0]  # 40 cm per keyframe
        kept = drop_pose_outliers(poses, sorted(poses), max_deviation=0.5)
        self.assertEqual(kept, sorted(poses))

    def test_refuses_to_keep_almost_nothing(self):
        from tool.convert_to_nerf_format import drop_pose_outliers

        rng = np.random.default_rng(0)
        poses = self._trajectory()
        for index in poses:
            poses[index][:3, 3] += rng.normal(0, 0.2, 3)
        with self.assertRaisesRegex(ValueError, "would keep only"):
            drop_pose_outliers(poses, sorted(poses), max_deviation=1e-3)


class SeedPointCloudTest(unittest.TestCase):
    def test_seed_cloud_round_trips_a_known_point(self):
        from tool.convert_to_nerf_format import build_seed_point_cloud, write_point_cloud_ply

        with tempfile.TemporaryDirectory() as temp_dir:
            map_dir = Path(temp_dir)
            # Unit focal length, principal point at the pixel we fill in, so the single
            # valid depth sample lands on the optical axis at z = 2 m.
            intrinsics = np.array([[1.0, 0, 1.0], [0, 1.0, 1.0], [0, 0, 1.0]])
            np.save(map_dir / "intrinsics.npy", intrinsics)
            np.save(map_dir / "rgb_camera_intrinsics.npy", intrinsics)
            depth = np.zeros((3, 3), dtype=np.float32)
            depth[1, 1] = 2.0
            import shelve

            with shelve.open(str(map_dir / "depths")) as depths:
                depths["100"] = depth

            image = np.zeros((3, 3, 3), dtype=np.uint8)
            image[1, 1] = (30, 20, 10)  # BGR
            camera_to_world = np.eye(4)
            camera_to_world[:3, 3] = [5.0, 0.0, 0.0]

            with mock.patch(
                "tool.convert_to_nerf_format.VideoDB",
                return_value=_mock_video_db({100: image}),
            ):
                points, colors = build_seed_point_cloud(
                    map_dir,
                    {100: camera_to_world},
                    np.eye(4),
                    pixel_stride=1,
                    min_depth=0.1,
                    max_depth=10.0,
                    max_points=100,
                    seed=0,
                )

            self.assertEqual(len(points), 1)
            # Camera sits at x=5 looking down +z, so the point is 2 m in front of it.
            np.testing.assert_allclose(points[0], [5.0, 0.0, 2.0], atol=1e-6)
            # Colors come out RGB, from the BGR image.
            np.testing.assert_array_equal(colors[0], [10, 20, 30])

            ply_path = map_dir / "sparse_pc.ply"
            write_point_cloud_ply(ply_path, points, colors)

            from plyfile import PlyData

            vertex = PlyData.read(str(ply_path))["vertex"]
            self.assertEqual(len(vertex), 1)
            np.testing.assert_allclose(
                [vertex["x"][0], vertex["y"][0], vertex["z"][0]], [5.0, 0.0, 2.0], atol=1e-6
            )
            self.assertEqual(
                [vertex["red"][0], vertex["green"][0], vertex["blue"][0]], [10, 20, 30]
            )

            # nerfstudio's dataparser reads the seed cloud with open3d, so when open3d
            # is importable check the file against that loader too. It lives in the
            # nerfstudio venv, not necessarily in the one running the tests.
            try:
                import open3d as o3d
            except ImportError:
                self.skipTest("open3d not installed in this environment")
            cloud = o3d.io.read_point_cloud(str(ply_path))
            self.assertEqual(len(cloud.points), 1)
            np.testing.assert_allclose(np.asarray(cloud.points)[0], [5.0, 0.0, 2.0], atol=1e-6)
            np.testing.assert_allclose(
                np.asarray(cloud.colors)[0], np.array([10, 20, 30]) / 255.0, atol=2e-3
            )


if __name__ == "__main__":
    unittest.main()
