"""Tests for the splat pruner.

Every rule in tool/prune_splat.py was validated by looking at screenshots, which is how a
rule that quietly deletes real geometry gets shipped. These build a synthetic scene where
the answer is known -- a dense wall, a needle, a floater in front of the wall, a lone
fragment, a detached blob -- and check that each rule fires on its own target and on
nothing else.
"""

import argparse
import shelve
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tool.prune_splat import build_rules, connected_blob_sizes, score_gaussians


def default_args(**overrides):
    args = argparse.Namespace(
        min_violations=3,
        min_opacity=0.05,
        max_scale=0.5,
        needle_ratio=0.2,
        needle_length=0.05,
        isolation_radius=0.0,
        min_cluster_voxels=0,
        keep_largest_cluster=False,
        max_range=0.0,
        require_support_beyond=0.0,
        min_support=1,
        max_height=None,
        min_view_spread=0.0,
        roi_size=None,
        roi_center=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class ShapeRuleTest(unittest.TestCase):
    """A wall gaussian is a flat disc; the artefact is a needle. Both have a huge max/min ratio."""

    def setUp(self):
        # disc (two long axes, one thin), needle (one long axis), small round, huge blob
        self.scales = np.array(
            [
                [0.05, 0.04, 0.001],  # disc on a wall -- must survive
                [0.30, 0.002, 0.002],  # needle -- must go
                [0.01, 0.01, 0.01],  # ordinary small gaussian
                [0.80, 0.60, 0.50],  # room-sized blob
                [0.02, 0.001, 0.001],  # short needle, invisible in practice
            ]
        )
        self.positions = np.zeros((5, 3))
        self.opacity = np.full(5, 0.9)
        self.zeros = np.zeros(5, dtype=np.int32)

    def _rules(self, **overrides):
        return build_rules(
            self.positions,
            self.opacity,
            self.scales,
            np.full(5, 10, dtype=np.int32),
            self.zeros,
            np.full(5, 5, dtype=np.int32),
            np.zeros(5),
            np.zeros(5),
            default_args(**overrides),
        )

    def test_needle_rule_spares_discs(self):
        needle = self._rules()["needle"]
        self.assertFalse(needle[0], "a flat disc on a wall must not be pruned")
        self.assertTrue(needle[1], "a long needle must be pruned")
        self.assertFalse(needle[2])
        self.assertFalse(needle[4], "a needle shorter than needle_length is not visible")

    def test_oversized_rule(self):
        self.assertTrue(self._rules()["oversized"][3])
        self.assertFalse(self._rules()["oversized"][0])


class IsolationAndClusterTest(unittest.TestCase):
    def test_isolation_only_hits_the_lonely(self):
        rng = np.random.default_rng(0)
        # A real surface is densely packed -- 3.2 cm to the 8th neighbour on the office
        # model -- so model it as a 2 cm lattice with a little jitter rather than random
        # scatter, which leaves Poisson gaps that are genuinely isolated.
        axis = np.arange(0, 1.0, 0.02)
        grid_x, grid_y = np.meshgrid(axis, axis)
        wall = np.column_stack(
            [grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)]
        ) + rng.normal(0, 0.002, (grid_x.size, 3))
        stray = np.array([[5.0, 5.0, 5.0], [5.4, 5.0, 5.0]])
        positions = np.vstack([wall, stray])
        from scipy.spatial import cKDTree

        neighbour = cKDTree(positions).query(positions, k=9)[0][:, 8]
        rules = build_rules(
            positions,
            np.full(len(positions), 0.9),
            np.full((len(positions), 3), 0.01),
            np.full(len(positions), 10, dtype=np.int32),
            np.zeros(len(positions), dtype=np.int32),
            np.full(len(positions), 5, dtype=np.int32),
            np.zeros(len(positions)),
            neighbour,
            default_args(isolation_radius=0.12),
        )
        isolated = rules["isolated"]
        self.assertFalse(isolated[:len(wall)].any(), "no part of a dense surface is isolated")
        self.assertTrue(isolated[len(wall):].all(), "two lonely gaussians are isolated")

    def test_detached_blob_is_a_separate_cluster(self):
        rng = np.random.default_rng(1)
        body = np.column_stack([rng.uniform(0, 1, 400), rng.uniform(0, 1, 400), np.zeros(400)])
        # a dense little cloud, far from the body: internally crowded, so only connectivity
        # can catch it -- exactly the case that defeated the isolation rule on real data.
        cloud = np.column_stack([rng.uniform(3, 3.1, 60), rng.uniform(3, 3.1, 60), np.full(60, 3.0)])
        positions = np.vstack([body, cloud])
        blob_id, sizes, count = connected_blob_sizes(positions)
        self.assertGreaterEqual(count, 2)
        self.assertNotEqual(blob_id[0], blob_id[-1], "cloud and body must be different blobs")
        self.assertGreater(sizes[blob_id[0]], sizes[blob_id[-1]], "the body is the larger blob")


class FreeSpaceCarvingTest(unittest.TestCase):
    """A gaussian in front of the measured surface is in free space; one on it is supported."""

    def test_counts_violations_and_support(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            map_dir = Path(temp_dir)
            width = height = 64
            intrinsics = np.array([[50.0, 0, 32.0], [0, 50.0, 32.0], [0, 0, 1]])
            # One camera at the origin looking down +z, seeing a wall at 3 m everywhere.
            depth_path = map_dir / "depth.png"
            cv2.imwrite(str(depth_path), np.full((height, width), 3000, np.uint16))
            cameras = [
                {"world_to_camera": np.eye(4), "centre": np.zeros(3), "depth": depth_path}
                for _ in range(4)
            ]
            positions = np.array(
                [
                    [0.0, 0.0, 3.0],  # on the wall
                    [0.0, 0.0, 1.0],  # floating 2 m in front of it
                    [0.0, 0.0, 3.1],  # just behind the wall, inside the margin
                ]
            )
            seen, free_space, supported, view_spread = score_gaussians(
                positions, cameras, intrinsics, width, height, margin=0.25, max_depth=6.0
            )
            # All four cameras sit at the same place here, so every observation points the
            # same way and the spread must be zero -- the degenerate case the rule is meant
            # to flag.
            self.assertTrue(np.allclose(view_spread, 0.0, atol=1e-6))
            self.assertTrue((seen == 4).all())
            self.assertEqual(list(free_space), [0, 4, 0], "only the floater is in free space")
            self.assertEqual(list(supported), [4, 0, 4], "the wall and its margin are supported")

            rules = build_rules(
                positions,
                np.full(3, 0.9),
                np.full((3, 3), 0.01),
                seen,
                free_space,
                supported,
                np.full(3, 1.0),
                np.full(3, 0.01),
                default_args(),
            )
            self.assertEqual(list(rules["free-space"]), [False, True, False])


if __name__ == "__main__":
    unittest.main()
