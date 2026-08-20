#!/usr/bin/env python3
"""Recover T_rgb_to_infra1 from a rosbag's /tf_static and write it into a map.

build_map_node derives this extrinsic from /tf while mapping, but it only recognises
two frame-naming conventions, and the office recordings use a third
(`camera_camera_left -> camera_camera_rgb`). Maps built before that case was handled
carry a `T_rgb_to_infra1.npy` holding `None`, and every tool that needs to put RGB
pixels in the infra1 frame -- the nerf and depthsplat converters, the BA refiner --
dies on it with "0-dimensional array given".

This reads the bag directly (rosbag2 is sqlite3 plus CDR payloads) so an existing map
can be repaired without rebuilding it.

    python tool/extract_rgb_extrinsic.py --bag-dir rosbags/office_umbrella \
        --map-dir output/map_office_umbrella
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
from tf2_msgs.msg import TFMessage

# Every pair is read as child -> frame, matching build_map_node's looper branch.
RGB_TO_LEFT_FRAMES = [
    ("camera_camera_left", "camera_camera_rgb"),
    ("cam_left", "cam_rgb"),
]


def read_static_transforms(bag_dir: Path) -> Dict[str, np.ndarray]:
    """Every /tf_static transform in the bag, keyed "frame->child"."""
    db_files = sorted(bag_dir.glob("*.db3"))
    if not db_files:
        raise FileNotFoundError(f"No .db3 file under {bag_dir}")
    transforms: Dict[str, np.ndarray] = {}
    for db_file in db_files:
        connection = sqlite3.connect(str(db_file))
        try:
            topic_ids = [
                row[0]
                for row in connection.execute("select id from topics where name='/tf_static'")
            ]
            for topic_id in topic_ids:
                for (payload,) in connection.execute(
                    f"select data from messages where topic_id={topic_id}"
                ):
                    for transform in deserialize_message(bytes(payload), TFMessage).transforms:
                        matrix = np.eye(4)
                        rotation = transform.transform.rotation
                        matrix[:3, :3] = Rotation.from_quat(
                            [rotation.x, rotation.y, rotation.z, rotation.w]
                        ).as_matrix()
                        translation = transform.transform.translation
                        matrix[:3, 3] = [translation.x, translation.y, translation.z]
                        key = f"{transform.header.frame_id}->{transform.child_frame_id}"
                        transforms[key] = matrix
        finally:
            connection.close()
    return transforms


def pick_rgb_extrinsic(transforms: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    for frame, child in RGB_TO_LEFT_FRAMES:
        matrix = transforms.get(f"{frame}->{child}")
        if matrix is not None:
            return matrix
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-dir", required=True, help="rosbag2 directory holding the .db3")
    parser.add_argument("--map-dir", required=True, help="map directory to repair")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing 4x4 extrinsic instead of refusing",
    )
    args = parser.parse_args()

    transforms = read_static_transforms(Path(args.bag_dir))
    print(f"/tf_static transforms found: {len(transforms)}")
    for key, matrix in sorted(transforms.items()):
        print(f"  {key}  t={np.round(matrix[:3, 3], 4)}")

    extrinsic = pick_rgb_extrinsic(transforms)
    if extrinsic is None:
        raise SystemExit(
            "No rgb-to-left transform in this bag. Known pairs: "
            + ", ".join(f"{a}->{b}" for a, b in RGB_TO_LEFT_FRAMES)
        )

    target = Path(args.map_dir) / "T_rgb_to_infra1.npy"
    if target.exists():
        existing = np.load(target, allow_pickle=True)
        if existing.shape == (4, 4) and not args.force:
            raise SystemExit(
                f"{target} already holds a 4x4 extrinsic; pass --force to overwrite"
            )
        backup = target.with_suffix(".npy.bak")
        if not backup.exists():
            np.save(backup, existing, allow_pickle=True)
            print(f"Backed up the old value to {backup}")

    np.save(target, extrinsic, allow_pickle=True)
    print(f"\nWrote {target}:\n{np.round(extrinsic, 4)}")
    print(f"translation norm: {np.linalg.norm(extrinsic[:3, 3]) * 1000:.1f} mm")


if __name__ == "__main__":
    main()
