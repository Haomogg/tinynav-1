#!/usr/bin/env python3
"""Add the loop closures the mapper's threshold missed, offline, from a saved map.

build_map_node does close loops, but it accepts a candidate only when the DINOv2/VLAD
similarity to a keyframe at least 10 s older exceeds 0.90, and it looks at one candidate
per keyframe. On map_office_big_table that yields 15 candidate pairs across 503 keyframes,
because the similarity to the best older frame is p50 0.798 / p90 0.883 / max 0.919 -- the
threshold sits above almost the whole distribution. So the pose graph is a chain of 502
sequential edges with essentially nothing tying revisits together, and the drift that
accumulates over a 22 m walk has nothing to correct it. That is the misalignment visible
in the splat.

Lowering the retrieval threshold on its own would invite false loops: an open-plan office
is full of near-identical desks. But the mapper already has the right defence -- every
candidate must survive keypoint matching plus PnP with at least 100 inliers -- so the
sensible division of labour is a generous retrieval and a strict geometric gate. That is
what this does, reusing the map's own stored features, the repo's own `estimate_pose`, and
the repo's own Ceres pose-graph solver.

Nothing in the map is modified: poses land in poses_loop.npy and transforms_loop.json.

    python tool/loop_closure_refine.py candidates --map-dir output/map_office_big_table
    python tool/loop_closure_refine.py verify    --map-dir output/map_office_big_table
    python tool/loop_closure_refine.py solve     --map-dir output/map_office_big_table
"""

import argparse
import json
import shelve
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from tinynav.core.math_utils import estimate_pose

# Same convention as convert_to_nerf_format: transforms.json holds OpenGL camera-to-world.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
# build_map_node's solve_pose_graph weights every edge this way, loop and sequential alike.
TRANSLATION_WEIGHT = np.array([10.0, 10.0, 10.0])
ROTATION_WEIGHT = np.array([30.0, 30.0, 30.0])
MIN_TIME_GAP_NS = 10e9


def load_map(map_dir: Path):
    poses = np.load(map_dir / "poses.npy", allow_pickle=True).item()
    intrinsics = np.load(map_dir / "intrinsics.npy", allow_pickle=True).astype(np.float64)
    timestamps = sorted(int(key) for key in poses)
    return poses, intrinsics, timestamps


def load_embeddings(map_dir: Path, timestamps: List[int]) -> np.ndarray:
    with shelve.open(str(map_dir / "embeddings"), flag="r") as database:
        vectors = [np.asarray(database[str(timestamp)], dtype=np.float64).ravel() for timestamp in timestamps]
    matrix = np.array(vectors)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def find_candidates(
    map_dir: Path, timestamps: List[int], threshold: float, top_k: int
) -> List[Tuple[int, int, float]]:
    """(curr, prev, similarity) for every retrieval candidate, newest frame first."""
    embeddings = load_embeddings(map_dir, timestamps)
    stamps = np.array(timestamps, dtype=np.float64)
    similarity = embeddings @ embeddings.T
    older = (stamps[None, :] + MIN_TIME_GAP_NS) < stamps[:, None]
    candidates: List[Tuple[int, int, float]] = []
    for row in range(len(timestamps)):
        allowed = np.nonzero(older[row] & (similarity[row] > threshold))[0]
        if len(allowed) == 0:
            continue
        for column in allowed[np.argsort(similarity[row, allowed])[-top_k:]]:
            candidates.append((timestamps[row], timestamps[column], float(similarity[row, column])))
    return candidates


def match_descriptors(features_prev: dict, features_curr: dict, ratio: float = 0.9):
    """Mutual nearest neighbours with a ratio test, on the stored SuperPoint descriptors.

    The mapper matches with a LightGlue TensorRT engine, which needs engines built for the
    local GPU; plain mutual nearest neighbour matching needs nothing but the descriptors
    that are already in the map. It finds fewer correspondences than LightGlue, which only
    makes the 100-inlier gate below stricter, never more permissive.
    """
    def unpack(features):
        keypoints = np.asarray(features["kpts"])[0]
        descriptors = np.asarray(features["descps"])[0]
        mask = np.asarray(features["mask"])[0].ravel().astype(bool)
        descriptors = descriptors / np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-12)
        return keypoints[mask], descriptors[mask]

    keypoints_prev, descriptors_prev = unpack(features_prev)
    keypoints_curr, descriptors_curr = unpack(features_curr)
    if len(descriptors_prev) < 20 or len(descriptors_curr) < 20:
        return np.empty((0, 2)), np.empty((0, 2))

    similarity = descriptors_prev @ descriptors_curr.T
    best_curr = np.argmax(similarity, axis=1)
    best_prev = np.argmax(similarity, axis=0)
    mutual = best_prev[best_curr] == np.arange(len(best_curr))
    # Ratio test against the runner-up: repeated office furniture produces confident but
    # ambiguous matches, and those are exactly the ones that fabricate a loop.
    partitioned = np.partition(similarity, -2, axis=1)
    good = partitioned[:, -1] > 1e-9
    good &= (partitioned[:, -2] / np.maximum(partitioned[:, -1], 1e-12)) < ratio
    keep = mutual & good
    return keypoints_prev[keep], keypoints_curr[best_curr[keep]]


def verify_candidates(
    map_dir: Path,
    candidates: List[Tuple[int, int, float]],
    intrinsics: np.ndarray,
    min_inliers: int,
) -> List[Tuple[int, int, np.ndarray, int]]:
    """Keep only candidates whose geometry agrees: PnP with >= min_inliers, as the mapper does."""
    verified = []
    with shelve.open(str(map_dir / "features"), flag="r") as features, shelve.open(
        str(map_dir / "depths"), flag="r"
    ) as depths:
        for curr, prev, _similarity in tqdm(candidates, desc="Verifying loops", unit="pair"):
            if str(curr) not in features or str(prev) not in features or str(curr) not in depths:
                continue
            keypoints_prev, keypoints_curr = match_descriptors(features[str(prev)], features[str(curr)])
            if len(keypoints_prev) < min_inliers:
                continue
            depth_curr = np.asarray(depths[str(curr)], dtype=np.float32)
            # Argument order copied from build_map_node.detect_loop_closure: the 3D points
            # come from the current frame's depth, the 2D observations from the previous
            # frame, and the result is used as T_prev_curr. Swapping these silently
            # reverses every loop edge.
            success, transform, _, _, inliers = estimate_pose(
                keypoints_prev, keypoints_curr, depth_curr, intrinsics
            )
            if success and len(inliers) >= min_inliers:
                verified.append((curr, prev, transform, len(inliers)))
    return verified


def build_sequential_edges(poses: Dict[int, np.ndarray], timestamps: List[int]):
    """Odometry edges, rebuilt from the saved poses.

    build_map_node builds these from raw odometry, which the map does not store per
    keyframe. The saved poses are the pose-graph result, but with at most 13 loop edges on
    this map the optimisation barely moved anything, so consecutive relative poses are the
    odometry relative poses to well within their own noise.
    """
    edges = []
    for previous, current in zip(timestamps, timestamps[1:]):
        relative = np.linalg.inv(np.asarray(poses[previous], dtype=np.float64)) @ np.asarray(
            poses[current], dtype=np.float64
        )
        edges.append((current, previous, relative))
    return edges


def solve(map_dir: Path, verified_path: Path, max_iterations: int) -> None:
    from tinynav.tinynav_cpp_bind import pose_graph_solve

    poses, _, timestamps = load_map(map_dir)
    payload = json.loads(verified_path.read_text())
    loop_edges = [
        (int(entry["curr"]), int(entry["prev"]), np.array(entry["transform"], dtype=np.float64))
        for entry in payload["loops"]
    ]
    edges = build_sequential_edges(poses, timestamps) + loop_edges
    print(f"pose graph: {len(timestamps)} nodes, {len(edges) - len(loop_edges)} sequential + {len(loop_edges)} loop edges")

    constraints = [
        (curr, prev, transform, TRANSLATION_WEIGHT, ROTATION_WEIGHT) for curr, prev, transform in edges
    ]
    initial = {timestamp: np.asarray(poses[timestamp], dtype=np.float64) for timestamp in timestamps}
    optimized = pose_graph_solve(initial, constraints, {min(timestamps): True}, max_iterations)

    movement = np.array(
        [
            np.linalg.norm(np.asarray(optimized[t])[:3, 3] - np.asarray(poses[t])[:3, 3])
            for t in timestamps
        ]
    )
    print(
        f"pose change: median {np.median(movement) * 100:.1f} cm, "
        f"p90 {np.percentile(movement, 90) * 100:.1f} cm, max {movement.max() * 100:.1f} cm"
    )

    np.save(map_dir / "poses_loop.npy", {t: np.asarray(optimized[t]) for t in timestamps}, allow_pickle=True)
    write_transforms(map_dir, optimized)


def write_transforms(map_dir: Path, poses: Dict[int, np.ndarray]) -> None:
    """transforms_loop.json, keeping every other field (depth paths, seed cloud) intact."""
    with (map_dir / "transforms.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    t_rgb_to_infra1 = np.asarray(
        np.load(map_dir / "T_rgb_to_infra1.npy", allow_pickle=True), dtype=np.float64
    )
    frames = []
    for frame in data["frames"]:
        timestamp = int(Path(frame["file_path"]).stem.split("_")[-1])
        if timestamp not in poses:
            continue
        camera_to_world = np.asarray(poses[timestamp], dtype=np.float64) @ t_rgb_to_infra1 @ OPENGL_TO_OPENCV
        updated = dict(frame)
        updated["transform_matrix"] = camera_to_world.tolist()
        frames.append(updated)
    data["frames"] = frames
    target = map_dir / "transforms_loop.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    print(f"Wrote {target} ({len(frames)} frames). poses.npy and transforms.json untouched.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("candidates", "verify", "solve", "all"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--map-dir", required=True)
        sub.add_argument("--threshold", type=float, default=0.85, help="retrieval similarity cutoff")
        sub.add_argument("--top-k", type=int, default=3, help="candidates kept per keyframe")
        sub.add_argument("--min-inliers", type=int, default=100, help="PnP inliers a loop must reach")
        sub.add_argument("--max-iterations", type=int, default=1024)
    args = parser.parse_args()

    map_dir = Path(args.map_dir)
    poses, intrinsics, timestamps = load_map(map_dir)
    verified_path = map_dir / "loop_candidates_verified.json"

    if args.stage in ("candidates", "verify", "all"):
        candidates = find_candidates(map_dir, timestamps, args.threshold, args.top_k)
        print(
            f"retrieval: {len(candidates)} candidate pairs from {len(timestamps)} keyframes "
            f"(threshold {args.threshold}, top_k {args.top_k})"
        )
    if args.stage in ("verify", "all"):
        verified = verify_candidates(map_dir, candidates, intrinsics, args.min_inliers)
        print(f"geometric gate: {len(verified)}/{len(candidates)} pairs reached {args.min_inliers} PnP inliers")
        if verified:
            gaps = np.array([(curr - prev) / 1e9 for curr, prev, _, _ in verified])
            counts = np.array([count for *_, count in verified])
            print(
                f"  time gaps: median {np.median(gaps):.1f} s max {gaps.max():.1f} s | "
                f"inliers: median {int(np.median(counts))} max {int(counts.max())}"
            )
        verified_path.write_text(
            json.dumps(
                {
                    "loops": [
                        {"curr": curr, "prev": prev, "transform": transform.tolist(), "inliers": count}
                        for curr, prev, transform, count in verified
                    ]
                }
            )
        )
        print(f"  wrote {verified_path}")
    if args.stage in ("solve", "all"):
        solve(map_dir, verified_path, args.max_iterations)


if __name__ == "__main__":
    main()
