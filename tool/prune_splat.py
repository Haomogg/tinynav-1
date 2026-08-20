#!/usr/bin/env python3
"""Carve the floaters out of an exported splat, using the map's own depth as the judge.

Two artefacts survive training no matter how the loss is tuned:

  * spray -- gaussians hanging in mid-air. Nothing in the photometric loss forbids them:
    seen from the few cameras that cover them they explain a bit of colour, and from every
    other angle nobody is looking.
  * far-field mush -- surfaces observed from one direction at 8-15 m, where stereo depth is
    already metres wrong and no amount of training can invent the missing views.

Both are visibility problems, and after training we have something the training loop never
used directly: a metric depth map per keyframe. A gaussian sitting in front of the surface
the sensor reported is in free space, and free space is carvable. That is a hard geometric
statement, not a heuristic tuned by eye -- the same principle as TSDF space carving.

Rules, each reported separately so it is obvious what did the work:

  free-space   gaussian sits at least `margin` in front of the measured surface, in at
               least `--min-violations` views, and is never within `margin` of a measured
               surface. This is the one that removes spray.
  unobserved   never lands inside any training camera's image at a plausible depth.
  faint        opacity below `--min-opacity` (the export already drops < 0.005).
  needle       one long axis with two short ones. A gaussian lying on a wall is legitimately
               a flat disc -- two long axes, one thin -- so the usual max/min scale ratio
               cannot tell good from bad: it flags discs and needles alike. The middle axis
               settles it. s2/s1 near 1 is a disc, near 0 is a needle, and a long needle is
               what renders as a silver streak across the room. Trained without scale
               regularization, 44% of this model's gaussians are needles.
  oversized    longest axis beyond `--max-scale`, the blobs that smear across a room.
  far          farther than `--max-range` from every camera position. Off by default: it
               deletes real geometry, and is meant for when a clean near-field model is
               more useful than a complete but smeared one.

    python tool/prune_splat.py --map-dir output/map_office_big_table \\
        --splat output/map_office_big_table/splat_stock.ply
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
from tqdm import tqdm

OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_cameras(map_dir: Path, transforms_name: str):
    with (map_dir / transforms_name).open(encoding="utf-8") as handle:
        data = json.load(handle)
    intrinsics = np.array(
        [[data["fl_x"], 0, data["cx"]], [0, data["fl_y"], data["cy"]], [0, 0, 1]],
        dtype=np.float64,
    )
    cameras = []
    for frame in data["frames"]:
        camera_to_world = np.asarray(frame["transform_matrix"], dtype=np.float64) @ OPENGL_TO_OPENCV
        cameras.append(
            {
                "world_to_camera": np.linalg.inv(camera_to_world),
                "centre": camera_to_world[:3, 3],
                "depth": map_dir / frame["depth_file_path"] if "depth_file_path" in frame else None,
            }
        )
    return cameras, intrinsics, int(data["w"]), int(data["h"])


def score_gaussians(
    positions: np.ndarray,
    cameras,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    margin: float,
    max_depth: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per gaussian: views that see it, views where it is in free space, views on a surface."""
    seen = np.zeros(len(positions), dtype=np.int32)
    free_space = np.zeros(len(positions), dtype=np.int32)
    supported = np.zeros(len(positions), dtype=np.int32)
    # Sum of unit viewing directions. Its length relative to `seen` says how spread out
    # those directions were: a gaussian only ever seen from one direction has all its unit
    # vectors pointing the same way (|sum| == count), and one seen from all around cancels
    # out. That distinguishes a surface that can be trusted from any angle from one that is
    # only defined from the single direction it was grazed from -- which is what renders as
    # a translucent sheet when viewed from anywhere else.
    direction_sum = np.zeros((len(positions), 3), dtype=np.float64)

    for camera in tqdm(cameras, desc="Carving", unit="view"):
        matrix = camera["world_to_camera"]
        in_camera = positions @ matrix[:3, :3].T + matrix[:3, 3]
        z = in_camera[:, 2]
        candidate = (z > 0.2) & (z < max_depth)
        if not np.any(candidate):
            continue
        index = np.nonzero(candidate)[0]
        projected = in_camera[index] @ intrinsics.T
        u = np.rint(projected[:, 0] / projected[:, 2]).astype(np.int64)
        v = np.rint(projected[:, 1] / projected[:, 2]).astype(np.int64)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        index, u, v = index[inside], u[inside], v[inside]
        if len(index) == 0:
            continue
        seen[index] += 1
        offset = positions[index] - camera["centre"]
        direction_sum[index] += offset / np.maximum(
            np.linalg.norm(offset, axis=1, keepdims=True), 1e-9
        )
        if camera["depth"] is None or not camera["depth"].exists():
            continue
        depth_map = cv2.imread(str(camera["depth"]), cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            continue
        measured = depth_map[v, u].astype(np.float64) / 1000.0
        valid = measured > 0.1
        difference = z[index] - measured
        # In front of the surface by more than the margin: the sensor saw straight through
        # where this gaussian claims to be.
        free_space[index[valid & (difference < -margin)]] += 1
        supported[index[valid & (np.abs(difference) <= margin)]] += 1
    # 0 deg = every view from the same direction, 90 deg = views spread over a hemisphere.
    concentration = np.linalg.norm(direction_sum, axis=1) / np.maximum(seen, 1)
    view_spread = np.degrees(np.arccos(np.clip(concentration, -1, 1)))
    return seen, free_space, supported, view_spread


def connected_blob_sizes(positions: np.ndarray, voxel: float = 0.05):
    """Label 5 cm voxel blobs; return each gaussian's blob id and the size of every blob."""
    from scipy import ndimage

    grid_origin = positions.min(axis=0) - voxel
    voxel_index = np.floor((positions - grid_origin) / voxel).astype(int)
    occupied = np.zeros(voxel_index.max(axis=0) + 2, dtype=bool)
    occupied[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]] = True
    labels, count = ndimage.label(occupied, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels[voxel_index[:, 0], voxel_index[:, 1], voxel_index[:, 2]], sizes, count


def build_rules(
    positions: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    seen: np.ndarray,
    free_space: np.ndarray,
    supported: np.ndarray,
    distance: np.ndarray,
    neighbour_distance: np.ndarray,
    args,
    view_spread: np.ndarray | None = None,
):
    """Every prune rule as a boolean mask, kept separate so each can be tested and reported."""
    sorted_scales = np.sort(scales, axis=1)[:, ::-1]
    longest, middle = sorted_scales[:, 0], sorted_scales[:, 1]
    rules = {
        "free-space": (supported == 0) & (free_space >= args.min_violations),
        "unobserved": seen == 0,
        "faint": opacity < args.min_opacity,
        "needle": (middle / np.maximum(longest, 1e-9) < args.needle_ratio)
        & (longest > args.needle_length),
        "oversized": longest > args.max_scale,
    }
    if args.isolation_radius > 0:
        rules["isolated"] = neighbour_distance > args.isolation_radius
    if args.min_cluster_voxels > 0 or args.keep_largest_cluster:
        per_gaussian, sizes, count = connected_blob_sizes(positions)
        if args.keep_largest_cluster:
            rules["off-cluster"] = per_gaussian != int(np.argmax(sizes))
        else:
            rules["off-cluster"] = sizes[per_gaussian] < args.min_cluster_voxels
        print(f"connected blobs: {count:,}, largest {int(sizes.max()):,} voxels")
    if args.max_range > 0:
        rules["far"] = distance > args.max_range
    if args.require_support_beyond > 0:
        # One confirming view is barely evidence: a single grazing observation of a
        # textureless surface confirms about as much as none. Requiring a handful of
        # independent views to agree is what separates reconstructed geometry from a guess.
        rules["unconfirmed"] = (supported < args.min_support) & (distance > args.require_support_beyond)
    if args.min_view_spread > 0 and view_spread is not None:
        rules["narrow-views"] = view_spread < args.min_view_spread
    if args.max_height is not None:
        rules["above-ceiling"] = positions[:, 2] > args.max_height
    if args.roi_size:
        size = np.array([float(v) for v in args.roi_size.split(",")], dtype=np.float64)
        if args.roi_center:
            centre = np.array([float(v) for v in args.roi_center.split(",")], dtype=np.float64)
        else:
            # Densest 0.5 m cell: the model is thickest where the camera lingered.
            cell = 0.5
            index = np.floor((positions - positions.min(axis=0)) / cell).astype(int)
            flat = np.ravel_multi_index(index.T, index.max(axis=0) + 1)
            busiest = np.argmax(np.bincount(flat))
            centre = positions[flat == busiest].mean(axis=0)
        print(f"ROI box {size} m centred on {np.round(centre, 2)}")
        rules["outside-roi"] = np.any(np.abs(positions - centre) > size / 2, axis=1)
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--splat", required=True)
    parser.add_argument("--output", default=None, help="default: <splat>_pruned.ply")
    parser.add_argument("--transforms", default="transforms_ba.json", help="pose file used for training")
    parser.add_argument("--margin", type=float, default=0.25, help="depth agreement band (m)")
    parser.add_argument("--max-depth", type=float, default=6.0, help="trust the depth map only this far")
    parser.add_argument("--min-violations", type=int, default=3, help="free-space views needed to carve")
    parser.add_argument("--min-opacity", type=float, default=0.05)
    parser.add_argument("--max-scale", type=float, default=0.5, help="longest axis (m)")
    parser.add_argument(
        "--needle-ratio",
        type=float,
        default=0.2,
        help="middle/longest axis below this counts as a needle rather than a disc",
    )
    parser.add_argument(
        "--needle-length",
        type=float,
        default=0.05,
        help="needles longer than this (m) are the visible streaks; shorter ones are harmless",
    )
    parser.add_argument("--max-range", type=float, default=0.0, help="0 = keep the far field")
    parser.add_argument(
        "--require-support-beyond",
        type=float,
        default=0.0,
        help="beyond this distance (m) from the nearest camera, keep only gaussians the depth "
        "sensor actually confirms. Near the camera the sensor is reliable and geometry is "
        "well conditioned; further out, anything it cannot confirm is a guess. 0 disables.",
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=1,
        help="how many views must confirm a gaussian's depth, beyond --require-support-beyond",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=None,
        help="drop gaussians above this world z. The ceiling of an office is textureless and "
        "only ever seen edge-on from below, so it reconstructs as fragments no matter how "
        "long you train; for a walkthrough it is usually better absent than smeared.",
    )
    parser.add_argument(
        "--isolation-radius",
        type=float,
        default=0.12,
        help="a gaussian whose k-th nearest neighbour is further than this (m) is floating "
        "debris rather than part of a surface; 0 disables",
    )
    parser.add_argument("--isolation-neighbours", type=int, default=8)
    parser.add_argument(
        "--roi-size",
        default=None,
        help="keep only an axis-aligned box, as W,D,H in metres (e.g. 8,8,3). Unlike every "
        "other rule this makes no claim about which gaussians are wrong -- it declares a "
        "region of interest, which is what a compact deliverable needs. Pair with "
        "--roi-center, or leave the centre out to place the box on the densest part of the "
        "model, which is where the camera spent its time.",
    )
    parser.add_argument("--roi-center", default=None, help="X,Y,Z of the box centre in metres")
    parser.add_argument(
        "--min-view-spread",
        type=float,
        default=0.0,
        help="degrees of angular spread the observing views must cover. A gaussian only ever "
        "seen from one direction is a surface that exists only for that direction, and it "
        "renders as a translucent sheet from anywhere else -- which is what novel-view "
        "'debris' actually is. 0 disables.",
    )
    parser.add_argument(
        "--min-cluster-voxels",
        type=int,
        default=0,
        help="drop gaussians belonging to a connected blob smaller than this many 5 cm "
        "voxels. A real scene is one connected body -- floor to desk to wall -- while a "
        "floating cloud of fragments is a separate blob, however dense it is inside. This "
        "catches what the isolation rule cannot: debris that keeps itself company. 0 "
        "disables; --keep-largest-cluster is the strictest form of the same idea.",
    )
    parser.add_argument("--keep-largest-cluster", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    map_dir, splat_path = Path(args.map_dir), Path(args.splat)
    ply = PlyData.read(str(splat_path))
    vertex = ply["vertex"]
    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"], dtype=np.float64)))
    scales = np.exp(
        np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(np.float64)
    )
    print(f"{splat_path.name}: {len(positions):,} gaussians")

    cameras, intrinsics, width, height = load_cameras(map_dir, args.transforms)
    with_depth = sum(1 for camera in cameras if camera["depth"] is not None)
    print(f"{len(cameras)} cameras, {with_depth} with a depth map")
    seen, free_space, supported, view_spread = score_gaussians(
        positions, cameras, intrinsics, width, height, args.margin, args.max_depth
    )

    centres = np.array([camera["centre"] for camera in cameras])
    distance = cKDTree(centres).query(positions)[0]

    # Isolation: a gaussian on a real surface is crowded by its neighbours (8th nearest is
    # 3.2 cm away on this model), while a fragment hanging in the air has nothing around it.
    # This needs no depth at all, which is what makes it work in the directions the sensor
    # cannot measure -- the ceiling, and anything specular or dark.
    neighbour_distance = cKDTree(positions).query(
        positions, k=args.isolation_neighbours + 1, workers=-1
    )[0][:, args.isolation_neighbours]

    rules = build_rules(
        positions, opacity, scales, seen, free_space, supported, distance, neighbour_distance,
        args, view_spread,
    )
    print(
        f"view spread (deg): p10 {np.percentile(view_spread, 10):.1f} "
        f"p50 {np.percentile(view_spread, 50):.1f} p90 {np.percentile(view_spread, 90):.1f}"
    )
    print(
        f"\ncamera height range: {centres[:, 2].min():.2f} .. {centres[:, 2].max():.2f} m "
        f"(--max-height is an absolute world z)"
    )

    doomed = np.zeros(len(positions), dtype=bool)
    print("\nrule            removed   (share)")
    for name, mask in rules.items():
        print(f"  {name:<13} {int(mask.sum()):>8,}   {100 * mask.mean():>5.2f}%")
        doomed |= mask
    keep = ~doomed
    print(f"  {'TOTAL':<13} {int(doomed.sum()):>8,}   {100 * doomed.mean():>5.2f}%  ->  {int(keep.sum()):,} kept")
    print(
        f"\nsupport: {100 * (supported > 0).mean():.1f}% of gaussians sit on a measured surface, "
        f"{100 * (seen == 0).mean():.1f}% are never seen"
    )

    if args.dry_run:
        return
    output = Path(args.output) if args.output else splat_path.with_name(splat_path.stem + "_pruned.ply")
    PlyData([PlyElement.describe(vertex.data[keep], "vertex")], text=False).write(str(output))
    print(f"\nWrote {output} ({output.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
