#!/usr/bin/env python3
"""
Convert TinyNav map artifacts into NeRF/Nerfstudio transforms.json format.

Usage:
    python tool/convert_to_nerf_format.py --map-dir tinynav_map
"""

import argparse
import json
import os
import shelve
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from tool.frame_quality import select_sharp_timestamps
from tool.video_db import VideoDB


def build_seed_point_cloud(
    map_dir: Path,
    poses: Dict[int, np.ndarray],
    t_rgb_to_infra1: np.ndarray,
    pixel_stride: int,
    min_depth: float,
    max_depth: float,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project the stereo depth into a colored world-frame point cloud.

    Without this, splatfacto starts from `num_random=50000` gaussians scattered in a
    cube (nerfstudio/models/splatfacto.py: `means = (torch.rand(...) - 0.5) * scale`),
    because a TinyNav transforms.json carries no `ply_file_path`. A normal 3DGS run is
    seeded with a COLMAP sparse cloud; we have something better already sitting in the
    map -- dense stereo depth -- so hand it over.

    Depth lives in the infra1 optical frame (same frame as poses.npy), and color comes
    from the RGB camera, so each point is projected through the RGB extrinsic to be
    tinted.
    """
    import shelve

    infra_intrinsics = np.load(map_dir / "intrinsics.npy", allow_pickle=True).astype(np.float64)
    rgb_intrinsics = np.load(map_dir / "rgb_camera_intrinsics.npy", allow_pickle=True).astype(np.float64)
    t_infra1_to_rgb = np.linalg.inv(t_rgb_to_infra1)
    rng = np.random.default_rng(seed)

    timestamps = sorted(poses)
    budget = max(1, max_points // len(timestamps))
    rgb_db = VideoDB(dir_path=str(map_dir / "rgb_images_db"), mode="read")
    point_chunks, color_chunks = [], []
    try:
        with shelve.open(str(map_dir / "depths"), flag="r") as depths:
            for timestamp in tqdm(timestamps, desc="Seeding point cloud", unit="frame"):
                key = str(int(timestamp))
                if key not in depths:
                    raise KeyError(f"Missing depth for timestamp {timestamp}")
                depth = np.asarray(depths[key])
                rgb_image = rgb_db.read(timestamp)
                if rgb_image is None:
                    raise KeyError(f"Missing RGB image for timestamp {timestamp}")

                rows = np.arange(0, depth.shape[0], pixel_stride)
                cols = np.arange(0, depth.shape[1], pixel_stride)
                u, v = np.meshgrid(cols, rows)
                z = depth[v, u]
                valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
                u, v, z = u[valid], v[valid], z[valid]
                if z.size == 0:
                    continue
                if z.size > budget:
                    picked = rng.choice(z.size, size=budget, replace=False)
                    u, v, z = u[picked], v[picked], z[picked]

                points_infra1 = np.column_stack(
                    (
                        (u - infra_intrinsics[0, 2]) * z / infra_intrinsics[0, 0],
                        (v - infra_intrinsics[1, 2]) * z / infra_intrinsics[1, 1],
                        z,
                    )
                )
                # Tint from the RGB camera: infra1 -> rgb -> pixel.
                points_rgb = points_infra1 @ t_infra1_to_rgb[:3, :3].T + t_infra1_to_rgb[:3, 3]
                in_front = points_rgb[:, 2] > 1e-6
                points_infra1, points_rgb = points_infra1[in_front], points_rgb[in_front]
                if not len(points_rgb):
                    continue
                pixel_u = np.rint(
                    rgb_intrinsics[0, 0] * points_rgb[:, 0] / points_rgb[:, 2] + rgb_intrinsics[0, 2]
                ).astype(np.int64)
                pixel_v = np.rint(
                    rgb_intrinsics[1, 1] * points_rgb[:, 1] / points_rgb[:, 2] + rgb_intrinsics[1, 2]
                ).astype(np.int64)
                height, width = rgb_image.shape[:2]
                inside = (
                    (pixel_u >= 0) & (pixel_u < width) & (pixel_v >= 0) & (pixel_v < height)
                )
                if not np.any(inside):
                    continue
                camera_to_world = np.asarray(poses[timestamp], dtype=np.float64)
                world = (
                    points_infra1[inside] @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
                )
                point_chunks.append(world.astype(np.float32))
                color_chunks.append(rgb_image[pixel_v[inside], pixel_u[inside], ::-1])
    finally:
        rgb_db.close()

    if not point_chunks:
        raise RuntimeError("No stereo depth projected into the RGB camera")
    return np.concatenate(point_chunks), np.concatenate(color_chunks).astype(np.uint8)


def mask_unreliable_depth(
    depth: np.ndarray, absolute_tolerance: float, relative_tolerance: float, min_neighbours: float
) -> np.ndarray:
    """Blank out depth pixels whose neighbourhood disagrees with them.

    Stereo on a textureless white ceiling does not fail silently -- it returns a sparse
    scatter of pixels whose depths disagree with each other by tens of centimetres. Used as
    supervision that scatter is faithfully reproduced as geometry: the depth-supervised run
    turned the ceiling into a cloud of mutually offset fragments, each one dutifully sitting
    where the sensor said, and each one passing every check the pruner can make because the
    sensor really did report it there.

    A real surface is locally smooth, so compare every pixel with the median of its 5x5
    neighbourhood and drop the ones that disagree by more than the sensor's own error at
    that range (z^2/(f*B) is roughly 3% of z here). Also drop pixels whose neighbourhood is
    mostly empty, which is what an isolated speckle looks like.
    """
    depth32 = depth.astype(np.float32)
    valid = np.isfinite(depth32) & (depth32 > 0.1)
    filled = np.where(valid, depth32, 0).astype(np.float32)

    median = cv2.medianBlur(filled, 5)
    deviation = np.abs(filled - median)
    tolerance = np.maximum(absolute_tolerance, relative_tolerance * filled)
    consistent = deviation <= tolerance

    neighbours = cv2.boxFilter(valid.astype(np.float32), -1, (5, 5), normalize=True)
    dense = neighbours >= min_neighbours

    return np.where(valid & consistent & dense, depth32, 0.0)


def multiview_consistency_mask(
    depth: np.ndarray,
    timestamp: int,
    neighbours: List[int],
    depth_store,
    poses: Dict[int, np.ndarray],
    intrinsics: np.ndarray,
    relative_tolerance: float,
    min_agreements: int,
) -> np.ndarray:
    """Keep only depth that other viewpoints independently confirm.

    Local smoothness cannot catch the failure that matters here. A learned stereo network
    facing a textureless white ceiling does not return noise -- it returns a smooth,
    confident, wrong surface, which passes every neighbourhood test and then gets baked into
    the model by the depth loss. What it cannot do is return the *same* wrong surface from a
    different viewpoint, so project this frame's depth into its neighbours and demand that
    their own measurements agree. This is the geometric-consistency filter used in MVS depth
    fusion, and it is the only test that separates a measured surface from a hallucinated
    one.
    """
    height, width = depth.shape
    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0.1))
    if len(rows) == 0:
        return np.zeros_like(depth, dtype=np.float32)
    z = depth[rows, cols]
    points = np.column_stack(
        (
            (cols - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    camera_to_world = np.asarray(poses[timestamp], dtype=np.float64)
    world = points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]

    agreements = np.zeros(len(rows), dtype=np.int32)
    for other in neighbours:
        key = str(int(other))
        if key not in depth_store:
            continue
        other_depth = np.asarray(depth_store[key], dtype=np.float64)
        world_to_other = np.linalg.inv(np.asarray(poses[other], dtype=np.float64))
        local = world @ world_to_other[:3, :3].T + world_to_other[:3, 3]
        in_front = local[:, 2] > 0.1
        u = np.full(len(rows), -1, dtype=np.int64)
        v = np.full(len(rows), -1, dtype=np.int64)
        u[in_front] = np.rint(
            intrinsics[0, 0] * local[in_front, 0] / local[in_front, 2] + intrinsics[0, 2]
        ).astype(np.int64)
        v[in_front] = np.rint(
            intrinsics[1, 1] * local[in_front, 1] / local[in_front, 2] + intrinsics[1, 2]
        ).astype(np.int64)
        inside = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(inside):
            continue
        measured = other_depth[v[inside], u[inside]]
        predicted = local[inside, 2]
        # Agreement means the other camera measured a surface at the distance this frame
        # implies; anything nearer is an occlusion, anything further is a disagreement.
        close = np.isfinite(measured) & (
            np.abs(measured - predicted) <= relative_tolerance * predicted
        )
        index = np.nonzero(inside)[0][close]
        agreements[index] += 1

    keep = agreements >= min_agreements
    filtered = np.zeros_like(depth, dtype=np.float32)
    filtered[rows[keep], cols[keep]] = z[keep]
    return filtered


def write_rgb_aligned_depth_maps(
    map_dir: Path,
    timestamps: List[int],
    t_rgb_to_infra1: np.ndarray,
    image_size: Tuple[int, int],
    output_dir: Path,
    max_depth: float,
    absolute_tolerance: float = 0.02,
    relative_tolerance: float = 0.03,
    min_neighbours: float = 0.6,
    poses: Dict[int, np.ndarray] | None = None,
    multiview_neighbours: int = 6,
    multiview_tolerance: float = 0.03,
    multiview_agreements: int = 2,
) -> Dict[int, str]:
    """Reproject each keyframe's stereo depth into the RGB camera, for depth supervision.

    splatfacto renders depth but never compares it to anything, so nothing stops it from
    explaining a surface with two layers of gaussians at different distances -- which is
    what ghosting is. A depth loss removes that freedom, and we have real metric depth to
    supply it. The catch is that the depth map lives in the infra1 optical frame at
    544x640 while training happens in the RGB camera at 1088x1920, so it has to be
    reprojected rather than merely resized: 5 cm of baseline between the two cameras is
    30 px of disparity at 1.4 m.

    Occlusion is handled with a z-buffer (nearest surface wins). The reprojection lands on
    roughly one in six RGB pixels, so a small min-filter fills the gaps -- deliberately
    without interpolation, because inventing depth between two surfaces at different
    distances is exactly the error this loss exists to punish.
    """
    infra_intrinsics = np.load(map_dir / "intrinsics.npy", allow_pickle=True).astype(np.float64)
    rgb_intrinsics = np.load(map_dir / "rgb_camera_intrinsics.npy", allow_pickle=True).astype(np.float64)
    t_infra1_to_rgb = np.linalg.inv(np.asarray(t_rgb_to_infra1, dtype=np.float64))
    height, width = int(image_size[0]), int(image_size[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[int, str] = {}
    total_valid = kept_valid = 0
    neighbour_lists: Dict[int, List[int]] = {}
    if poses is not None and multiview_agreements > 0:
        centres = np.array([np.asarray(poses[t], dtype=np.float64)[:3, 3] for t in timestamps])
        for index, timestamp in enumerate(timestamps):
            order = np.argsort(np.linalg.norm(centres - centres[index], axis=1))
            # Skip the frame itself and its immediate neighbours: a 2 cm baseline confirms a
            # hallucination as readily as a real surface.
            picked = [
                timestamps[j]
                for j in order
                if np.linalg.norm(centres[j] - centres[index]) > 0.10
            ][:multiview_neighbours]
            neighbour_lists[timestamp] = picked
    with shelve.open(str(map_dir / "depths"), flag="r") as depths:
        for timestamp in tqdm(timestamps, desc="Reprojecting depth to RGB", unit="frame"):
            key = str(int(timestamp))
            if key not in depths:
                raise KeyError(f"Missing depth for timestamp {timestamp}")
            depth = np.asarray(depths[key], dtype=np.float64)
            raw_valid = int(np.count_nonzero(np.isfinite(depth) & (depth > 0.1) & (depth <= max_depth)))
            if absolute_tolerance > 0 or min_neighbours > 0:
                depth = mask_unreliable_depth(
                    depth, absolute_tolerance, relative_tolerance, min_neighbours
                )
            if neighbour_lists:
                depth = multiview_consistency_mask(
                    depth,
                    timestamp,
                    neighbour_lists[timestamp],
                    depths,
                    poses,
                    infra_intrinsics,
                    multiview_tolerance,
                    multiview_agreements,
                )
            rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0.1) & (depth <= max_depth))
            kept_valid += len(rows)
            total_valid += raw_valid
            if len(rows) == 0:
                continue
            z = depth[rows, cols]
            points_infra1 = np.column_stack(
                (
                    (cols - infra_intrinsics[0, 2]) * z / infra_intrinsics[0, 0],
                    (rows - infra_intrinsics[1, 2]) * z / infra_intrinsics[1, 1],
                    z,
                )
            )
            points_rgb = points_infra1 @ t_infra1_to_rgb[:3, :3].T + t_infra1_to_rgb[:3, 3]
            in_front = points_rgb[:, 2] > 0.1
            points_rgb = points_rgb[in_front]
            pixel_u = np.rint(
                rgb_intrinsics[0, 0] * points_rgb[:, 0] / points_rgb[:, 2] + rgb_intrinsics[0, 2]
            ).astype(np.int64)
            pixel_v = np.rint(
                rgb_intrinsics[1, 1] * points_rgb[:, 1] / points_rgb[:, 2] + rgb_intrinsics[1, 2]
            ).astype(np.int64)
            inside = (pixel_u >= 0) & (pixel_u < width) & (pixel_v >= 0) & (pixel_v < height)
            if not np.any(inside):
                continue
            buffer = np.full((height, width), np.inf, dtype=np.float32)
            np.minimum.at(buffer, (pixel_v[inside], pixel_u[inside]), points_rgb[inside, 2])
            # Min-filter over the gaps: cv2.erode on a buffer where invalid is +inf takes
            # the nearest valid neighbour and leaves untouched holes as inf.
            filled = cv2.erode(buffer, np.ones((3, 5), np.uint8))
            buffer = np.where(np.isinf(buffer), filled, buffer)
            # cv2.erode turns +inf into FLT_MAX, which survives an isfinite() check and
            # then overflows on the conversion to millimetres, so bound by max_depth here.
            valid = np.isfinite(buffer) & (buffer > 0.1) & (buffer <= max_depth)
            millimetres = np.zeros(buffer.shape, dtype=np.uint16)
            millimetres[valid] = np.clip(
                buffer[valid].astype(np.float64) * 1000.0, 0, 65535
            ).astype(np.uint16)
            file_path = output_dir / f"depth_{timestamp}.png"
            if not cv2.imwrite(str(file_path), millimetres):
                raise RuntimeError(f"Failed to write {file_path}")
            written[int(timestamp)] = os.path.relpath(file_path, map_dir)
    if total_valid:
        print(
            f"Depth reliability filter kept {100 * kept_valid / total_valid:.1f}% of the "
            f"measured pixels ({total_valid - kept_valid:,} dropped as inconsistent)"
        )
    print(f"Wrote {len(written)} RGB-aligned depth maps to {output_dir} (16-bit mm PNG)")
    return written


def write_point_cloud_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Minimal binary ply with colors, readable by open3d (nerfstudio's ply loader)."""
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    record = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    record["x"], record["y"], record["z"] = points[:, 0], points[:, 1], points[:, 2]
    record["red"], record["green"], record["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(record.tobytes())


def drop_pose_outliers(
    poses: Dict[int, np.ndarray],
    timestamps: list,
    max_deviation: float,
    window: int = 5,
) -> list:
    """Drop keyframes whose translation disagrees with their temporal neighbours.

    Stereo VO occasionally throws a single keyframe far off the trajectory (on
    map_record_table the frame at t=2.3 s sits 4.33 m away from every other pose, with
    5.5 m and 3.9 m jumps on either side of it). Splatting has no notion of an outlier
    pose: it paints that frame's pixels wherever the pose says they belong, so one bad
    keyframe buys a ghost surface plus a whole depth image of junk in the seed cloud.

    The test is local, not global -- a real trajectory travels, so each keyframe is
    compared against what its neighbours predict for it: every neighbour votes with
    `pos_k + (index - k) * velocity`, and the median of those votes is the reference. The
    velocity is the median step over the window, so constant-speed motion predicts itself
    exactly (including at the first and last keyframe, where the neighbours only exist on
    one side), and taking medians twice keeps one or two glitched neighbours from
    dragging the prediction with them.
    """
    positions = np.array([np.asarray(poses[t], dtype=np.float64)[:3, 3] for t in timestamps])
    kept, dropped = [], []
    for index, timestamp in enumerate(timestamps):
        low = max(0, index - window)
        high = min(len(timestamps), index + window + 1)
        neighbour_index = np.array([k for k in range(low, high) if k != index])
        if len(neighbour_index) == 0:
            kept.append(timestamp)
            continue
        steps = np.diff(positions[low:high], axis=0)
        velocity = np.median(steps, axis=0) if len(steps) else np.zeros(3)
        votes = positions[neighbour_index] + (index - neighbour_index)[:, None] * velocity
        deviation = float(np.linalg.norm(positions[index] - np.median(votes, axis=0)))
        (kept if deviation <= max_deviation else dropped).append(timestamp)
    if dropped:
        print(f"Pose outliers dropped (> {max_deviation} m from neighbours): {len(dropped)}")
    if len(kept) < 2:
        raise ValueError(f"Pose outlier filter would keep only {len(kept)} frames")
    return kept


def convert_nerf_format(
    output_dir: Path,
    poses: Dict[int, np.ndarray],
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    t_rgb_to_infra1: np.ndarray,
    ply_file_path: str | None = None,
    depth_paths: Dict[int, str] | None = None,
) -> None:
    camera_model = "PINHOLE"
    fl_x = float(intrinsics[0, 0])
    fl_y = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    h = int(image_size[0])
    w = int(image_size[1])
    frames = []
    opencv_to_opengl_convention = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )

    for timestamp, camera_to_world_pose in poses.items():
        camera_to_world_opengl = (
            camera_to_world_pose @ t_rgb_to_infra1 @ opencv_to_opengl_convention
        )
        frame = {
            "file_path": f"images/image_{timestamp}.png",
            "transform_matrix": camera_to_world_opengl.tolist(),
        }
        if depth_paths is not None:
            # nerfstudio's dataparser asserts that either every frame carries a depth path
            # or none does, so a frame whose depth is missing has to leave with it.
            if timestamp not in depth_paths:
                continue
            frame["depth_file_path"] = depth_paths[timestamp]
        frames.append(frame)

    data = {
        "camera_model": camera_model,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "frames": frames,
    }
    if ply_file_path is not None:
        # nerfstudio's dataparser reads this key and hands the points to splatfacto as
        # seed_points; without it splatfacto random-initializes 50k gaussians.
        data["ply_file_path"] = ply_file_path

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "transforms.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def infer_image_size(images_dir: Path) -> Tuple[int, int]:
    image_candidates = sorted(images_dir.glob("image_*.png"))
    if not image_candidates:
        raise FileNotFoundError(f"No image_*.png found under: {images_dir}")

    image = cv2.imread(str(image_candidates[0]), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_candidates[0]}")
    return image.shape[:2]


def export_rgb_images(
    map_dir: Path,
    images_dir: Path,
    timestamps: list[int],
) -> Tuple[int, int]:
    images_dir.mkdir(parents=True, exist_ok=True)
    image_size = None
    rgb_db_dir = map_dir / "rgb_images_db"
    rgb_video_db = VideoDB(dir_path=str(rgb_db_dir), mode="read")
    for timestamp in tqdm(timestamps, desc="Exporting RGB images", unit="img"):
        rgb_image = rgb_video_db.read(timestamp)
        if rgb_image is None:
            raise KeyError(f"Missing rgb image timestamp {timestamp} in {rgb_db_dir / 'meta.json'}")
        if image_size is None:
            image_size = rgb_image.shape[:2]
        cv2.imwrite(str(images_dir / f"image_{timestamp}.png"), rgb_image)
    rgb_video_db.close()

    if image_size is None:
        raise RuntimeError("No RGB images exported; poses may be empty")
    return image_size


def infer_t_rgb_to_infra1_from_tf_messages(
    tf_messages: Dict[int, Dict[str, np.ndarray]]
) -> np.ndarray:
    latest_tf: Dict[str, np.ndarray] = {}
    for timestamp_ns in sorted(tf_messages.keys()):
        for edge_key, transform in tf_messages[timestamp_ns].items():
            latest_tf[edge_key] = np.asarray(transform, dtype=np.float64)

    # Some bags name the camera frames directly instead of publishing the full
    # camera_link + *_optical_frame tree: looper bags use cam_left/cam_rgb, the office
    # recordings use camera_camera_left/camera_camera_rgb. Same reading in both cases, and
    # build_map_node.tf_callback recognises the same pairs -- keep the two lists together.
    for frame, child in (("cam_left", "cam_rgb"), ("camera_camera_left", "camera_camera_rgb")):
        if f"{frame}->{child}" in latest_tf:
            return latest_tf[f"{frame}->{child}"]

    required_keys = [
        "camera_link->camera_infra1_frame",
        "camera_infra1_frame->camera_infra1_optical_frame",
        "camera_link->camera_color_frame",
        "camera_color_frame->camera_color_optical_frame",
    ]
    missing = [k for k in required_keys if k not in latest_tf]
    if missing:
        raise KeyError(
            "Missing TF edges in tf_messages.npy required to derive T_rgb_to_infra1: "
            + ", ".join(missing)
        )

    t_infra1_to_link = latest_tf["camera_link->camera_infra1_frame"]
    t_infra1_optical_to_infra1 = latest_tf[
        "camera_infra1_frame->camera_infra1_optical_frame"
    ]
    t_rgb_to_link = latest_tf["camera_link->camera_color_frame"]
    t_rgb_optical_to_rgb = latest_tf["camera_color_frame->camera_color_optical_frame"]
    return (
        np.linalg.inv(t_infra1_optical_to_infra1)
        @ np.linalg.inv(t_infra1_to_link)
        @ t_rgb_to_link
        @ t_rgb_optical_to_rgb
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TinyNav map directory to NeRF transforms.json"
    )
    parser.add_argument(
        "--map-dir",
        required=True,
        help="TinyNav map directory containing poses.npy/intrinsics/images",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Override images directory (default: <map-dir>/images)",
    )
    parser.add_argument(
        "--reuse-existing-images",
        action="store_true",
        help="Do not export from rgb_images_db/video.mp4; reuse existing image_*.png files",
    )
    parser.add_argument(
        "--drop-blurriest",
        type=float,
        default=0.0,
        help="drop this fraction of the blurriest frames (0.33 = worst third). Fast "
        "rotation smears frames and splatting weights every frame equally.",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=0.0,
        help="absolute Laplacian-variance cutoff on frame sharpness",
    )
    parser.add_argument(
        "--no-seed-points",
        dest="seed_points",
        action="store_false",
        help="skip the stereo-depth seed point cloud; splatfacto then random-initializes "
        "50k gaussians in a cube instead of starting from real geometry",
    )
    parser.add_argument("--pixel-stride", type=int, default=4, help="depth subsampling for the seed cloud")
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument(
        "--max-depth",
        type=float,
        default=5.0,
        help="seed-cloud depth cutoff. Stereo error grows with the square of depth, so a "
        "30 m cutoff on an indoor scene sprays far-field noise across tens of metres "
        "(map_record_table: a 1.5 m table produced a 38x47x17 m cloud) and those points "
        "become fog gaussians that never get culled.",
    )
    parser.add_argument(
        "--max-pose-deviation",
        type=float,
        default=0.5,
        help="drop keyframes sitting further than this (m) from the median of their "
        "temporal neighbours; 0 disables the filter",
    )
    parser.add_argument("--max-points", type=int, default=1_000_000, help="seed cloud size cap")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--depth-maps",
        action="store_true",
        help="also export RGB-aligned 16-bit depth maps and reference them from "
        "transforms.json, so training can be supervised with the stereo depth",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="cap the number of keyframes, sampled evenly in time after the quality filters. "
        "A slow capture produces near-duplicate neighbours whose marginal value is low, "
        "while COLMAP's exhaustive matching is O(n^2) and the datamanager caches every image "
        "in RAM: 2321 keyframes is 2.7M image pairs and ~14 GB of cached pixels. Even "
        "sampling keeps the coverage that thinning by sharpness would bunch up.",
    )
    parser.add_argument(
        "--depth-tolerance",
        type=float,
        default=0.02,
        help="a depth pixel may differ from its 5x5 median by this much (m) plus "
        "--depth-tolerance-rel of its range before it is treated as unreliable; 0 disables "
        "the filter",
    )
    parser.add_argument("--depth-tolerance-rel", type=float, default=0.03)
    parser.add_argument(
        "--depth-multiview-agreements",
        type=int,
        default=2,
        help="how many other viewpoints must independently measure a surface at the same "
        "place before a depth pixel is used for supervision; 0 disables the check",
    )
    parser.add_argument("--depth-multiview-neighbours", type=int, default=6)
    parser.add_argument("--depth-multiview-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--depth-min-neighbours",
        type=float,
        default=0.6,
        help="fraction of the 5x5 neighbourhood that must also carry depth",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=6.0,
        help="depth cutoff for the supervision maps. Stereo error grows as z^2/(f*B) "
        "(f*B = 30.4 here: 13 cm at 2 m, 53 cm at 4 m, 2.1 m at 8 m), so supervising with "
        "far depth would teach the model the sensor's noise.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_dir = Path(args.map_dir)
    images_dir = Path(args.images_dir) if args.images_dir else map_dir / "images"

    poses = np.load(map_dir / "poses.npy", allow_pickle=True).item()
    intrinsics = np.load(map_dir / "rgb_camera_intrinsics.npy", allow_pickle=True)
    tf_messages_path = map_dir / "tf_messages.npy"
    t_rgb_to_infra1 = None
    if tf_messages_path.exists():
        tf_messages = np.load(tf_messages_path, allow_pickle=True).item()
        try:
            t_rgb_to_infra1 = infer_t_rgb_to_infra1_from_tf_messages(tf_messages)
            print(f"Using TF data from {tf_messages_path} to derive T_rgb_to_infra1")
        except KeyError as error:
            # An unfamiliar frame naming should not stop the conversion: build_map_node
            # already derived the same extrinsic while mapping and saved it next door.
            print(f"Could not derive T_rgb_to_infra1 from {tf_messages_path.name}: {error}")
            print("Falling back to T_rgb_to_infra1.npy")
    if t_rgb_to_infra1 is None:
        t_rgb_to_infra1 = np.load(map_dir / "T_rgb_to_infra1.npy", allow_pickle=True)
        print("Using T_rgb_to_infra1.npy")
    t_rgb_to_infra1 = np.asarray(t_rgb_to_infra1, dtype=object)
    if t_rgb_to_infra1.shape != (4, 4):
        # build_map_node stores None when it does not recognise the bag's TF frame names,
        # and the failure surfaces much later as a linalg error on a 0-d array.
        raise SystemExit(
            f"{map_dir / 'T_rgb_to_infra1.npy'} does not hold a 4x4 extrinsic "
            f"(got shape {t_rgb_to_infra1.shape}). Recover it from the bag with:\n"
            f"  python tool/extract_rgb_extrinsic.py --bag-dir <rosbag dir> --map-dir {map_dir}"
        )
    t_rgb_to_infra1 = t_rgb_to_infra1.astype(np.float64)
    timestamps = sorted(int(k) for k in poses.keys())
    # Before the sharpness pass: a broken pose is worse than a blurry frame, and dropping
    # it first keeps the sharpness quantile from being computed over frames that are
    # leaving anyway.
    if args.max_pose_deviation > 0.0:
        timestamps = drop_pose_outliers(poses, timestamps, args.max_pose_deviation)
    if args.drop_blurriest > 0.0 or args.min_sharpness > 0.0:
        timestamps = select_sharp_timestamps(
            map_dir, timestamps, args.drop_blurriest, args.min_sharpness
        )
    if 0 < args.max_frames < len(timestamps):
        picked = np.linspace(0, len(timestamps) - 1, args.max_frames).round().astype(int)
        timestamps = [timestamps[index] for index in sorted(set(picked.tolist()))]
        print(f"Thinned to {len(timestamps)} keyframes, evenly spaced in time")
    # Rebuild poses from the surviving timestamps: transforms.json is written from this
    # dict, so a frame left in here that was not exported would break training with a
    # missing-file error (and vice versa). Plain int keys match the exported filenames.
    poses = {timestamp: np.asarray(poses[timestamp]) for timestamp in timestamps}

    if args.reuse_existing_images:
        image_size = infer_image_size(images_dir)
    else:
        image_size = export_rgb_images(map_dir, images_dir, timestamps)

    ply_file_path = None
    if args.seed_points:
        points, colors = build_seed_point_cloud(
            map_dir,
            poses,
            np.asarray(t_rgb_to_infra1, dtype=np.float64),
            args.pixel_stride,
            args.min_depth,
            args.max_depth,
            args.max_points,
            args.seed,
        )
        write_point_cloud_ply(map_dir / "sparse_pc.ply", points, colors)
        ply_file_path = "sparse_pc.ply"
        print(f"Seed point cloud: {len(points)} points -> {map_dir / 'sparse_pc.ply'}")

    depth_paths = None
    if args.depth_maps:
        depth_paths = write_rgb_aligned_depth_maps(
            map_dir,
            timestamps,
            t_rgb_to_infra1,
            image_size,
            map_dir / "depths_rgb",
            args.depth_max,
            args.depth_tolerance,
            args.depth_tolerance_rel,
            args.depth_min_neighbours,
            poses,
            args.depth_multiview_neighbours,
            args.depth_multiview_tolerance,
            args.depth_multiview_agreements,
        )

    convert_nerf_format(
        output_dir=map_dir,
        poses=poses,
        intrinsics=intrinsics,
        image_size=image_size,
        t_rgb_to_infra1=t_rgb_to_infra1,
        ply_file_path=ply_file_path,
        depth_paths=depth_paths,
    )
    print(f"Wrote NeRF transforms to {map_dir / 'transforms.json'} ({len(poses)} frames)")


if __name__ == "__main__":
    main()
