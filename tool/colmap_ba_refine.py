#!/usr/bin/env python3
"""Refine TinyNav poses with COLMAP's bundle adjuster, for gaussian splatting.

Why this exists: 3DGS assumes poses that are consistent to well under a pixel, which is
what a COLMAP reconstruction delivers. TinyNav poses come from stereo VO, and measured on
these maps they disagree with image geometry by roughly a centimetre and a degree between
neighbouring keyframes -- 10-20 px at fx=768 and 1.5 m. The model can only explain a
surface that lands in two places at once by smearing it, which is the mush we see.

The plan is deliberately *not* "let COLMAP reconstruct the scene". Indoor, low-texture
sequences make incremental SfM fragile, and we already have poses that are locally good.
So COLMAP is used as a refiner:

  1. export our poses + intrinsics as a COLMAP model (this file, `export` stage)
  2. extract and match features
  3. triangulate points with the poses held FIXED -- the mean reprojection error this
     reports is the first objective, standard measurement of how far off our poses are
  4. bundle-adjust with poses free and intrinsics fixed
  5. write a new transforms.json; poses.npy and the original transforms.json are left
     untouched

Stages 2-4 need pycolmap and land in this file once the installed version is known.
Stage 1 and its self-check run standalone:

    python tool/colmap_ba_refine.py export --map-dir output/map_office_umbrella
"""

import argparse
import json
import shelve
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# transforms.json stores camera-to-world in the OpenGL convention (x right, y up, z back)
# because that is what nerfstudio reads; COLMAP wants world-to-camera in the OpenCV
# convention (x right, y down, z forward). The flip is its own inverse.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """COLMAP's images.txt order: QW QX QY QZ."""
    trace = np.trace(rotation)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quaternion = np.array([w, x, y, z])
    return quaternion / np.linalg.norm(quaternion)


def load_transforms(map_dir: Path) -> Tuple[dict, List[dict]]:
    with (map_dir / "transforms.json").open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data, data["frames"]


def world_to_camera(frame: dict) -> np.ndarray:
    """OpenGL camera-to-world (as stored) -> OpenCV world-to-camera (as COLMAP wants)."""
    camera_to_world_opencv = np.asarray(frame["transform_matrix"], dtype=np.float64) @ OPENGL_TO_OPENCV
    return np.linalg.inv(camera_to_world_opencv)


def export_colmap_model(map_dir: Path, model_dir: Path) -> int:
    data, frames = load_transforms(map_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    with (model_dir / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(
            f"1 PINHOLE {data['w']} {data['h']} "
            f"{data['fl_x']} {data['fl_y']} {data['cx']} {data['cy']}\n"
        )

    with (model_dir / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id, frame in enumerate(sorted(frames, key=lambda f: f["file_path"]), start=1):
            matrix = world_to_camera(frame)
            q = rotation_to_quaternion(matrix[:3, :3])
            t = matrix[:3, 3]
            name = Path(frame["file_path"]).name
            handle.write(
                f"{image_id} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {name}\n\n"
            )

    (model_dir / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
        encoding="utf-8",
    )
    return len(frames)


def check_projection_roundtrip(map_dir: Path, sample_frames: int = 6) -> None:
    """Verify the exported world-to-camera matrices against the converter's own maths.

    A wrong axis flip or a rotation applied on the wrong side still writes a perfectly
    well-formed COLMAP model; it only shows up as a bundle adjustment that converges to
    nonsense. So take the same 3D points down two independent paths and require the same
    pixel: the converter's path (infra1 point -> RGB camera via T_rgb_to_infra1 ->
    rgb intrinsics) and this file's path (infra1 point -> world via poses.npy -> the
    exported world-to-camera -> rgb intrinsics). Agreement is a floating-point question,
    not a matter of degree, so the tolerance is sub-pixel and a bug misses it by hundreds.
    """
    depth_shelf = map_dir / "depths"
    poses = np.load(map_dir / "poses.npy", allow_pickle=True).item()
    infra_intrinsics = np.load(map_dir / "intrinsics.npy", allow_pickle=True).astype(np.float64)
    rgb_intrinsics = np.load(map_dir / "rgb_camera_intrinsics.npy", allow_pickle=True).astype(np.float64)
    t_rgb_to_infra1 = np.asarray(
        np.load(map_dir / "T_rgb_to_infra1.npy", allow_pickle=True), dtype=np.float64
    )
    t_infra1_to_rgb = np.linalg.inv(t_rgb_to_infra1)

    _, frames = load_transforms(map_dir)
    rng = np.random.default_rng(0)
    picked = rng.choice(len(frames), size=min(sample_frames, len(frames)), replace=False)

    errors = []
    with shelve.open(str(depth_shelf), flag="r") as depths:
        for index in picked:
            frame = frames[index]
            timestamp = int(Path(frame["file_path"]).stem.split("_")[-1])
            depth = np.asarray(depths[str(timestamp)], dtype=np.float64)
            rows = np.arange(0, depth.shape[0], 16)
            cols = np.arange(0, depth.shape[1], 16)
            u, v = np.meshgrid(cols, rows)
            z = depth[v, u]
            valid = np.isfinite(z) & (z > 0.3) & (z < 5.0)
            if valid.sum() < 50:
                continue
            u, v, z = u[valid], v[valid], z[valid]
            points_infra1 = np.stack(
                (
                    (u - infra_intrinsics[0, 2]) * z / infra_intrinsics[0, 0],
                    (v - infra_intrinsics[1, 2]) * z / infra_intrinsics[1, 1],
                    z,
                ),
                axis=1,
            )

            # Path 1: the converter's projection into the RGB camera.
            points_rgb = points_infra1 @ t_infra1_to_rgb[:3, :3].T + t_infra1_to_rgb[:3, 3]
            in_front = points_rgb[:, 2] > 1e-6
            reference = points_rgb[in_front] @ rgb_intrinsics.T
            reference = reference[:, :2] / reference[:, 2:3]

            # Path 2: through the world and back with the exported matrix.
            camera_to_world = np.asarray(poses[timestamp], dtype=np.float64)
            world = points_infra1[in_front] @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
            matrix = world_to_camera(frame)
            camera = world @ matrix[:3, :3].T + matrix[:3, 3]
            exported = camera @ rgb_intrinsics.T
            exported = exported[:, :2] / exported[:, 2:3]

            errors.append(float(np.median(np.linalg.norm(reference - exported, axis=1))))

    if not errors:
        raise SystemExit("self-check failed: no frame had usable depth")
    error = float(np.median(errors))
    print(f"self-check: {len(errors)} frames, median reprojection disagreement = {error:.4f} px")
    if error > 0.5:
        raise SystemExit(
            "self-check FAILED: the exported world-to-camera matrices disagree with the "
            "converter's projection. Do not run BA on this model."
        )
    print("self-check passed: exported poses are equivalent to the converter's")


def stage_features(map_dir: Path, matcher: str, overlap: int) -> None:
    import pycolmap

    colmap_dir = map_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    database_path = colmap_dir / "database.db"
    if database_path.exists():
        print(f"{database_path} exists; delete it to re-extract")
        return
    images_dir = map_dir / "images"

    # SINGLE: every frame comes from the same physical camera, so one camera entry. The
    # model and parameters have to be spelled out: left to itself COLMAP guesses
    # SIMPLE_RADIAL with a focal length from EXIF, which both clashes with the PINHOLE
    # model in the pose file we hand to triangulation and makes two-view geometric
    # verification run on a wrong calibration.
    data, _ = load_transforms(map_dir)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = f"{data['fl_x']},{data['fl_y']},{data['cx']},{data['cy']}"
    pycolmap.extract_features(
        database_path=database_path,
        image_path=images_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
    )
    print(f"features extracted into {database_path}")

    if matcher == "exhaustive":
        # 255 images is 32k pairs, which COLMAP handles comfortably, and it finds the
        # long-baseline pairs an orbit produces without needing a vocabulary tree.
        pycolmap.match_exhaustive(database_path=database_path)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = overlap
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(database_path=database_path, pairing_options=pairing)
    # Database has no constructor in pycolmap 4.x; open() is a static factory.
    database = pycolmap.Database.open(str(database_path))
    print(
        f"matching done: {database.num_verified_image_pairs()} verified pairs "
        f"over {database.num_images()} images"
    )
    database.close()


def export_model_with_database_ids(map_dir: Path, model_dir: Path) -> int:
    """Same model as `export`, but image ids taken from the database.

    triangulate_points pairs the reconstruction with the database, and the two only line
    up if the image ids match, so the ids cannot be invented here.
    """
    import pycolmap

    database = pycolmap.Database.open(str(map_dir / "colmap" / "database.db"))
    try:
        name_to_id = {image.name: image.image_id for image in database.read_all_images()}
    finally:
        database.close()

    data, frames = load_transforms(map_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            f"1 PINHOLE {data['w']} {data['h']} "
            f"{data['fl_x']} {data['fl_y']} {data['cx']} {data['cy']}\n"
        )
    written = 0
    with (model_dir / "images.txt").open("w", encoding="utf-8") as handle:
        for frame in frames:
            name = Path(frame["file_path"]).name
            if name not in name_to_id:
                continue
            matrix = world_to_camera(frame)
            q = rotation_to_quaternion(matrix[:3, :3])
            t = matrix[:3, 3]
            handle.write(
                f"{name_to_id[name]} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {name}\n\n"
            )
            written += 1
    (model_dir / "points3D.txt").write_text("", encoding="utf-8")
    print(f"model with database ids: {written}/{len(frames)} frames matched a database image")
    return written


def stage_triangulate(map_dir: Path) -> None:
    import pycolmap

    colmap_dir = map_dir / "colmap"
    model_dir = colmap_dir / "sparse_vo_ids"
    export_model_with_database_ids(map_dir, model_dir)
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    output_dir = colmap_dir / "sparse_triangulated"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Poses stay exactly as our VO produced them, so the reprojection error printed below
    # is a measurement of them and nothing else: how far our poses are from explaining the
    # images, in pixels, by COLMAP's own reckoning rather than a homemade estimator.
    result = pycolmap.triangulate_points(
        reconstruction=reconstruction,
        database_path=colmap_dir / "database.db",
        image_path=map_dir / "images",
        output_path=output_dir,
        clear_points=True,
        refine_intrinsics=False,
    )
    print("\n=== triangulation with VO poses held FIXED ===")
    print(result.summary())
    print(f"\n>>> VO poses: mean reprojection error = {result.compute_mean_reprojection_error():.3f} px")
    print("    (3DGS wants this well under 1 px)")


def filter_points(reconstruction, min_track_length: int, max_error: float) -> int:
    """Drop tracks that cannot constrain anything, before they get a vote in BA.

    Two-view tracks with a small triangulation angle have almost no depth information, so
    their 3D position is free to slide along the ray; left in the problem they are exactly
    the residuals a solver will happily send to infinity.
    """
    doomed = [
        point_id
        for point_id, point in reconstruction.points3D.items()
        if point.track.length() < min_track_length
        or (point.has_error and point.error > max_error)
    ]
    for point_id in doomed:
        reconstruction.delete_point3D(point_id)
    return len(doomed)


def stage_sfm(map_dir: Path, single_model: bool = True) -> None:
    """Reconstruct from scratch with COLMAP's incremental mapper, ignoring the VO poses.

    Everything else in this file treats the VO poses as truth and only polishes them, which
    inherits their error: triangulating with them fixed measured 1.711 px on this map, and a
    plain bundle adjustment on top bent the scene rather than fixing it. A standard 3DGS
    pipeline never starts from odometry -- it runs full structure-from-motion, whose poses
    are sub-pixel consistent by construction because registration, retriangulation, global
    BA and outlier filtering all iterate until they are.

    This is affordable here because the expensive part is already done: the database holds
    32385 geometrically verified pairs over 452 images, a dense match graph, so the mapper
    only has to register and optimise.

    Intrinsics stay fixed -- they are measured, and letting SfM refine them would make the
    result incomparable with every other pose set we have.
    """
    import pycolmap

    colmap_dir = map_dir / "colmap"
    database_path = colmap_dir / "database.db"
    if not database_path.exists():
        raise SystemExit(f"{database_path} missing -- run the 'features' stage first")
    output_dir = colmap_dir / "sparse_sfm"
    output_dir.mkdir(parents=True, exist_ok=True)

    options = pycolmap.IncrementalPipelineOptions()
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    if single_model:
        # A split into several models would leave most images unregistered and give us
        # nothing to train on; better to know that happened than to silently get a fragment.
        options.multiple_models = False

    print("running incremental SfM (this is the slow one: registration + repeated global BA)")
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=map_dir / "images",
        output_path=output_dir,
        options=options,
    )
    if not reconstructions:
        raise SystemExit("SfM produced no reconstruction")
    total_images = len(load_transforms(map_dir)[1])
    for index, reconstruction in reconstructions.items():
        print(
            f"  model {index}: {reconstruction.num_reg_images()}/{total_images} images registered, "
            f"{reconstruction.num_points3D()} points, "
            f"mean track length {reconstruction.compute_mean_track_length():.2f}, "
            f"mean reprojection error {reconstruction.compute_mean_reprojection_error():.3f} px"
        )
    best = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    (colmap_dir / "sparse_ba").mkdir(parents=True, exist_ok=True)
    print(f"\n>>> SfM poses: mean reprojection error = {best.compute_mean_reprojection_error():.3f} px")
    print(f"    (VO poses measured 1.711 px on this map; 3DGS wants well under 1 px)")
    best.write(str(colmap_dir / "sparse_ba"))  # reuse the path the validate stage reads
    print(f"    wrote the largest model to {colmap_dir / 'sparse_ba'}")


def stage_bundle_adjust(
    map_dir: Path,
    write_seed_cloud: bool,
    rounds: int = 2,
    min_track_length: int = 3,
    max_error: float = 4.0,
) -> None:
    import pycolmap

    colmap_dir = map_dir / "colmap"
    reconstruction = pycolmap.Reconstruction(str(colmap_dir / "sparse_triangulated"))
    before = reconstruction.compute_mean_reprojection_error()
    poses_before = {
        image.name: np.asarray(image.projection_center(), dtype=np.float64)
        for image in reconstruction.images.values()
    }

    options = pycolmap.BundleAdjustmentOptions()
    # Intrinsics are measured, not estimated: let BA move only the poses and the points.
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_rig_from_world = True
    options.refine_points3D = True
    # BundleAdjustmentOptions defaults to a TRIVIAL loss -- no robust kernel -- which is
    # fine inside the incremental pipeline where tracks have already been vetted, and
    # catastrophic on a freshly triangulated set: on this map it sent the mean reprojection
    # error to 1e146 px. COLMAP's own global BA uses Cauchy, so use it here too.
    options.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    options.ceres.loss_function_scale = 1.0
    options.ceres.solver_options.max_num_iterations = 500
    options.min_track_length = min_track_length

    for round_index in range(rounds):
        dropped = filter_points(reconstruction, min_track_length, max_error)
        # Cauchy saturates on gross outliers, which is what makes it robust -- and also
        # means the solver leaves them wherever they flew off to. So filter after solving
        # as well, not only before.
        # Poses are free, so the problem has seven unconstrained degrees of freedom
        # (rotation, translation, scale). COLMAP fixes that by holding two camera poses;
        # without it the whole reconstruction can drift while every residual stays happy.
        config = pycolmap.BundleAdjustmentConfig()
        for image_id in reconstruction.reg_image_ids():
            config.add_image(image_id)
        config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
        adjuster = pycolmap.create_default_bundle_adjuster(options, config, reconstruction)
        summary = adjuster.solve()
        reconstruction.update_point_3d_errors()
        blown = filter_points(reconstruction, min_track_length, max_error)
        reconstruction.update_point_3d_errors()
        print(
            f"  round {round_index + 1}: dropped {dropped} weak + {blown} diverged tracks, "
            f"{reconstruction.num_points3D()} points left, "
            f"mean reprojection error {reconstruction.compute_mean_reprojection_error():.3f} px, "
            f"ceres termination {summary.termination_type}"
        )

    after = reconstruction.compute_mean_reprojection_error()
    moved = np.array(
        [
            np.linalg.norm(np.asarray(image.projection_center(), dtype=np.float64) - poses_before[image.name])
            for image in reconstruction.images.values()
        ]
    )
    output_dir = colmap_dir / "sparse_ba"
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(output_dir))

    print("\n=== bundle adjustment (poses free, intrinsics fixed) ===")
    print(f">>> mean reprojection error: {before:.3f} px  ->  {after:.3f} px")
    print(
        f"    camera centres moved: median {np.median(moved) * 100:.1f} cm, "
        f"p90 {np.percentile(moved, 90) * 100:.1f} cm, max {moved.max() * 100:.1f} cm"
    )
    write_refined_transforms(map_dir, reconstruction, write_seed_cloud)


def similarity_to_original(refined_centres: np.ndarray, original_centres: np.ndarray):
    """Umeyama alignment of BA'd camera centres back onto the VO ones.

    Bundle adjustment with every pose free leaves 7 degrees of freedom unconstrained --
    the whole reconstruction can rotate, translate and rescale without changing a single
    reprojection residual. Left alone that would put the splat in a different frame from
    the rest of the map (occupancy grid, POIs, planner) and rescale metric geometry that
    came from stereo depth. Solving for that similarity and undoing it pins the gauge
    without touching the reprojection error, and the recovered scale doubles as a check:
    it should come out at 1.
    """
    refined_mean = refined_centres.mean(axis=0)
    original_mean = original_centres.mean(axis=0)
    refined_centred = refined_centres - refined_mean
    original_centred = original_centres - original_mean
    u, singular_values, vt = np.linalg.svd(refined_centred.T @ original_centred)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    rotation = rotation.T
    variance = (refined_centred ** 2).sum()
    scale = float(singular_values @ np.diag(correction).clip(-1, 1)) / variance if variance > 0 else 1.0
    translation = original_mean - scale * rotation @ refined_mean
    return rotation, translation, scale


def write_refined_transforms(map_dir: Path, reconstruction, write_seed_cloud: bool) -> None:
    """transforms_ba.json next to the original; nothing existing is modified."""
    data, frames = load_transforms(map_dir)
    by_name = {image.name: image for image in reconstruction.images.values()}

    original_centres, refined_poses, kept_frames = [], [], []
    for frame in frames:
        name = Path(frame["file_path"]).name
        image = by_name.get(name)
        if image is None or not image.has_pose:
            continue
        world_to_cam = np.eye(4)
        world_to_cam[:3, :4] = image.cam_from_world().matrix()
        camera_to_world_opencv = np.linalg.inv(world_to_cam)
        refined_poses.append(camera_to_world_opencv)
        original_centres.append(
            (np.asarray(frame["transform_matrix"], dtype=np.float64) @ OPENGL_TO_OPENCV)[:3, 3]
        )
        kept_frames.append(frame)

    refined_poses = np.array(refined_poses)
    original_centres = np.array(original_centres)
    rotation, translation, scale = similarity_to_original(refined_poses[:, :3, 3], original_centres)
    print(
        f"    gauge alignment back onto the VO frame: scale {scale:.5f} "
        f"(1.0 = BA kept metric scale)"
    )
    if abs(scale - 1.0) > 0.02:
        print("    WARNING: BA rescaled the scene by more than 2%; stereo depth says otherwise")

    refined_frames, movement = [], []
    for frame, pose in zip(kept_frames, refined_poses):
        aligned = np.eye(4)
        aligned[:3, :3] = rotation @ pose[:3, :3]
        aligned[:3, 3] = scale * rotation @ pose[:3, 3] + translation
        # Copy the frame and replace only the pose: everything else the converter put there
        # -- depth_file_path above all -- has to survive, or depth supervision silently has
        # nothing to supervise with.
        refined_frame = dict(frame)
        refined_frame["transform_matrix"] = (aligned @ OPENGL_TO_OPENCV).tolist()
        refined_frames.append(refined_frame)
        original = np.asarray(frame["transform_matrix"], dtype=np.float64) @ OPENGL_TO_OPENCV
        movement.append(np.linalg.norm(aligned[:3, 3] - original[:3, 3]))

    movement = np.array(movement)
    print(
        f"    pose change after removing the global similarity: median "
        f"{np.median(movement) * 100:.1f} cm, p90 {np.percentile(movement, 90) * 100:.1f} cm"
    )
    refined = dict(data)
    refined["frames"] = refined_frames

    if write_seed_cloud:
        # The stereo seed cloud was built with the old poses, so it now disagrees with the
        # refined cameras by exactly the error we just removed. Rebuild it.
        from tool.convert_to_nerf_format import build_seed_point_cloud, write_point_cloud_ply

        t_rgb_to_infra1 = np.asarray(
            np.load(map_dir / "T_rgb_to_infra1.npy", allow_pickle=True), dtype=np.float64
        )
        infra1_poses: Dict[int, np.ndarray] = {}
        for frame in refined_frames:
            timestamp = int(Path(frame["file_path"]).stem.split("_")[-1])
            camera_to_world_opencv = np.asarray(frame["transform_matrix"], dtype=np.float64) @ OPENGL_TO_OPENCV
            infra1_poses[timestamp] = camera_to_world_opencv @ np.linalg.inv(t_rgb_to_infra1)
        points, colors = build_seed_point_cloud(
            map_dir, infra1_poses, t_rgb_to_infra1, 4, 0.2, 3.0, 1_000_000, 0
        )
        write_point_cloud_ply(map_dir / "sparse_pc_ba.ply", points, colors)
        refined["ply_file_path"] = "sparse_pc_ba.ply"
        print(f"Rebuilt the seed cloud from refined poses: {len(points)} points")

    target = map_dir / "transforms_ba.json"
    with target.open("w", encoding="utf-8") as handle:
        json.dump(refined, handle, indent=2)
    print(f"Wrote {target} ({len(refined_frames)} frames). transforms.json and poses.npy untouched.")
    print("Train on it by pointing the run script at a copy of the map dir, or by")
    print(f"  cp {target} {map_dir / 'transforms.json'}   # after backing the original up")


def depth_agreement(map_dir: Path, transforms_name: str, sample_images: int = 60) -> float:
    """Median relative disagreement between COLMAP's 3D points and the stereo depth map.

    A bundle adjustment that bends the scene and one that corrects real drift both lower
    the reprojection error, so reprojection error cannot tell them apart. Stereo depth
    can: it measures the same geometry through a completely independent channel (one
    baseline, one instant) and it never saw the poses. If the BA'd poses put COLMAP's
    triangulated points closer to what the depth sensor reported, the correction was real.
    """
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(map_dir / "colmap" / "sparse_ba"))
    with (map_dir / transforms_name).open(encoding="utf-8") as handle:
        data = json.load(handle)
    poses = {
        Path(frame["file_path"]).name: np.asarray(frame["transform_matrix"], dtype=np.float64)
        for frame in data["frames"]
    }
    rgb_intrinsics = np.load(map_dir / "rgb_camera_intrinsics.npy", allow_pickle=True).astype(np.float64)
    infra_intrinsics = np.load(map_dir / "intrinsics.npy", allow_pickle=True).astype(np.float64)
    t_rgb_to_infra1 = np.asarray(
        np.load(map_dir / "T_rgb_to_infra1.npy", allow_pickle=True), dtype=np.float64
    )

    rng = np.random.default_rng(0)
    image_list = [image for image in reconstruction.images.values() if image.name in poses]
    picked = rng.choice(len(image_list), size=min(sample_images, len(image_list)), replace=False)

    # The reconstruction and the pose file need not share a gauge: a from-scratch SfM model
    # carries an arbitrary similarity, and even the refine path writes poses that were
    # aligned back onto the VO frame afterwards. Comparing 3D points from one gauge against
    # depth measured in the other would be measuring the gauge, not the geometry, so solve
    # for the similarity between the two camera-centre clouds and move the points with it.
    reconstruction_centres = np.array(
        [np.asarray(image.projection_center(), dtype=np.float64) for image in image_list]
    )
    json_centres = np.array(
        [(poses[image.name] @ OPENGL_TO_OPENCV)[:3, 3] for image in image_list]
    )
    gauge_rotation, gauge_translation, gauge_scale = similarity_to_original(
        reconstruction_centres, json_centres
    )
    residual = np.linalg.norm(
        (gauge_scale * (gauge_rotation @ reconstruction_centres.T).T + gauge_translation) - json_centres,
        axis=1,
    )
    print(
        f"  gauge fit to {transforms_name}: scale {gauge_scale:.4f}, "
        f"centre residual median {np.median(residual) * 100:.1f} cm"
    )

    ratios: List[float] = []
    with shelve.open(str(map_dir / "depths"), flag="r") as depths:
        for index in picked:
            image = image_list[index]
            timestamp = int(Path(image.name).stem.split("_")[-1])
            if str(timestamp) not in depths:
                continue
            depth_map = np.asarray(depths[str(timestamp)], dtype=np.float64)
            camera_to_world = poses[image.name] @ OPENGL_TO_OPENCV
            world_to_camera = np.linalg.inv(camera_to_world)
            for point2D in image.points2D:
                if not point2D.has_point3D():
                    continue
                xyz = np.asarray(reconstruction.points3D[point2D.point3D_id].xyz, dtype=np.float64)
                xyz = gauge_scale * (gauge_rotation @ xyz) + gauge_translation
                in_rgb = world_to_camera[:3, :3] @ xyz + world_to_camera[:3, 3]
                if in_rgb[2] <= 0.3 or in_rgb[2] > 5.0:
                    continue
                # The depth map lives in the infra1 frame, so walk the point over there
                # and read the sensor's own answer at the pixel it lands on.
                in_infra1 = t_rgb_to_infra1[:3, :3] @ in_rgb + t_rgb_to_infra1[:3, 3]
                if in_infra1[2] <= 0.3:
                    continue
                u = int(round(infra_intrinsics[0, 0] * in_infra1[0] / in_infra1[2] + infra_intrinsics[0, 2]))
                v = int(round(infra_intrinsics[1, 1] * in_infra1[1] / in_infra1[2] + infra_intrinsics[1, 2]))
                if not (0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]):
                    continue
                measured = depth_map[v, u]
                if not np.isfinite(measured) or measured <= 0.3 or measured > 5.0:
                    continue
                ratios.append(abs(in_infra1[2] - measured) / measured)
    if not ratios:
        raise SystemExit("no point landed on a valid depth pixel; cannot validate")
    ratios = np.array(ratios)
    print(
        f"  {transforms_name}: {len(ratios)} point-depth comparisons, "
        f"median relative disagreement {np.median(ratios) * 100:.2f}%, "
        f"p90 {np.percentile(ratios, 90) * 100:.2f}%"
    )
    return float(np.median(ratios))


def stage_write(map_dir: Path, write_seed_cloud: bool) -> None:
    """Turn whatever is in colmap/sparse_ba into transforms_ba.json.

    Used after the `sfm` stage, whose model is already bundle-adjusted -- running another BA
    on it would only add noise. Note that a from-scratch SfM reconstruction has an arbitrary
    similarity gauge (its scale is whatever the initial pair happened to imply), so unlike
    the refine path, the alignment onto the VO frame here is doing real work and its scale
    factor is expected to differ from 1.
    """
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(map_dir / "colmap" / "sparse_ba"))
    print(
        f"model: {reconstruction.num_reg_images()} images, {reconstruction.num_points3D()} points, "
        f"mean reprojection error {reconstruction.compute_mean_reprojection_error():.3f} px"
    )
    write_refined_transforms(map_dir, reconstruction, write_seed_cloud)


def stage_validate(map_dir: Path) -> None:
    print("=== independent check: COLMAP points vs stereo depth ===")
    print("(same 3D points, judged by the depth sensor, which never saw either pose set)")
    vo = depth_agreement(map_dir, "transforms.json")
    ba = depth_agreement(map_dir, "transforms_ba.json")
    if ba < vo:
        print(f"\n>>> BA improved depth agreement ({vo * 100:.2f}% -> {ba * 100:.2f}%): the pose")
        print("    correction is real, not the solver bending the scene to fit.")
    else:
        print(f"\n>>> BA made depth agreement WORSE ({vo * 100:.2f}% -> {ba * 100:.2f}%): the lower")
        print("    reprojection error came from deforming the scene. Do not train on these poses.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    export_parser = subparsers.add_parser("export", help="write a COLMAP text model from transforms.json")
    export_parser.add_argument("--map-dir", required=True)
    export_parser.add_argument("--model-dir", default=None, help="default: <map-dir>/colmap/sparse_vo")
    export_parser.add_argument("--skip-check", action="store_true")

    features_parser = subparsers.add_parser("features", help="extract SIFT features and match")
    features_parser.add_argument("--map-dir", required=True)
    features_parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    features_parser.add_argument("--overlap", type=int, default=15, help="sequential matcher window")

    triangulate_parser = subparsers.add_parser(
        "triangulate", help="triangulate with poses fixed and report the pose error in px"
    )
    triangulate_parser.add_argument("--map-dir", required=True)

    ba_parser = subparsers.add_parser("ba", help="bundle adjust and write transforms_ba.json")
    ba_parser.add_argument("--map-dir", required=True)
    ba_parser.add_argument(
        "--no-seed-cloud",
        dest="write_seed_cloud",
        action="store_false",
        help="keep the old sparse_pc.ply instead of rebuilding it from the refined poses",
    )

    sfm_parser = subparsers.add_parser(
        "sfm", help="full COLMAP structure-from-motion, ignoring the VO poses entirely"
    )
    sfm_parser.add_argument("--map-dir", required=True)
    sfm_parser.add_argument("--allow-multiple-models", dest="single_model", action="store_false")

    write_parser = subparsers.add_parser(
        "write", help="write transforms_ba.json from colmap/sparse_ba without re-running BA"
    )
    write_parser.add_argument("--map-dir", required=True)
    write_parser.add_argument("--no-seed-cloud", dest="write_seed_cloud", action="store_false")

    validate_parser = subparsers.add_parser(
        "validate", help="judge the BA'd poses against stereo depth, an independent channel"
    )
    validate_parser.add_argument("--map-dir", required=True)

    all_parser = subparsers.add_parser("all", help="export + features + triangulate + ba + validate")
    all_parser.add_argument("--map-dir", required=True)
    all_parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    all_parser.add_argument("--overlap", type=int, default=15)
    all_parser.add_argument("--no-seed-cloud", dest="write_seed_cloud", action="store_false")

    args = parser.parse_args()
    map_dir = Path(args.map_dir)

    if args.stage in ("export", "all"):
        model_dir = (
            Path(args.model_dir)
            if args.stage == "export" and args.model_dir
            else map_dir / "colmap" / "sparse_vo"
        )
        count = export_colmap_model(map_dir, model_dir)
        print(f"Wrote a COLMAP model for {count} images to {model_dir}")
        if args.stage == "all" or not args.skip_check:
            check_projection_roundtrip(map_dir)
    if args.stage in ("features", "all"):
        stage_features(map_dir, args.matcher, args.overlap)
    if args.stage == "sfm":
        stage_sfm(map_dir, args.single_model)
    if args.stage == "write":
        stage_write(map_dir, args.write_seed_cloud)
    if args.stage in ("triangulate", "all"):
        stage_triangulate(map_dir)
    if args.stage in ("ba", "all"):
        stage_bundle_adjust(map_dir, args.write_seed_cloud)
    if args.stage in ("validate", "all"):
        stage_validate(map_dir)


if __name__ == "__main__":
    main()
