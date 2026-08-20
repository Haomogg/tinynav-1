#!/bin/bash
# TinyNav map -> 3DGS (nerfstudio splatfacto), tuned for quality over speed.
#
#   bash scripts/run_3dgs_generation.sh                 # install + convert + train + export
#   bash scripts/run_3dgs_generation.sh --remote         # = --no-install --no-convert --no-uv
#   iterations=30000 bash scripts/run_3dgs_generation.sh --no-install   # quick sanity run
#
# Every value below can be overridden from the command line, e.g.
#   map_path=/tinynav/output/map_record_table drop_blurriest=0 bash scripts/run_3dgs_generation.sh
#
# Running --remote means the GPU box needs these in sync, or it silently trains with
# whatever it had last (a run was wasted this way, using stale defaults):
#   scripts/run_3dgs_generation.sh          this file
#   tool/prune_splat.py                     the post-export prune
#   tool/train_depth_splatfacto.py          only when depth_supervision=true
#   <map>/transforms_ba.json <map>/sparse_pc_ba.ply <map>/images/ <map>/depths_rgb/
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Paths. data_path must hold poses.npy + rgb_images_db (the converter writes
# transforms.json, images/ and sparse_pc.ply into it).
map_path="${map_path:-$repo_root/output/map_record_table}"
data_path="${data_path:-$map_path}"
output_path="${output_path:-$map_path/nerf_output}"

# ---------------------------------------------------------------------------
# Data quality (convert stage)
# ---------------------------------------------------------------------------
# Drop the blurriest frames. On map_record_table sharpness spans 38x (Laplacian
# variance p10 19 / p90 177) and correlates -0.36 with angular velocity: the blur is
# from turning at ~23 deg/s. 458 keyframes over 5 m is redundant, so trading a third
# of them for detail is cheap. Set 0 to keep every frame.
drop_blurriest="${drop_blurriest:-0.33}"
# Seed the gaussians from stereo depth. Without a ply, splatfacto starts from 50k
# random gaussians in a cube (num_random=50000) because nothing in transforms.json
# points at a point cloud -- a normal 3DGS run gets a COLMAP cloud here.
seed_points="${seed_points:-true}"
seed_pixel_stride="${seed_pixel_stride:-4}"
seed_max_points="${seed_max_points:-1000000}"
# Stereo error grows with depth squared, so a far cutoff turns an indoor scene's seed
# cloud into a cloud of noise (30 m on map_record_table: a 1.5 m table produced points
# spread over 38x47x17 m, which become fog gaussians). Set this to the scene, not to the
# sensor's range.
seed_max_depth="${seed_max_depth:-5.0}"
# Drop keyframes whose pose disagrees with their neighbours by more than this (m). One
# broken VO keyframe paints a ghost surface at whatever pose it claims.
max_pose_deviation="${max_pose_deviation:-0.5}"

# ---------------------------------------------------------------------------
# Training length. Everything iteration-coupled is derived from this.
# ---------------------------------------------------------------------------
# 100k = 3.3x the stock 30000. Nothing here is time-constrained (one map, trained
# overnight), and full-resolution detail keeps improving long after the stock schedule
# ends, so buy quality with wall-clock: expect roughly 4-7 h on a TITAN RTX.
iterations="${iterations:-100000}"
# ExponentialDecayScheduler clamps t = clip((step-warmup)/(max_steps-warmup), 0, 1),
# so past max_steps the LR sits exactly at lr_final -- means at 1.6e-6, camera-opt at
# 5e-7. Both default to 30000: leaving them there makes every iteration past 30k a
# no-op for geometry. Scale them with the run.
means_max_steps="${means_max_steps:-$iterations}"
camera_opt_max_steps="${camera_opt_max_steps:-$iterations}"
# Densification is what grows the model; the steps after it are what sharpen it. Half the
# run each way. Watch VRAM during the growth phase: splatfacto has no cap on gaussian count
# (only splatfacto-mcmc does), so on a shared 24 GB card a dense seed cloud plus full
# resolution can walk into an OOM hours in -- lower this or raise densify_grad_thresh if it
# does.
stop_split_at="${stop_split_at:-50000}"
# Stock ratio inside the growth phase (4000/15000).
stop_screen_size_at="${stop_screen_size_at:-$((stop_split_at * 4 / 15))}"
# splatfacto-big is exactly splatfacto with cull_alpha_thresh 0.1 -> 0.005 (set below)
# and this threshold 0.0008 -> 0.0005, i.e. it splits gaussians on a weaker gradient
# signal and ends up with a denser, more detailed model. Passing the two values keeps
# the `splatfacto` method name, so output paths and the export command stay put. Raise
# back to 0.0008 if VRAM runs out.
densify_grad_thresh="${densify_grad_thresh:-0.0005}"

# ---------------------------------------------------------------------------
# Appearance and shape. splatfacto's defaults assume COLMAP-grade poses and a camera with
# fixed exposure; of those two, the pose assumption is now satisfied (see docs -- train on
# the SfM poses from tool/colmap_ba_refine.py, not on poses.npy), but the rig's RGB stream
# still runs auto-exposure for the whole capture, which the bilateral grid handles.
# The knobs that fought bad poses -- low SH degree, scale regularization, depth supervision
# -- measurably hurt once the poses are good, and their defaults reflect that.
# ---------------------------------------------------------------------------
# The ISP moves between frames, so the same surface arrives at different exposure and
# white balance; without a per-image correction the model can only average them, which
# reads as haze. Bilateral Guided Radiance Field Processing; nerfstudio already ships the
# optimizer group for it (lr 2e-3), just switched off.
use_bilateral_grid="${use_bilateral_grid:-True}"
# 0.005 is splatfacto-big's value, chosen to keep faint gaussians for extra detail. With
# noisy poses those faint gaussians are the fog, so go back to the stock 0.1 and let them
# be culled. This is the one knob to flip first if the result is still hazy.
cull_alpha_thresh="${cull_alpha_thresh:-0.1}"
# Degree 1 was worth trying while the poses were bad: with 10-20 px of pose noise, degree 3
# explains "same point, different colour each frame" as view-dependent specularity and
# surfaces turn into a shimmering film. Once the poses come from SfM that pressure is gone
# and degree 3 is simply more appearance detail, so it wins.
sh_degree="${sh_degree:-3}"
# Only has an effect when use_scale_regularization is on, which by default it is not --
# see the note there. Left at 4 because that is the value that did suppress streaks when
# the regularizer was in use.
max_gauss_ratio="${max_gauss_ratio:-4.0}"
# strategy=mcmc switches densification to 3DGS-MCMC: dead gaussians get relocated instead
# of split, with opacity and scale regularization built in. It is markedly less sensitive
# to a bad initialization than the default strategy, which is the situation here. Note
# that mcmc ignores densify_grad_thresh / stop_split_at and obeys max_gs_num instead.
strategy="${strategy:-default}"
max_gs_num="${max_gs_num:-1000000}"
mcmc_opacity_reg="${mcmc_opacity_reg:-0.01}"
mcmc_scale_reg="${mcmc_scale_reg:-0.01}"
# antialiased fixes the aliasing of tiny gaussians at full resolution, but nerfstudio
# warns that a PLY exported in this mode renders wrong in classic-mode viewers -- which
# includes tool/poi_editor.py, i.e. exactly how we judge the result. Left on classic
# deliberately; only switch it if the viewer is switched too.
rasterize_mode="${rasterize_mode:-classic}"
# Off, which is both the stock nerfstudio setting and what the A/B comparison chose. It
# reshapes *every* gaussian toward max_gauss_ratio rather than trimming outliers (measured:
# the whole model's anisotropy piles up at whatever limit is set, p99 4.1 at ratio 4), and a
# gaussian lying on a wall is legitimately a thin disc. Forcing discs toward spheres makes
# surfaces grainy and edges blunt, which is visible on desks and faces. The streaks it was
# meant to prevent are better handled after export, where tool/prune_splat.py can tell a
# needle (one long axis) from a disc (two) and delete only the needle.
use_scale_regularization="${use_scale_regularization:-False}"

# ---------------------------------------------------------------------------
# Depth supervision. splatfacto renders depth and compares it to nothing, so a surface
# explained by two layers of gaussians at different distances costs the same as one layer
# -- that is what the ghosting is. We have metric stereo depth, so supervise with it.
# Requires the converter to have been run with --depth-maps.
# ---------------------------------------------------------------------------
# L1 and SSIM split the photometric loss (1-lambda)*L1 + lambda*SSIM. L1 is happy with a
# blurry average, SSIM scores local structure, so raising it is the knob that buys visible
# texture rather than merely lower per-pixel error.
ssim_lambda="${ssim_lambda:-0.2}"
# Passed through to the depth-supervision launcher; see tool/train_depth_splatfacto.py.
opacity_entropy_mult="${opacity_entropy_mult:-0.01}"
accumulation_mult="${accumulation_mult:-0.05}"

# ---------------------------------------------------------------------------
# Pruning (after export). Training cannot see which of its gaussians are junk: a fragment
# floating in mid-air explains a little colour from the few cameras that cover it and is
# invisible to every other view. After training we can test each gaussian against the
# metric depth maps -- in front of a measured surface means free space, and free space is
# carvable -- plus shape and connectivity. See tool/prune_splat.py for what each rule does.
# ---------------------------------------------------------------------------
prune="${prune:-true}"
prune_max_height="${prune_max_height:-}"              # empty = keep the ceiling
prune_require_support_beyond="${prune_require_support_beyond:-1.5}"
prune_min_support="${prune_min_support:-5}"
prune_isolation_radius="${prune_isolation_radius:-0.10}"
# Applied in one pass, on the freshly exported ply. Do not re-run the pruner on an already
# pruned file: removing gaussians breaks connectivity, so the cluster rule then deletes
# whole regions it would have kept (measured: 51% of an already-pruned model).
prune_keep_largest_cluster="${prune_keep_largest_cluster:-true}"

depth_supervision="${depth_supervision:-false}"
depth_loss_mult="${depth_loss_mult:-0.2}"
depth_loss_max_depth="${depth_loss_max_depth:-6.0}"

# Pose refinement. splatfacto ships camera-optimizer.mode=off. TinyNav poses come from
# stereo VO, not SfM, and disagree frame-to-frame by ~9 mm (a few pixels at 1.5 m), so
# let training absorb that. Only nerfstudio has this knob -- FastGS and the original
# 3DGS codebase cannot refine poses at all. Set to off to disable.
camera_optimizer_mode="${camera_optimizer_mode:-SO3xR3}"
# splatfacto's camera-opt LR is 1e-4, sized for COLMAP poses that are already
# sub-pixel-consistent. Measured on this map (SIFT+depth+PnP vs the map's relative
# poses), our VO poses disagree with image geometry by ~2-3 cm and ~1 deg, which at
# fx=768 and 1.5 m is 10-20 px -- an order of magnitude more than 1e-4 can walk back
# during training. nerfacto uses 1e-3 for the same optimizer. Lower it back to 1e-4 if
# the run goes unstable (loss spiking, gaussians drifting off-scene).
camera_opt_lr="${camera_opt_lr:-1e-3}"

# cache_images defaults to "gpu" with float32 tensors, and the automatic downgrade to
# cpu only triggers above 500 images. ~300 full-resolution frames would pin ~7.6 GB of
# VRAM for image data alone, which matters on a shared 24 GB card.
cache_images="${cache_images:-cpu}"
# splatfacto overrides the eval cadence down to 100 / 1000 steps. Over a 100k run that
# is a thousand full-resolution renders plus LPIPS, all of it pure overhead here.
steps_per_eval_image="${steps_per_eval_image:-2000}"
steps_per_eval_all_images="${steps_per_eval_all_images:-20000}"

# Resolution note: splatfacto's progressive schedule (num_downscales=2,
# resolution_schedule=3000) trains at 1/4 res until step 3000 and 1/2 until 6000, then
# native 1088x1920 for the rest -- coarse-to-fine, worth keeping. The dataparser's own
# auto-downscale never fires because it requires an images_2/ folder to exist, and
# --downscale-factor 1 pins that explicitly.
downscale_factor="${downscale_factor:-1}"

# Which GPU to run on, e.g. cuda_devices=1. Exported for the whole script, not just
# ns-train: the export step is a second CUDA process, and without this it would land on
# device 0 hours later, which on a shared box is usually someone else's job. Left unset
# it means "every visible GPU", and both nerfstudio steps then take device 0.
cuda_devices="${cuda_devices:-}"
if [[ -n "$cuda_devices" ]]; then
    export CUDA_VISIBLE_DEVICES="$cuda_devices"
fi

# Resume an interrupted run instead of starting over: a dropped ssh session kills a
# multi-hour run, but the trainer checkpoints every 2000 steps (keeping only the latest,
# save_only_latest_checkpoint=True). `resume_dir=auto` loads this run's own newest
# checkpoint and keeps going to $iterations; a path loads from somewhere else. Empty
# starts from scratch. splatfacto's load_state_dict resizes the gaussian tensors to the
# checkpoint's count, so a mid-densification resume is fine.
resume_dir="${resume_dir:-}"
if [[ "$resume_dir" == "auto" ]]; then
    resume_dir="$output_path/experiment/splatfacto/0/nerfstudio_models"
fi
train_args=()
if [[ -n "$resume_dir" ]]; then
    if [[ ! -d "$resume_dir" ]]; then
        echo "resume_dir is not a directory: $resume_dir" >&2
        exit 2
    fi
    train_args+=(--load-dir "$resume_dir")
fi
run_install=true      # `uv pip install .[3dgs]`  -> --no-install to skip
run_convert=true      # TinyNav map -> transforms.json + sparse_pc.ply (CPU)
runner="uv run"       # command prefix; --no-uv sets it to "" (e.g. conda on a remote box)

for arg in "$@"; do
    case "$arg" in
        --remote)     run_install=false; run_convert=false; runner="" ;;  # = the three below
        --no-install) run_install=false ;;
        --no-convert) run_convert=false ;;
        --no-uv)      runner="" ;;
        *) echo "Unknown argument: $arg (use --remote / --no-install / --no-convert / --no-uv)" >&2; exit 2 ;;
    esac
done

if [[ "$run_install" == "true" ]]; then
    uv pip install .[3dgs]
fi

if [[ "$run_convert" == "true" ]]; then
    convert_args=(
        --map-dir "$data_path"
        --drop-blurriest "$drop_blurriest"
        --max-pose-deviation "$max_pose_deviation"
    )
    if [[ "$seed_points" == "true" ]]; then
        convert_args+=(
            --pixel-stride "$seed_pixel_stride"
            --max-points "$seed_max_points"
            --max-depth "$seed_max_depth"
        )
    else
        convert_args+=(--no-seed-points)
    fi
    $runner python tool/convert_to_nerf_format.py "${convert_args[@]}"
fi

# The dev container (uv venv) and the GPU box (conda) do not run the same nerfstudio
# release, and the newer knobs -- strategy/mcmc, bilateral grid -- simply do not exist in
# older ones. Passing an unknown flag makes tyro abort after the imports, i.e. half a
# minute into what was supposed to be an overnight run, so ask the installed ns-train
# what it accepts and drop the rest with a note.
supported_flags="$($runner ns-train splatfacto --help 2>/dev/null || true)"
if [[ -z "$supported_flags" ]]; then
    echo "Warning: 'ns-train splatfacto --help' produced nothing; passing core flags only" >&2
fi
add_model_flag() {
    local flag="$1" value="$2"
    if [[ -n "$supported_flags" ]] && grep -q -- "$flag" <<<"$supported_flags"; then
        train_args+=("$flag" "$value")
    else
        echo "  note: this nerfstudio has no $flag -- skipped (wanted $value)"
    fi
}

add_model_flag --pipeline.model.cull-alpha-thresh "$cull_alpha_thresh"
add_model_flag --pipeline.model.sh-degree "$sh_degree"
add_model_flag --pipeline.model.max-gauss-ratio "$max_gauss_ratio"
add_model_flag --pipeline.model.rasterize-mode "$rasterize_mode"
add_model_flag --pipeline.model.use-bilateral-grid "$use_bilateral_grid"
add_model_flag --pipeline.model.use-scale-regularization "$use_scale_regularization"
add_model_flag --pipeline.model.ssim-lambda "$ssim_lambda"

train_entry="ns-train"
if [[ "$depth_supervision" == "true" ]]; then
    # tool/train_depth_splatfacto.py patches splatfacto's datamanager and loss, then hands
    # over to ns-train's own entrypoint, so every flag below still applies. It imports only
    # nerfstudio and torch, which is what makes it work on the GPU box where this repo is
    # not installed.
    train_entry="python $repo_root/tool/train_depth_splatfacto.py"
    export DEPTH_LOSS_MULT="$depth_loss_mult"
    export DEPTH_LOSS_MAX_DEPTH="$depth_loss_max_depth"
    export OPACITY_ENTROPY_MULT="$opacity_entropy_mult"
    export ACCUMULATION_MULT="$accumulation_mult"
    # Without this splatfacto skips the depth render while training, leaving nothing to
    # compare the measured depth against.
    add_model_flag --pipeline.model.output-depth-during-training True
    if ! grep -q '"depth_file_path"' "$data_path/transforms.json" 2>/dev/null \
       && ! grep -q '"depth_file_path"' "$data_path" 2>/dev/null; then
        echo "depth_supervision=true but no depth_file_path in $data_path -- rerun the" >&2
        echo "  converter with --depth-maps first." >&2
        exit 2
    fi
fi
if [[ "$strategy" != "default" ]]; then
    if grep -q -- "--pipeline.model.strategy" <<<"$supported_flags"; then
        train_args+=(--pipeline.model.strategy "$strategy")
        add_model_flag --pipeline.model.max-gs-num "$max_gs_num"
        add_model_flag --pipeline.model.mcmc-opacity-reg "$mcmc_opacity_reg"
        add_model_flag --pipeline.model.mcmc-scale-reg "$mcmc_scale_reg"
    else
        echo "  WARNING: strategy=$strategy requested but this nerfstudio only has the" >&2
        echo "           default densification strategy; running with it instead." >&2
    fi
fi

echo "Training $iterations iterations: split until $stop_split_at (grad thresh $densify_grad_thresh),"
echo "  screen-size until $stop_screen_size_at, LR schedules stretched to $means_max_steps,"
echo "  camera optimizer $camera_optimizer_mode (lr $camera_opt_lr), downscale $downscale_factor"
echo "  GPU: ${cuda_devices:-<unpinned, will use device 0>}"
echo "  resume: ${resume_dir:-<from scratch>}"
echo "  anti-mush: strategy $strategy, cull_alpha $cull_alpha_thresh, sh_degree $sh_degree,"
echo "    bilateral_grid $use_bilateral_grid, max_gauss_ratio $max_gauss_ratio, rasterize $rasterize_mode"

# torch 2.6 flipped torch.load's weights_only default to True, and nerfstudio loads
# checkpoints without passing the argument (utils/eval_utils.py, engine/trainer.py), so
# both the export below and a resume above die on the numpy scalar inside a checkpoint.
# torch only honours this variable when the callsite left weights_only unset, which is
# exactly nerfstudio's case. These are our own checkpoints, so nothing is trusted here
# that was not already trusted.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

MAX_JOBS=1 $runner $train_entry splatfacto \
      --output-dir "$output_path" \
      --experiment-name experiment \
      --method-name splatfacto \
      --timestamp 0 \
      ${train_args[@]+"${train_args[@]}"} \
      --max-num-iterations "$iterations" \
      --steps-per-eval-image "$steps_per_eval_image" \
      --steps-per-eval-all-images "$steps_per_eval_all_images" \
      --optimizers.means.scheduler.max-steps "$means_max_steps" \
      --optimizers.camera-opt.scheduler.max-steps "$camera_opt_max_steps" \
      --optimizers.camera-opt.optimizer.lr "$camera_opt_lr" \
      --pipeline.model.stop-split-at "$stop_split_at" \
      --pipeline.model.stop-screen-size-at "$stop_screen_size_at" \
      --pipeline.model.camera-optimizer.mode "$camera_optimizer_mode" \
      --pipeline.model.densify-grad-thresh "$densify_grad_thresh" \
      --pipeline.datamanager.cache-images "$cache_images" \
      --viewer.quit-on-train-completion True \
    nerfstudio-data \
      --data "$data_path" \
      --downscale-factor "$downscale_factor" \
      --center-method none \
      --auto-scale-poses False \
      --orientation_method none

$runner ns-export gaussian-splat \
      --load-config "$output_path/experiment/splatfacto/0/config.yml" \
      --output-dir "$output_path"

view_path="$output_path/splat.ply"
if [[ "$prune" == "true" ]]; then
    # The pruner needs the same poses training used, which is a json file when data_path
    # points at one (the SfM path) and transforms.json inside the map otherwise.
    if [[ "$data_path" == *.json ]]; then
        prune_transforms="$(basename "$data_path")"
    else
        prune_transforms="transforms.json"
    fi
    prune_args=(
        --map-dir "$map_path"
        --splat "$output_path/splat.ply"
        --output "$output_path/splat_clean.ply"
        --transforms "$prune_transforms"
        --require-support-beyond "$prune_require_support_beyond"
        --min-support "$prune_min_support"
        --isolation-radius "$prune_isolation_radius"
    )
    [[ -n "$prune_max_height" ]] && prune_args+=(--max-height "$prune_max_height")
    [[ "$prune_keep_largest_cluster" == "true" ]] && prune_args+=(--keep-largest-cluster)
    # Never let this lose a finished training run: the pruner needs plyfile/scipy/cv2, which
    # the GPU box's shared environment may not have, and splat.ply is already on disk.
    if $runner python "$repo_root/tool/prune_splat.py" "${prune_args[@]}"; then
        view_path="$output_path/splat_clean.ply"
    else
        echo "Pruning failed; the unpruned $output_path/splat.ply is still there." >&2
    fi
fi

echo
echo "Gaussian PLY: $view_path"
echo "View it with:  uv run python tool/poi_editor.py --tinynav-map-path $map_path --splat-path $view_path"
