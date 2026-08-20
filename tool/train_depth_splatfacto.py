#!/usr/bin/env python3
"""ns-train splatfacto, with the stereo depth supervising the geometry.

Why a launcher instead of a nerfstudio method: registering a custom method needs an
installed package with a `nerfstudio.method_configs` entry point, and the GPU box runs
nerfstudio out of a shared conda environment where this repo is not installed. Patching
two functions at import time and then handing control to ns-train needs nothing installed
and leaves the shared environment untouched.

What the patches do:

  1. splatfacto's datamanager builds an InputDataset, which carries images and nothing
     else. Swapping in DepthDataset makes `batch["depth_image"]` appear, loaded from the
     `depth_file_path` entries the converter writes and rescaled to the camera resolution.
  2. splatfacto renders depth but never compares it to anything, so two layers of
     gaussians at different distances explain a surface exactly as well as one -- which is
     what ghosting is. The added loss is a masked L1 between rendered and measured depth.

Usage mirrors ns-train, with the multiplier in the environment:

    DEPTH_LOSS_MULT=0.2 python tool/train_depth_splatfacto.py \
        --output-dir ... --pipeline.model.output-depth-during-training True \
        nerfstudio-data --data ...

`--pipeline.model.output-depth-during-training True` is required: without it splatfacto
skips the depth render during training and there is nothing to supervise.
"""

import os
import sys

import torch
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datasets.depth_dataset import DepthDataset
from nerfstudio.models.splatfacto import SplatfactoModel

DEPTH_LOSS_MULT = float(os.environ.get("DEPTH_LOSS_MULT", "0.2"))
# Depth further out than this is not supervised. Stereo error grows as z^2/(f*B), so with
# f*B = 30.4 a single pixel of disparity is 13 cm at 2 m and 2.1 m at 8 m: supervising with
# far depth teaches the model the sensor's noise instead of the scene.
DEPTH_MAX = float(os.environ.get("DEPTH_LOSS_MAX_DEPTH", "6.0"))
# Fraction of training steps to wait before the depth loss switches on. Splatfacto spends
# its first thousands of steps at reduced resolution while gaussians are still coarse;
# pinning geometry that early fights densification instead of guiding it.
DEPTH_WARMUP_FRACTION = float(os.environ.get("DEPTH_LOSS_WARMUP_FRACTION", "0.05"))
# Ghosting survives an expected-depth loss through one specific loophole: two half
# transparent layers straddling the true surface render an alpha-weighted depth that lands
# exactly on it (0.5 at 1.95 m + 0.5 at 2.05 m reads as 2.00 m). The loophole is only open
# while both layers stay semi-transparent -- at alpha 0.7 the front layer already occludes
# enough to pull the mean to 1.973 m, which the depth loss then punishes. So these two terms
# close it from the opacity side rather than trying to make the depth loss cleverer.
#
# Entropy on per-gaussian opacity: maximal at alpha 0.5, zero at 0 or 1. It forces the
# optimizer to choose -- delete the duplicate layer, or make it solid and be corrected by
# the depth loss.
OPACITY_ENTROPY_MULT = float(os.environ.get("OPACITY_ENTROPY_MULT", "0.01"))
# Wherever the depth sensor reports a surface, the ray must be fully absorbed. Two
# half-transparent layers accumulate to 0.75, so this penalises them directly.
ACCUMULATION_MULT = float(os.environ.get("ACCUMULATION_MULT", "0.05"))


def patch_datamanager_to_load_depth() -> None:
    FullImageDatamanager.dataset_type = property(lambda self: DepthDataset)


def patch_model_with_depth_loss() -> None:
    original_get_loss_dict = SplatfactoModel.get_loss_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = original_get_loss_dict(self, outputs, batch, metrics_dict)
        if DEPTH_LOSS_MULT <= 0:
            return loss_dict
        measured = batch.get("depth_image")
        rendered = outputs.get("depth")
        if measured is None or rendered is None:
            return loss_dict
        warmup_steps = int(DEPTH_WARMUP_FRACTION * self.config.stop_split_at * 2)
        if self.step < warmup_steps:
            return loss_dict

        # The render follows splatfacto's progressive resolution schedule, so the measured
        # depth has to be downscaled the same way before the two can be compared.
        measured = self._downscale_if_required(measured).to(rendered.device)
        if measured.shape[:2] != rendered.shape[:2]:
            return loss_dict
        if measured.dim() == 2:
            measured = measured[..., None]
        valid = (measured > 0.1) & (measured <= DEPTH_MAX)
        if not bool(valid.any()):
            return loss_dict
        # Plain L1 in metres: our depth is metric and the poses are unscaled
        # (--auto-scale-poses False), so rendered and measured live in the same units and
        # no scale-invariant trickery is needed -- or wanted, since the metric scale is
        # exactly the information a depth sensor adds over multi-view geometry.
        loss_dict["depth_loss"] = DEPTH_LOSS_MULT * torch.abs(rendered[valid] - measured[valid]).mean()

        if ACCUMULATION_MULT > 0:
            accumulation = outputs.get("accumulation")
            if accumulation is not None and accumulation.shape[:2] == valid.shape[:2]:
                # Only where the sensor saw something: empty space should stay empty.
                loss_dict["accumulation_loss"] = (
                    ACCUMULATION_MULT * (1.0 - accumulation[valid].clamp(0, 1)).mean()
                )
        if OPACITY_ENTROPY_MULT > 0:
            alpha = torch.sigmoid(self.opacities).clamp(1e-6, 1 - 1e-6)
            entropy = -(alpha * torch.log(alpha) + (1 - alpha) * torch.log(1 - alpha))
            loss_dict["opacity_entropy"] = OPACITY_ENTROPY_MULT * entropy.mean()
        return loss_dict

    SplatfactoModel.get_loss_dict = get_loss_dict


def main() -> None:
    patch_datamanager_to_load_depth()
    patch_model_with_depth_loss()
    print(
        f"[depth supervision] masked L1, mult {DEPTH_LOSS_MULT}, "
        f"depth <= {DEPTH_MAX} m, warmup {DEPTH_WARMUP_FRACTION:.0%} of the run",
        flush=True,
    )
    from nerfstudio.scripts.train import entrypoint

    sys.argv[0] = "ns-train"
    entrypoint()


if __name__ == "__main__":
    main()
