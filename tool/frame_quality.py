#!/usr/bin/env python3
"""Frame quality helpers shared by the gaussian-splatting converters.

Splatting weights every input frame equally, so frames smeared by fast rotation drag
the whole model toward mush. On `map_record_table` the sharpness spread is 38x between
the best and worst frames (Laplacian variance p10 19 vs p90 177) and correlates -0.36
with angular velocity (median 23 deg/s), -0.07 with linear velocity: the blur comes
from turning too fast, not from walking too fast. A TinyNav map is usually heavily
redundant (hundreds of keyframes over a few metres), so dropping the blurriest third
costs little coverage and buys detail.
"""

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm


def measure_sharpness(image_bgr: np.ndarray) -> float:
    """Variance of the Laplacian at half resolution: higher is sharper.

    Halving first so the score reflects real image structure instead of sensor noise,
    and stays comparable across a map's frames.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_sharpness(map_dir: Path, timestamps: List[int]) -> Dict[int, float]:
    """Sharpness of every listed keyframe, read straight from the map's RGB db."""
    from tool.video_db import VideoDB

    rgb_db = VideoDB(dir_path=str(map_dir / "rgb_images_db"), mode="read")
    scores: Dict[int, float] = {}
    try:
        for timestamp in tqdm(timestamps, desc="Scoring sharpness", unit="img"):
            image = rgb_db.read(timestamp)
            if image is None:
                raise KeyError(f"Missing RGB image for timestamp {timestamp}")
            scores[timestamp] = measure_sharpness(image)
    finally:
        rgb_db.close()
    return scores


def select_sharp_timestamps(
    map_dir: Path,
    timestamps: List[int],
    drop_blurriest: float = 0.0,
    min_sharpness: float = 0.0,
) -> List[int]:
    """Drop motion-blurred keyframes, and always report the sharpness spread.

    `drop_blurriest` is a fraction (0.33 drops the worst third) and needs no knowledge
    of the rig; `min_sharpness` is an absolute cutoff, useful once the numbers for a
    given camera are known. The stricter of the two wins. Both default to off so the
    unfiltered baseline stays reproducible.
    """
    if not 0.0 <= drop_blurriest < 1.0:
        raise ValueError("drop_blurriest must be in [0, 1)")
    if not timestamps:
        raise ValueError("No timestamps to score")

    scores = score_sharpness(map_dir, timestamps)
    values = np.array([scores[t] for t in timestamps])
    print(
        f"Sharpness (Laplacian variance): p10 {np.percentile(values, 10):.0f} / "
        f"median {np.median(values):.0f} / p90 {np.percentile(values, 90):.0f}"
    )

    threshold = min_sharpness
    if drop_blurriest > 0.0:
        threshold = max(threshold, float(np.quantile(values, drop_blurriest)))
    if threshold <= 0.0:
        return list(timestamps)

    kept = [t for t in timestamps if scores[t] >= threshold]
    if len(kept) < 2:
        raise ValueError(
            f"Sharpness threshold {threshold:.0f} would keep only {len(kept)} frames"
        )
    print(f"Sharpness cutoff {threshold:.0f}: keeping {len(kept)}/{len(timestamps)} frames")
    return kept
