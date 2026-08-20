#!/bin/bash
set -euo pipefail

# Usage: run_rosbag_record.sh [--output DIR]
#   DIR may be either the bag directory to create, or an existing directory to
#   record into -- in the latter case a timestamped map_record_* subdir is used.
#   If --output is not given, a timestamped dir is created under XDG_DATA_HOME/tinynav/rosbags.

output_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output|-o) output_dir="$2"; shift 2 ;;
        *) echo "Usage: $0 [--output DIR]" >&2; exit 1 ;;
    esac
done

timestamp="$(date +%Y%m%d_%H%M%S)"

if [ -z "$output_dir" ]; then
    xdg_data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
    output_dir="${xdg_data_home}/tinynav/rosbags"
    mkdir -p "$output_dir"
fi

# ros2 bag record refuses to write into an existing directory, so treat one as a
# record root and create a fresh bag inside it.
if [ -d "$output_dir" ]; then
    output_dir="${output_dir%/}/map_record_${timestamp}"
fi

mkdir -p "$(dirname "$output_dir")"
echo "Recording to ${output_dir}"

ros2 bag record \
    --output "${output_dir}" \
    --max-cache-size 2147483648 \
    /camera/camera/infra1/camera_info \
    /camera/camera/infra1/image_rect_raw \
    /camera/camera/infra1/metadata \
    /camera/camera/infra2/camera_info \
    /camera/camera/infra2/image_rect_raw \
    /camera/camera/infra2/metadata \
    /camera/camera/imu \
    /camera/camera/color/image_raw \
    /camera/camera/color/camera_info \
    /camera/camera/color/image_rect_raw/compressed \
    /camera/camera/vio_image \
    /tf_static
