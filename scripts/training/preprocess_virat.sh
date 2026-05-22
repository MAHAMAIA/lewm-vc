#!/usr/bin/env bash
# Preprocess VIRAT Ground dataset for Track 1 training
#
# Usage:
#   bash scripts/training/preprocess_virat.sh [data_dir]
#
# Default data_dir: datasets/virat_ground/
#
# This script:
#   1. Downloads VIRAT Ground (if not present)
#   2. Converts each video to 256x256 PNG frames
#   3. Organizes into per-video subdirectories
#
# Prerequisites: wget, ffmpeg, ~50 GB free disk space

set -euo pipefail

DATA_DIR="${1:-datasets/virat_ground}"
OUTPUT_DIR="datasets/virat_preprocessed"
VIRAT_URL="https://data.kitware.com/api/v1/collection/56f56db28d777f753209ba9f/download"
MAX_FRAMES_PER_VIDEO=1000
FRAME_RATE=30

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

echo "=== VIRAT Ground Preprocessing ==="
echo "Data dir:   $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Step 1: Download if not present
if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null | head -1)" ]; then
    echo "Downloading VIRAT Ground from Kitware..."
    echo "  URL: $VIRAT_URL"
    echo "  This is a large download (several GB). May take a while."
    wget -q --show-progress "$VIRAT_URL" -O /tmp/virat_ground.zip
    echo "  Extracting..."
    unzip -q /tmp/virat_ground.zip -d "$DATA_DIR"
    rm /tmp/virat_ground.zip
    echo "  Extracted to $DATA_DIR"
else
    echo "Data directory already populated, skipping download."
fi

# Step 2: Find and preprocess video files
VIDEO_COUNT=0
TOTAL_FRAMES=0

echo ""
echo "Preprocessing videos..."

for f in "$DATA_DIR"/VIRAT_S_*.mp4 "$DATA_DIR"/VIRAT_S_*.mpg "$DATA_DIR"/*.mp4 "$DATA_DIR"/*.avi; do
    [ -f "$f" ] || continue

    basename=$(basename "$f")
    name="${basename%.*}"
    output_subdir="$OUTPUT_DIR/$name"

    if [ -d "$output_subdir" ] && [ "$(ls -A "$output_subdir" 2>/dev/null | head -1)" ]; then
        echo "  [skip] $name — already processed"
        continue
    fi

    mkdir -p "$output_subdir"
    echo "  Processing: $name"

    # Get video duration in frames
    n_frames=$(ffprobe -v error -select_streams v:0 -count_packets \
        -show_entries stream=nb_read_packets \
        -of csv=p=0 "$f" 2>/dev/null || echo "$MAX_FRAMES_PER_VIDEO")

    # Limit to max frames
    n_frames=$(( n_frames > MAX_FRAMES_PER_VIDEO ? MAX_FRAMES_PER_VIDEO : n_frames ))
    [ "$n_frames" -eq 0 ] && n_frames="$MAX_FRAMES_PER_VIDEO"

    # Extract frames as PNGs
    ffmpeg -i "$f" \
        -vf "scale=256:256,fps=$FRAME_RATE" \
        -frames:v "$n_frames" \
        -q:v 1 \
        "$output_subdir/frame_%04d.png" \
        -y 2>/dev/null

    actual_frames=$(ls "$output_subdir"/*.png 2>/dev/null | wc -l)
    VIDEO_COUNT=$((VIDEO_COUNT + 1))
    TOTAL_FRAMES=$((TOTAL_FRAMES + actual_frames))
    echo "    $actual_frames frames extracted"
done

echo ""
echo "=== Preprocessing complete ==="
echo "Videos processed: $VIDEO_COUNT"
echo "Total frames:     $TOTAL_FRAMES"
echo "Output:           $OUTPUT_DIR"
