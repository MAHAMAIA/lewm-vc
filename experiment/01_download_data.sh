#!/usr/bin/env bash
# Experiment 01: Download PEViD-HD and UVG datasets
set -euo pipefail
source "$(dirname "$0")/env.sh"

PEVID_URL="https://data.vision.ee.ethz.ch/cvl/PEViD-HD/PEViD-HD.zip"
UVG_URL="http://ultravideo.fi/UVG_videos/UVG.zip"

mkdir -p "$DATASET_DIR"

echo "=== Experiment 01: Downloading PEViD-HD ==="
if [ -d "$DATASET_DIR/pevid-hd" ] && [ "$(ls -A "$DATASET_DIR/pevid-hd" 2>/dev/null)" ]; then
    echo "  PEViD-HD already exists, skipping."
else
    echo "  Downloading from $PEVID_URL ..."
    wget -q --show-progress "$PEVID_URL" -O /tmp/pevid.zip
    mkdir -p "$DATASET_DIR/pevid-hd"
    unzip -q /tmp/pevid.zip -d "$DATASET_DIR/pevid-hd/"
    rm /tmp/pevid.zip
    echo "  PEViD-HD downloaded to $DATASET_DIR/pevid-hd/"
fi

echo "=== Experiment 01: Downloading UVG ==="
if [ -d "$DATASET_DIR/uvg" ] && [ "$(ls -A "$DATASET_DIR/uvg" 2>/dev/null)" ]; then
    echo "  UVG already exists, skipping."
else
    echo "  Downloading from $UVG_URL ..."
    wget -q --show-progress "$UVG_URL" -O /tmp/uvg.zip
    mkdir -p "$DATASET_DIR/uvg"
    unzip -q /tmp/uvg.zip -d "$DATASET_DIR/uvg/"
    rm /tmp/uvg.zip
    echo "  UVG downloaded to $DATASET_DIR/uvg/"
fi

echo "=== Experiment 01 complete ==="
