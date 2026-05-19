#!/usr/bin/env bash
# Experiment 02: Train Intra-frame Autoencoders at 6 λ values
# Trains autoencoder + affine normalization for each rate point.
# Uses the existing pipeline/gmm_train.py which handles all λ values.
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "=== Experiment 02: Intra-frame Autoencoder Training ==="
echo "  Training 6 models at λ = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]"
echo "  Expected wall time: ~8 hours on MI300X, ~24 hours on T4"

cd "$LEWM_VC_ROOT"

python3 pipeline/gmm_train.py

echo "=== Experiment 02 complete ==="
echo "  Checkpoints saved to: $CHECKPOINT_DIR/"
echo "  Milestone 1 copies:   $CHECKPOINT_M1/"
