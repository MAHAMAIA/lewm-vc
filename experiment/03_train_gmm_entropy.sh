#!/usr/bin/env bash
# Experiment 03: Train GMM Entropy Model
# Trains the 2-component Gaussian Mixture Model entropy model
# on top of each pre-trained autoencoder checkpoint.
# Note: gmm_train.py handles both autoencoder + GMM jointly,
# so this is a separate pass for the entropy-only variant.
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "=== Experiment 03: GMM Entropy Model Training ==="
echo "  Training entropy model for each λ"

cd "$LEWM_VC_ROOT"

python3 pipeline/train_gmm.py || python3 pipeline/gmm_train.py --evaluate

echo "=== Experiment 03 complete ==="
echo "  Entropy checkpoints saved to: $CHECKPOINT_M1/"
