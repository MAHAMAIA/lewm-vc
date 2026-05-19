#!/usr/bin/env bash
# LeWM-VC Experiment Environment
# Source this file before running experiments: source experiment/env.sh

export LEWM_VC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$LEWM_VC_ROOT/src:$PYTHONPATH"
export DATASET_DIR="$LEWM_VC_ROOT/datasets"
export CHECKPOINT_DIR="$LEWM_VC_ROOT/checkpoints"
export CHECKPOINT_M1="$LEWM_VC_ROOT/checkpoints_milestone1"
export CHECKPOINT_M2="$LEWM_VC_ROOT/checkpoints_milestone2"

echo "LeWM-VC environment ready"
echo "  Root:     $LEWM_VC_ROOT"
echo "  Dataset:  $DATASET_DIR"
echo "  M1 ckpts: $CHECKPOINT_M1"
echo "  M2 ckpts: $CHECKPOINT_M2"
echo "  Device:   $(python3 -c 'import torch; print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))')"
