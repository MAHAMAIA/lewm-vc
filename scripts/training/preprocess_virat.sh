#!/usr/bin/env bash
# Preprocess VIRAT Ground dataset (delegates to Python download_virat.py)
#
# Usage:
#   bash scripts/training/preprocess_virat.sh           # default: datasets/virat/
#   bash scripts/training/preprocess_virat.sh --help     # show Python script help
#
# See: scripts/download_virat.py for the canonical download/preprocessing script.
# This wrapper is kept for backward compatibility with Track 1 Colab notebooks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$SCRIPT_DIR/download_virat.py" "$@"
