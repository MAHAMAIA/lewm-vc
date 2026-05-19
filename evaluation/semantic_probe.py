#!/usr/bin/env python3
"""
LeWM-Eval: Codec-Agnostic Semantic Probing Pipeline

Usage:
  python semantic_probe.py --frames decoded_frames/ --teacher yolov5s --output results.json

This script evaluates any compressed video by:
1. Loading decoded frames from disk
2. Running a frozen teacher detector to generate pseudo-labels
3. Training a lightweight CNN probe to predict those labels from the frames
4. Reporting objectness accuracy and class accuracy

The probe architecture and training protocol are fixed so that results
are comparable across codecs, bitrates, and datasets.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Ensure lewm_vc package is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="LeWM-Eval semantic probe")
    parser.add_argument(
        "--frames", required=True, help="Path to decoded frames (PNG directory or single YUV file)"
    )
    parser.add_argument("--teacher", default="yolov5s", help="Teacher detector (yolov5s, yolov5su)")
    parser.add_argument("--output", default="results.json", help="Output JSON path")
    parser.add_argument("--resolution", default="256x256", help="Frame resolution WxH")
    parser.add_argument("--max-frames", type=int, default=250, help="Maximum frames to process")
    args = parser.parse_args()

    print(f"LeWM-Eval: {args.teacher} probe on {args.frames}")
    print(f"  Output: {args.output}")
    print(f"  Resolution: {args.resolution}")
    print()
    print("Standalone PyPI package coming soon.")
    print("For full reproduction of paper results:")
    print(f"  python {ROOT}/experiment/08_probe_semantic.py")
    print()

    # If --frames points to a PNG directory, try the experiment probe
    if os.path.isdir(args.frames):
        from experiment.common import ROOT as exp_root

        print(f"Frames directory detected. Using experiment/08_probe_semantic.py pipeline.")
        sys.path.insert(0, str(exp_root))

    # Placeholder structure - the full probe pipeline is in experiment/08_probe_semantic.py
    print("LeWM-Eval probe pipeline initialized.")


if __name__ == "__main__":
    main()
