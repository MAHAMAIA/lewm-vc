#!/usr/bin/env python3
"""
MI300X Training Sprint — Full Run
Sequences all steps end-to-end:
  1. Extract FPN features from SeaDronesSee + SMD
  2. Ablate pyramid levels (P3/P4/P5) to pick the best
  3. Full retrain compressor + temporal predictor on chosen level
  4. Validate with PSNR, BPP, downstream mAP, latent probe
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SPRINT_DIR = Path(__file__).resolve().parent


def log(msg: str):
    t = time.strftime("%H:%M:%S")
    print(f"\n[{t}] {msg}")


def run(cmd: list[str], desc: str, cwd: str | None = None) -> int:
    log(f"Running: {desc}")
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or str(SPRINT_DIR))
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="MI300X FPN Training Sprint Orchestrator")
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        choices=["all", "extract", "ablate", "train", "validate", "1", "2", "3", "4"],
        help="Which step(s) to run",
    )
    parser.add_argument("--seadronessee", type=str, default="datasets/seadronessee/frames")
    parser.add_argument("--smd", type=str, default="datasets/smd/frames")
    parser.add_argument("--feature-dir", type=str, default="datasets/fpn_features")
    parser.add_argument("--coco-dir", type=str, default="datasets/coco")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ablate-levels", type=str, nargs="+", default=["P3", "P4", "P5"])
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hyper-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ablate-steps", type=int, default=10000)
    parser.add_argument(
        "--train-steps",
        type=int,
        default=38000,
        help="Total training steps (warmup + rd + temporal)",
    )
    args = parser.parse_args()

    steps_map = {
        "all": [1, 2, 3, 4],
        "extract": [1],
        "ablate": [2],
        "train": [3],
        "validate": [4],
        "1": [1],
        "2": [2],
        "3": [3],
        "4": [4],
    }
    steps = steps_map[args.steps]
    overall_t0 = time.time()

    # ── Step 1: Extract FPN features ──────────────────────────────────
    if 1 in steps:
        log("Step 1/4: Extracting FPN features (P3/P4/P5) from datasets")
        rc = run(
            [
                sys.executable,
                "extract_features.py",
                "--seadronessee",
                args.seadronessee,
                "--smd",
                args.smd,
                "--output",
                args.feature_dir,
                "--levels",
                *args.ablate_levels,
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
            ],
            "FPN feature extraction",
        )
        if rc != 0:
            print("  Aborting — feature extraction failed")
            sys.exit(1)

    # ── Step 2: Ablation ──────────────────────────────────────────────
    if 2 in steps:
        log("Step 2/4: Running pyramid level ablation")
        rc = run(
            [
                sys.executable,
                "ablate_pyramid.py",
                "--feature-dir",
                args.feature_dir,
                "--levels",
                *args.ablate_levels,
                "--latent-dim",
                str(args.latent_dim),
                "--hyper-channels",
                str(args.hyper_channels),
                "--batch-size",
                str(args.batch_size),
                "--train-steps",
                str(args.ablate_steps),
                "--device",
                args.device,
                "--output",
                "ablations/fpn_pyramid",
            ],
            "FPN level ablation",
        )
        if rc != 0:
            print("  Aborting — ablation failed")
            sys.exit(1)

        # Read recommendation
        report_path = SPRINT_DIR / "ablations" / "fpn_pyramid" / "ablation_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            fpn_level = report.get("recommended_level", "P4")
            log(f"Ablation recommends: {fpn_level}")
        else:
            fpn_level = "P4"
            log(f"No ablation report found, defaulting to {fpn_level}")
    else:
        fpn_level = "P4"

    # ── Step 3: Full training ─────────────────────────────────────────
    if 3 in steps:
        log(f"Step 3/4: Full training on {fpn_level}")
        rc = run(
            [
                sys.executable,
                "train_fpn.py",
                "--roots",
                args.seadronessee,
                args.smd,
                "--fpn-level",
                fpn_level,
                "--latent-dim",
                str(args.latent_dim),
                "--hyper-channels",
                str(args.hyper_channels),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--output",
                f"checkpoints/fpn_compress/{fpn_level}",
                "--warmup-steps",
                "3000",
                "--rd-steps",
                "15000",
                "--temporal-steps",
                "20000",
            ],
            "Full compressor + predictor training",
        )
        if rc != 0:
            print("  Training failed — check logs above")
            sys.exit(1)

    # ── Step 4: Validation ────────────────────────────────────────────
    if 4 in steps:
        ckpt_path = SPRINT_DIR / "checkpoints" / "fpn_compress" / fpn_level / "best.pt"
        if not ckpt_path.exists():
            log(f"No checkpoint found at {ckpt_path}, skipping validation")
        else:
            log(f"Step 4/4: Validating {ckpt_path}")
            rc = run(
                [
                    sys.executable,
                    "validate.py",
                    "--mode",
                    "all",
                    "--ckpt",
                    str(ckpt_path),
                    "--fpn-level",
                    fpn_level,
                    "--feature-dir",
                    args.feature_dir,
                    "--coco-dir",
                    args.coco_dir,
                    "--device",
                    args.device,
                ],
                "Full validation (recon + mAP + probe)",
            )

    elapsed_h = (time.time() - overall_t0) / 3600
    log(f"Sprint complete! ({elapsed_h:.1f}h total)")
    print(f"\n  Recommended FPN level: {fpn_level}")
    print(f"  Checkpoints: checkpoints/fpn_compress/{fpn_level}/")
    print(f"  Results:     {ckpt_path}.eval.json" if "ckpt_path" in dir() else "")


if __name__ == "__main__":
    main()
