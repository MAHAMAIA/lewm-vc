#!/usr/bin/env python3
"""
Launch parallel multi-lambda training runs.

Reads the lambda schedule from config and spawns N concurrent training
processes, each at a different RD lambda value.

On MI300X (192GB HBM3): max 3 concurrent for 14.7M-param model.
On consumer GPUs (24GB): 1-2 concurrent runs, or sequential.

Usage:
    python scripts/train_all_lambdas.py                                         # all lambdas, auto concurrency
    python scripts/train_all_lambdas.py --config configs/train_config.yaml
    python scripts/train_all_lambdas.py --lambdas 0.02 0.05 0.08               # override lambda values
    python scripts/train_all_lambdas.py --parallel 3                            # force 3 concurrent
    python scripts/train_all_lambdas.py --sequential                            # one at a time
    python scripts/train_all_lambdas.py --gpu-ids 0 1 2 3                       # assign specific GPUs
    python scripts/train_all_lambdas.py --dry-run                               # show what would run
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def detect_gpu_count() -> int:
    """Detect available GPUs via nvidia-smi or rocm-smi."""
    for cmd in ["nvidia-smi", "rocm-smi"]:
        try:
            result = subprocess.run(
                [cmd, "--list-gpus"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                lines = [line for line in result.stdout.strip().split("\n") if line]
                return len(lines)
        except FileNotFoundError:
            continue
    return 0


def estimate_concurrency(gpu_count: int, total_gpu_memory_gb: int | None = None) -> int:
    """
    Estimate how many concurrent training runs a single GPU can handle.

    Heuristics:
      - MI300X (192GB): 3-4 concurrent runs
      - A100 80GB: 1-2 concurrent
      - Consumer (24GB): 1 at a time
    """
    if total_gpu_memory_gb is None:
        # Try to detect
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Total VRAM" in line or "Memory Size" in line:
                        parts = line.split()
                        for p in parts:
                            if "GB" in p:
                                total_gpu_memory_gb = float(p.replace("GB", ""))
                                break
        except FileNotFoundError:
            pass

        if total_gpu_memory_gb is None:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    mem_mb = [
                        int(line.strip()) for line in result.stdout.strip().split("\n") if line
                    ]
                    if mem_mb:
                        total_gpu_memory_gb = mem_mb[0] / 1024
            except FileNotFoundError:
                pass

    if total_gpu_memory_gb is None:
        return 1  # conservative default

    if total_gpu_memory_gb >= 128:
        return 3
    elif total_gpu_memory_gb >= 64:
        return 2
    else:
        return 1


def main():
    parser = argparse.ArgumentParser(description="Launch multi-lambda training")
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Training config path",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="*",
        default=None,
        help="Override lambda values (default: from config)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Max concurrent runs (default: auto-detect based on GPU memory)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run lambdas one at a time",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="*",
        default=None,
        help="Assign specific GPU IDs to each run (e.g. --gpu-ids 0 1 2 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[0, 1, 2, 3, 4],
        help="Training phase to run (default: 1 — joint RD)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing runs (finds latest checkpoints)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Lambda values
    lambdas = args.lambdas or config.get("lambda_schedule", {}).get("values", [0.05])
    lambdas = sorted(set(lambdas))
    print(f"Lambda schedule: {lambdas}")

    stagger_seconds = 45

    # Detect concurrency
    gpu_count = detect_gpu_count()
    print(f"GPUs detected: {gpu_count}")

    if args.sequential:
        max_concurrent = 1
    elif args.parallel is not None:
        max_concurrent = args.parallel
    elif args.gpu_ids is not None:
        max_concurrent = len(args.gpu_ids)
    elif gpu_count >= 1:
        runs_per_gpu = estimate_concurrency(gpu_count)
        max_concurrent = gpu_count * runs_per_gpu
    else:
        max_concurrent = 1

    max_concurrent = min(max_concurrent, len(lambdas))
    print(f"Max concurrent: {max_concurrent}")

    if max_concurrent <= 0:
        print("Error: no concurrency (no GPUs detected, no --gpu-ids)")
        sys.exit(1)

    # Determine CUDA_VISIBLE_DEVICES per run
    gpu_assignment = args.gpu_ids or list(range(gpu_count)) if gpu_count > 0 else [0]

    # Build commands
    checkpoint_dir = Path(config.get("checkpoint", {}).get("dir", "checkpoints"))

    commands = []
    for i, lam in enumerate(lambdas):
        gpu_id = gpu_assignment[i % len(gpu_assignment)]
        lambda_cp_dir = checkpoint_dir / f"lambda_{lam}"

        cmd = [
            sys.executable,
            "-m",
            "src.scripts.train",
            "--config",
            args.config,
            "--phase",
            str(args.phase),
            "--lambda",
            str(lam),
        ]

        # Resume: find latest checkpoint for this lambda
        if args.resume and lambda_cp_dir.exists():
            checkpoints = sorted(lambda_cp_dir.glob("*.pt"))
            if checkpoints:
                latest = str(checkpoints[-1])
                cmd.extend(["--resume", latest])
                print(f"  lambda={lam}: resuming from {latest}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        commands.append(
            {
                "lambda": lam,
                "gpu": gpu_id,
                "cmd": cmd,
                "env": env,
                "checkpoint_dir": lambda_cp_dir,
            }
        )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Running {len(commands)} training jobs ({max_concurrent} concurrent)")
    for c in commands:
        print(f"  lambda={c['lambda']:.4f}  GPU={c['gpu']}  {' '.join(c['cmd'][:5])} ...")
    print(f"{'=' * 60}\n")

    if args.dry_run:
        print("Dry run — exiting")
        return

    # Launch with job control
    running: list[dict] = []
    completed = 0
    failed = 0

    while commands or running:
        # Fill available slots
        while commands and len(running) < max_concurrent:
            job = commands.pop(0)
            print(f"[launch] lambda={job['lambda']:.4f} GPU={job['gpu']}")
            if running:
                print(f"  staggering {stagger_seconds}s after previous launch")
                time.sleep(stagger_seconds)
            proc = subprocess.Popen(
                job["cmd"],
                env=job["env"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append({"proc": proc, **job})

        # Wait for any to finish
        if running:
            done = []
            for i, job in enumerate(running):
                ret = job["proc"].poll()
                if ret is not None:
                    stdout, _ = job["proc"].communicate()
                    if ret == 0:
                        completed += 1
                        status = "OK"
                    else:
                        failed += 1
                        status = f"FAIL (code {ret})"
                        # Print last 10 lines on failure
                        if stdout:
                            lines = stdout.strip().split("\n")
                            for line in lines[-10:]:
                                print(f"  [{job['lambda']:.4f}] {line}")
                    print(
                        f"[done] lambda={job['lambda']:.4f} GPU={job['gpu']} {status} "
                        f"({completed + failed}/{len(lambdas)})"
                    )
                    done.append(i)

            for i in reversed(done):
                running.pop(i)

            if not done:
                time.sleep(5)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"All jobs finished: {completed} OK, {failed} failed")
    print(f"Checkpoints: {checkpoint_dir}/lambda_*/")
    if failed > 0:
        print("Review logs in individual job output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
