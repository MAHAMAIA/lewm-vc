"""
Production training wrapper for Sentinel (LeWM-VC).

Wraps src/scripts/train.py with:
  - W&B experiment tracking
  - Run manifest (config hash, git hash, data snapshot)
  - Best checkpoint by validation loss
  - Graceful signal handling

Usage:
    python scripts/train_production.py --config configs/train_config.yaml --phase 0 --wandb
    python scripts/train_production.py --config configs/train_config.yaml --resume <run_id>
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path (for imports like "from src.scripts.train...")
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import yaml


# ---------------------------------------------------------------------------
# W&B optional import
# ---------------------------------------------------------------------------
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        return len(result.stdout.strip()) > 0
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Dataset hash (sha256 of sorted video dir listings)
# ---------------------------------------------------------------------------
def compute_dataset_hash(roots: list[str]) -> dict:
    """Compute a content-addressed hash of the dataset."""
    entries = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for video_dir in sorted(root_path.iterdir()):
            if video_dir.is_dir():
                for frame in sorted(video_dir.iterdir()):
                    if frame.suffix in (".png", ".jpg", ".jpeg"):
                        entries.append(str(frame.relative_to(root_path.parent)))
    hasher = hashlib.sha256()
    for e in entries:
        hasher.update(e.encode())
    return {
        "roots": roots,
        "num_videos": len(set(e.split("/")[1] for e in entries) if entries else []),
        "num_frames": len(entries),
        "sha256": hasher.hexdigest() if entries else "empty",
    }


# ---------------------------------------------------------------------------
# Run ID generation
# ---------------------------------------------------------------------------
def generate_run_id(phase: int, lambda_val: float) -> str:
    short_hash = hashlib.sha256(os.urandom(8)).hexdigest()[:6]
    return f"sentinel-p{phase}-l{lambda_val}-{short_hash}"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
MANIFEST_FILENAME = "manifest.json"


def write_manifest(checkpoint_dir: Path, data: dict):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / MANIFEST_FILENAME
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_manifest(checkpoint_dir: Path) -> dict | None:
    path = checkpoint_dir / MANIFEST_FILENAME
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def update_manifest(checkpoint_dir: Path, updates: dict):
    manifest = load_manifest(checkpoint_dir) or {}
    manifest.update(updates)
    write_manifest(checkpoint_dir, manifest)


# ---------------------------------------------------------------------------
# Signal handler for graceful shutdown
# ---------------------------------------------------------------------------
_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    if _interrupted:
        print("\n  Force exit.", flush=True)
        sys.exit(1)
    print(f"\n  Signal {signum} received, saving checkpoint...", flush=True)
    _interrupted = True


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGHUP, _signal_handler)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Sentinel (production wrapper)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda, cpu)")
    parser.add_argument(
        "--lambda",
        type=float,
        dest="lambda_val",
        default=0.05,
        help="RD lambda value",
    )
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3, 4], help="Phase")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint dir or path",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B experiment tracking",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run ID (auto-generated if not set)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Resolve resume
    resume_path = None
    if args.resume:
        resume_candidate = Path(args.resume)
        if resume_candidate.is_dir():
            manifest = load_manifest(resume_candidate)
            if manifest:
                print(f"  Found manifest: {manifest.get('run_id', 'unknown')}")
            # Pick most recent step checkpoint
            ckpts = sorted(resume_candidate.glob("step_*.pt"))
            if ckpts:
                resume_path = str(ckpts[-1])
                print(f"  Resuming from: {resume_path}")
            else:
                print("  No step checkpoints found in dir, starting fresh")
        elif resume_candidate.exists():
            resume_path = args.resume
        else:
            print(f"  Resume path not found: {args.resume}")
            sys.exit(1)

    # Generate run ID and checkpoint dir
    run_id = args.run_id or generate_run_id(args.phase, args.lambda_val)
    ckpt_base = Path(config.get("checkpoint", {}).get("dir", "checkpoints"))
    checkpoint_dir = ckpt_base / run_id
    config.setdefault("checkpoint", {})["dir"] = str(checkpoint_dir)

    print(f"[run] {run_id}")
    print(f"[checkpoints] {checkpoint_dir}")

    # W&B init
    wandb_run = None
    if args.wandb:
        if not HAS_WANDB:
            print("  [wandb] not installed (pip install wandb). Skipping.")
        elif not os.environ.get("WANDB_API_KEY"):
            print("  [wandb] WANDB_API_KEY not set. Skipping.")
        else:
            wandb_cfg = config.get("wandb", {})
            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "sentinel"),
                entity=wandb_cfg.get("entity"),
                name=run_id,
                config=config,
                id=run_id,
                resume="allow",
            )
            print(f"  [wandb] {wandb_run.url}")

    # Compute dataset hash
    data_roots = config.get("data", {}).get("roots", [])
    abs_roots = []
    for r in data_roots:
        p = Path(r)
        abs_root = str(p.absolute()) if not p.is_absolute() else r
        abs_roots.append(abs_root)
    data_info = compute_dataset_hash(abs_roots)
    if data_info["num_frames"] == 0:
        print("  [warning] no frames found in dataset roots:", abs_roots)
        for r in abs_roots:
            print(f"    {r} exists={Path(r).exists()}")

    # Write manifest
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "phase": args.phase,
        "lambda_rd": args.lambda_val,
        "config_file": os.path.abspath(args.config),
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12],
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "data": data_info,
        "wandb_run": wandb_run.url if wandb_run else None,
        "wandb_run_id": wandb_run.id if wandb_run else None,
        "best_checkpoint": None,
        "interrupted": False,
    }
    manifest_path = write_manifest(checkpoint_dir, manifest)
    print(f"[manifest] {manifest_path}")

    # Patch LeWMTrainer for W&B + best-checkpoint tracking
    # (must import here so the module is patched before train() is called)
    from src.scripts.train import LeWMTrainer, train as lewm_train

    _original_log_metrics = LeWMTrainer.log_metrics
    _original_save_checkpoint = LeWMTrainer.save_checkpoint

    def _patched_log_metrics(self, metrics, step):
        _original_log_metrics(self, metrics, step)
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)
        if "val/total_loss" in metrics:
            self._last_val_loss = metrics["val/total_loss"]

    def _patched_save_checkpoint(self, name, optimizer=None):
        path = _original_save_checkpoint(self, name, optimizer)
        val_loss = getattr(self, "_last_val_loss", None)
        best_loss = getattr(self, "_best_val_loss", float("inf"))
        if val_loss is not None and val_loss < best_loss:
            self._best_val_loss = val_loss
            best_path = self.checkpoint_dir / "best.pt"
            shutil.copy(path, best_path)
            print(f"    -> NEW BEST (val_loss={val_loss:.4f}) saved to {best_path.name}")
            # Update manifest with best checkpoint path
            manifest_path = self.checkpoint_dir.parent / "manifest.json"
            if manifest_path.exists():
                try:
                    m = json.loads(manifest_path.read_text())
                    m["best_checkpoint"] = str(best_path)
                    manifest_path.write_text(json.dumps(m, indent=2))
                except Exception:
                    pass
        return path

    LeWMTrainer.log_metrics = _patched_log_metrics
    LeWMTrainer.save_checkpoint = _patched_save_checkpoint

    # Build models and train
    # (duplicated from train.py's __main__ to avoid import coupling)
    model_cfg = config.get("model", {})
    latent_dim = model_cfg.get("latent_dim", 192)
    patch_size = model_cfg.get("patch_size", 16)

    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer
    from lewm_vc.utils import RateController

    encoder = LeWMEncoder(
        latent_dim=latent_dim,
        patch_size=patch_size,
        hidden_dim=model_cfg.get("encoder", {}).get("hidden_dim", 192),
        num_layers=model_cfg.get("encoder", {}).get("num_layers", 6),
        num_heads=model_cfg.get("encoder", {}).get("num_heads", 3),
        semantic_surprise=model_cfg.get("encoder", {}).get("semantic_surprise", True),
    )
    predictor = LeWMPredictor(
        latent_dim=latent_dim,
        hidden_dim=model_cfg.get("predictor", {}).get("hidden_dim", 256),
        num_layers=model_cfg.get("predictor", {}).get("num_layers", 8),
        num_heads=model_cfg.get("predictor", {}).get("num_heads", 4),
        context_len=model_cfg.get("predictor", {}).get("context_len", 4),
    )
    decoder = LeWMDecoder(
        latent_dim=latent_dim,
        hidden_dim=model_cfg.get("decoder", {}).get("hidden_dim", 512),
    )
    entropy_model = HyperpriorEntropy(
        latent_dim=latent_dim,
        hyper_channels=model_cfg.get("entropy", {}).get("hyper_channels", 256),
        num_components=model_cfg.get("entropy", {}).get("num_components", 2),
    )
    quantizer = Quantizer()
    rate_controller = RateController(latent_dim=latent_dim)

    device = args.device
    for m in [encoder, predictor, decoder, entropy_model, quantizer, rate_controller]:
        m.to(device)

    # Run training
    try:
        lewm_train(
            encoder=encoder,
            predictor=predictor,
            decoder=decoder,
            entropy_model=entropy_model,
            quantizer=quantizer,
            rate_controller=rate_controller,
            config=config,
            device=device,
            lambda_val=args.lambda_val,
            resume_from=resume_path,
            start_phase=args.phase,
        )
    except KeyboardInterrupt:
        print("\n  Training interrupted by user.")
        manifest["interrupted"] = True
    except SystemExit:
        raise
    except BaseException as e:
        print(f"\n  Training failed: {e}")
        manifest["interrupted"] = True
        raise

    # Post-training: find and record best checkpoint
    best_pt = checkpoint_dir / "best.pt"
    if best_pt.exists():
        manifest["best_checkpoint"] = str(best_pt)
        print(f"[best] {best_pt}")
    else:
        # Fallback: use final checkpoint
        final_pts = sorted(checkpoint_dir.glob("final_step_*.pt"))
        if final_pts:
            shutil.copy(final_pts[-1], best_pt)
            manifest["best_checkpoint"] = str(best_pt)
            print(f"[best] (final) {best_pt}")

    manifest["interrupted"] = _interrupted
    update_manifest(checkpoint_dir, manifest)

    # W&B artifact upload
    if wandb_run is not None and best_pt.exists():
        artifact = wandb.Artifact(f"model-{run_id}", type="model")
        artifact.add_file(str(best_pt))
        artifact.add_file(str(manifest_path))
        wandb_run.log_artifact(artifact)

    # Close W&B
    if wandb_run is not None:
        wandb_run.finish()

    print(f"\n{done}")
    print(f"  run_id:    {run_id}")
    print(f"  manifest:  {manifest_path}")
    print(f"  best:      {manifest.get('best_checkpoint', 'none')}")
    print(f"  wandb:     {'yes' if wandb_run else 'no'}")


if __name__ == "__main__":
    main()
