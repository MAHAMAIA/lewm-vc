#!/usr/bin/env python3
"""
Lightweight eval monitor — watches a training run's checkpoint directory and
runs quick temporal IPPP evaluation every N steps.

Usage:
    # Monitor active training run (auto-detects latest checkpoint dir)
    python scripts/eval_monitor.py --run-dir checkpoints/sentinel-p0-l60.0-13aacf

    # Specify validation data (single clip or root dir)
    python scripts/eval_monitor.py --run-dir checkpoints/sentinel-p0-l60.0-13aacf \
        --data datasets/virat/frames/VIRAT_S_000000

    # Manually evaluate a specific checkpoint
    python scripts/eval_monitor.py --checkpoint path/to/step_5000.pt \
        --data datasets/virat/frames/VIRAT_S_000000

Output:
    - eval_log.json:   Cumulative evaluation results
    - TensorBoard:     Written to the run's TB log dir (runs/lambda_<val>/) if available
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import lpips as lpips_lib

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


def load_model(checkpoint_path, config_path, device):
    import yaml

    config = yaml.safe_load(open(config_path))
    model_cfg = config.get("model", {})
    latent_dim = model_cfg.get("latent_dim", 192)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer

    encoder = LeWMEncoder(
        latent_dim=latent_dim,
        patch_size=model_cfg.get("patch_size", 16),
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
    quantizer.set_mode("inference")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = ckpt.get("models", ckpt)
    for name, m in [
        ("encoder", encoder),
        ("predictor", predictor),
        ("decoder", decoder),
        ("entropy_model", entropy_model),
    ]:
        sd = model_state.get(name)
        if sd is not None:
            m.load_state_dict(sd, strict=False)

    for m in [encoder, predictor, decoder, entropy_model, quantizer]:
        m.to(device)
        m.eval()

    return encoder, predictor, decoder, entropy_model, quantizer


def discover_frames(data_path, max_frames=32):
    p = Path(data_path)
    if not p.exists():
        print(f"  [error] data path not found: {data_path}")
        return []
    if p.is_dir():
        subdirs = sorted([d for d in p.iterdir() if d.is_dir()])
        if subdirs:
            clip_dir = subdirs[0]
        else:
            clip_dir = p
        all_frames = sorted(clip_dir.rglob("*.png"))
    else:
        all_frames = [p]
    if max_frames and len(all_frames) > max_frames:
        all_frames = all_frames[:max_frames]
    return all_frames


def psnr(mse):
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


@torch.no_grad()
def evaluate_checkpoint(models, frame_paths, device, image_size, lpips_fn, task_probe=None):
    encoder, predictor, decoder, entropy_model, quantizer = models
    recon_latents = []
    context_len = getattr(predictor, "context_len", 4)

    from PIL import Image

    results = []
    for i, fpath in enumerate(frame_paths):
        img = Image.open(fpath).convert("RGB").resize((image_size, image_size))
        tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        latent, _ = encoder(tensor, return_surprise=True)

        if i == 0:
            quant_latent = quantizer(latent)
            recon = decoder(quant_latent)
            rate, _ = entropy_model(quant_latent)
            frame_type = "I"
            recon_latents.append(quant_latent)
        else:
            ctx = recon_latents[max(0, len(recon_latents) - context_len) :]
            pred_mean, _ = predictor(ctx)
            residual = latent - pred_mean
            quant_residual = quantizer(residual)
            rate, _ = entropy_model(quant_residual)
            recon_latent = pred_mean + quant_residual
            recon = decoder(recon_latent)
            frame_type = "P"
            recon_latents.append(recon_latent)

        h, w = tensor.shape[2], tensor.shape[3]
        bpp = rate.sum().item() / (h * w)
        mse_val = F.mse_loss(recon, tensor).item()
        lpips_val = lpips_fn(recon, tensor).item() if lpips_fn else 0.0
        task_val = F.mse_loss(task_probe(recon), task_probe(tensor)).item() if task_probe else 0.0

        results.append(
            {
                "frame": fpath.name,
                "type": frame_type,
                "psnr": round(psnr(mse_val), 2),
                "bpp": round(bpp, 6),
                "mse": round(mse_val, 8),
                "lpips": round(lpips_val, 4),
                "task_loss": round(task_val, 6),
            }
        )

    # Aggregate
    count = len(results)
    i_frames = [r for r in results if r["type"] == "I"]
    p_frames = [r for r in results if r["type"] == "P"]
    i_bpp = sum(r["bpp"] for r in i_frames) / max(1, len(i_frames))
    p_bpp = sum(r["bpp"] for r in p_frames) / max(1, len(p_frames))
    avg_bpp = sum(r["bpp"] for r in results) / count
    avg_mse = sum(r["mse"] for r in results) / count
    avg_lpips = sum(r["lpips"] for r in results) / count
    avg_task = sum(r["task_loss"] for r in results) / count

    summary = {
        "num_frames": count,
        "psnr": round(psnr(avg_mse), 2),
        "psnr_i": round(sum(r["psnr"] for r in i_frames) / max(1, len(i_frames)), 2),
        "psnr_p": round(sum(r["psnr"] for r in p_frames) / max(1, len(p_frames)), 2),
        "bpp": round(avg_bpp, 6),
        "i_bpp": round(i_bpp, 6),
        "p_bpp": round(p_bpp, 6),
        "p_i_ratio": round(p_bpp / i_bpp, 3) if i_bpp > 0 else 0,
        "temporal_gain_pct": round((1.0 - p_bpp / i_bpp) * 100, 1) if i_bpp > 0 else 0,
        "lpips": round(avg_lpips, 4),
        "task_loss": round(avg_task, 6),
        "mse": round(avg_mse, 8),
    }
    return summary, results


def write_tensorboard(log_dir, step, summary, tag_prefix="eval"):
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))
        for key, val in summary.items():
            if isinstance(val, (int, float)):
                writer.add_scalar(f"{tag_prefix}/{key}", val, step)
        writer.close()
    except Exception as e:
        print(f"  [tb] write failed: {e}")


def load_eval_log(log_path):
    if log_path.exists():
        try:
            return json.loads(log_path.read_text())
        except Exception:
            return []
    return []


def save_eval_log(log_path, entries):
    log_path.write_text(json.dumps(entries, indent=2))


def find_step(checkpoint_path):
    name = Path(checkpoint_path).stem
    if name.startswith("step_"):
        return int(name.split("_")[1])
    return 0


def find_latest_checkpoint(checkpoint_dir, exclude_steps=None):
    if exclude_steps is None:
        exclude_steps = set()
    ckpt_dir = Path(checkpoint_dir)
    # Check both top-level and lambda subdirectories
    candidates = []
    for d in [ckpt_dir] + sorted(ckpt_dir.iterdir()):
        if d.is_dir():
            candidates.extend(d.glob("step_*.pt"))
    if not candidates:
        # Also check step_*.pt in the checkpoint dir itself
        candidates = list(ckpt_dir.glob("step_*.pt"))
    candidates = [c for c in candidates if find_step(c) not in exclude_steps]
    candidates.sort(key=lambda p: find_step(p))
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="LeWM-VC eval monitor")
    parser.add_argument(
        "--run-dir",
        help="Training run directory to watch (e.g. checkpoints/sentinel-p0-l60.0-XXXX)",
    )
    parser.add_argument("--checkpoint", help="Single checkpoint to evaluate (one-shot)")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument(
        "--data",
        default="datasets/virat/frames",
        help="Validation data: clip dir or root dir (uses first clip)",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=32, help="Frames per eval")
    parser.add_argument("--interval", type=int, default=120, help="Poll interval (seconds)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tb-dir", help="TensorBoard log dir (default: auto-detect from run)")
    parser.add_argument("--skip-lpips", action="store_true", help="Skip LPIPS (faster)")
    args = parser.parse_args()

    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[eval_monitor] device={device}")

    # Resolve data
    frame_paths = discover_frames(args.data, args.num_frames)
    if not frame_paths:
        print(f"[error] no frames found in {args.data}")
        sys.exit(1)
    print(f"[eval_monitor] eval data: {len(frame_paths)} frames from {args.data}")

    # Init LPIPS
    lpips_fn = None
    if not args.skip_lpips and HAS_LPIPS:
        lpips_fn = lpips_lib.LPIPS(net="vgg", verbose=False).to(device).eval()
        print(f"[eval_monitor] LPIPS enabled")

    # Task probe
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from lewm_vc.utils.task_probe import create_task_probe

        task_probe = create_task_probe("resnet18").to(device).eval()
        print(f"[eval_monitor] task_probe enabled")
    except Exception:
        task_probe = None
        print(f"[eval_monitor] task_probe disabled")

    # Single checkpoint mode
    if args.checkpoint:
        print(f"\n[eval_monitor] Evaluating single checkpoint: {args.checkpoint}")
        models = load_model(args.checkpoint, args.config, device)
        summary, _ = evaluate_checkpoint(
            models, frame_paths, device, args.image_size, lpips_fn, task_probe
        )
        print("\n=== Results ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return

    # Watch mode
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[error] run dir not found: {run_dir}")
        sys.exit(1)

    log_path = run_dir / "eval_log.json"
    eval_log = load_eval_log(log_path)
    evaluated_steps = {e.get("step", 0) for e in eval_log}
    print(f"[eval_monitor] watching {run_dir} (already evaluated: {sorted(evaluated_steps)})")
    print(f"[eval_monitor] poll interval: {args.interval}s")

    # TensorBoard log dir
    tb_dir = args.tb_dir
    if tb_dir is None:
        # Auto-detect: lambda value from run name, then runs/lambda_<val>/
        lambda_val = None
        for part in run_dir.name.split("-"):
            if part.startswith("l") and part[1:].replace(".", "").isdigit():
                lambda_val = part[1:]
                break
        if lambda_val:
            tb_dir = str(Path("runs") / f"lambda_{lambda_val}")
    if tb_dir:
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        print(f"[eval_monitor] TensorBoard: {tb_dir}")

    while True:
        try:
            ckpt = find_latest_checkpoint(run_dir, exclude_steps=evaluated_steps)
            if ckpt is None:
                print(
                    f"  [poll] no new checkpoints (eval'd steps: {sorted(evaluated_steps)}), sleeping {args.interval}s"
                )
                time.sleep(args.interval)
                continue

            step = find_step(ckpt)
            print(f"\n>>> Evaluating checkpoint: {ckpt} (step {step})")
            t0 = time.time()

            models = load_model(str(ckpt), args.config, device)
            summary, _ = evaluate_checkpoint(
                models, frame_paths, device, args.image_size, lpips_fn, task_probe
            )
            elapsed = time.time() - t0

            # Log
            entry = {"step": step, "checkpoint": str(ckpt), "elapsed_s": round(elapsed, 1)}
            entry.update(summary)
            eval_log.append(entry)
            save_eval_log(log_path, eval_log)
            evaluated_steps.add(step)

            print(f"  done in {elapsed:.1f}s")
            print(
                f"  PSNR={summary['psnr']} dB | BPP={summary['bpp']} | I={summary['i_bpp']} P={summary['p_bpp']} ratio={summary['p_i_ratio']}"
            )
            if lpips_fn:
                print(f"  LPIPS={summary['lpips']} | task_loss={summary['task_loss']}")

            # TensorBoard
            if tb_dir:
                write_tensorboard(tb_dir, step, summary)

        except KeyboardInterrupt:
            print("\n[eval_monitor] stopped")
            break
        except Exception as e:
            import traceback

            print(f"  [error] {e}")
            traceback.print_exc()
            print(f"  sleeping {args.interval}s before retry...")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
