"""
SVC Dual-Layer Training Script.

Trains the scalable video coding layers on top of a trained LeWM-VC base codec.

Architecture:
  Encoder → LatentSplitter → [BL (64ch, 4-bit), EL (128ch, 8-bit)] → LatentFuser → Decoder

  BL (base layer): sent continuously to cloud → zero-padded to 192ch → decoded for AI pipeline
  EL (enhancement layer): stored on edge → fused with BL → decoded for human review

Two losses:
  1. Task loss:    MSE(frozen ResNet features of BL-only decode, original frame)
                   → forces BL to preserve detection/recognition features
  2. Recon loss:   MSE + LPIPS of BL+EL full decode vs original
                   → ensures fusion quality for human viewing

Training freezes encoder, predictor, and base decoder weights.
Fine-tunes: splitter projections, fusion layers, decoder for BL-only mode.

Usage:
    python scripts/train_svc.py --checkpoint path/to/best.pt --config configs/train_config.yaml
    python scripts/train_svc.py --checkpoint best.pt --lr 1e-4 --steps 15000
    python scripts/train_svc.py --checkpoint best.pt --val-clip datasets/virat/frames/VIRAT_S_000000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# SVC Training Engine
# ---------------------------------------------------------------------------


def build_base_models(config_path: str, checkpoint_path: str, device: str):
    """Load trained base codec and build SVC components on top."""
    import yaml

    config = yaml.safe_load(open(config_path))
    model_cfg = config.get("model", {})
    latent_dim = model_cfg.get("latent_dim", 192)
    patch_size = model_cfg.get("patch_size", 16)

    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer
    from lewm_vc.svc import LatentFuser, LatentSplitter, MultiRateQuantizer, SVCDecoder

    # Base models
    encoder = LeWMEncoder(
        latent_dim=latent_dim,
        patch_size=patch_size,
        hidden_dim=model_cfg.get("encoder", {}).get("hidden_dim", 192),
        num_layers=model_cfg.get("encoder", {}).get("num_layers", 6),
        num_heads=model_cfg.get("encoder", {}).get("num_heads", 3),
        semantic_surprise=True,
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
    entropy = HyperpriorEntropy(
        latent_dim=latent_dim,
        hyper_channels=model_cfg.get("entropy", {}).get("hyper_channels", 256),
        num_components=model_cfg.get("entropy", {}).get("num_components", 2),
    )
    quantizer = Quantizer()
    quantizer.set_mode("inference")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = ckpt.get("models", ckpt)
    for name, m in [
        ("encoder", encoder),
        ("predictor", predictor),
        ("decoder", decoder),
        ("entropy_model", entropy),
    ]:
        sd = model_state.get(name)
        if sd is not None:
            m.load_state_dict(sd, strict=False)
            print(f"  loaded {name}")

    # Freeze base models
    for m in [encoder, predictor, entropy, quantizer]:
        m.requires_grad_(False)
        m.eval()

    # SVC components
    base_dim = 64
    splitter = LatentSplitter(latent_dim=latent_dim, base_dim=base_dim, use_learned_split=True)
    fuser = LatentFuser(latent_dim=latent_dim, base_dim=base_dim, use_learned_fusion=True)
    svc_quant = MultiRateQuantizer(num_levels_bl=16, num_levels_el=256)
    svc_decoder = SVCDecoder(decoder, latent_dim=latent_dim, base_dim=base_dim, fuser=fuser)

    # Only splitter, fuser, and decoder are trainable
    models = {
        "encoder": encoder,
        "predictor": predictor,
        "decoder": decoder,
        "entropy": entropy,
        "quantizer": quantizer,
        "splitter": splitter,
        "fuser": fuser,
        "svc_decoder": svc_decoder,
        "svc_quant": svc_quant,
    }

    return models, latent_dim, base_dim


def get_dataloader(data_root: str, image_size: int = 256, batch_size: int = 8, num_frames: int = 1):
    """Simple frame loader from VIRAT clips."""
    from PIL import Image

    frames = sorted(Path(data_root).rglob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNGs found under {data_root}")
    print(f"  Found {len(frames)} frames")

    class _FrameLoader(torch.utils.data.Dataset):
        def __init__(self, paths, img_size):
            self.paths = paths
            self.img_size = img_size

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB").resize((self.img_size, self.img_size))
            t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
            return {"frames": t.unsqueeze(0), "path": str(self.paths[idx])}

    ds = _FrameLoader(frames, image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def compute_svc_loss(
    models: dict,
    batch: dict,
    device: str,
    task_probe: nn.Module,
    hpips_fn,
    lambda_task: float = 10.0,
    lambda_recon: float = 1.0,
) -> dict:
    """Compute SVC dual-layer loss for one batch.

    Loss = lambda_task * task_loss(BL-only decode, original)
         + lambda_recon * recon_loss(BL+EL full decode, original)
    """
    frames = batch["frames"].to(device)  # [B, 1, 3, H, W]
    if frames.dim() == 5:
        frames = frames[:, 0]  # [B, 3, H, W]

    enc = models["encoder"]
    splitter = models["splitter"]
    svc_quant = models["svc_quant"]
    svc_dec = models["svc_decoder"]
    dec = models["decoder"]

    with torch.no_grad():
        latent, _ = enc(frames, return_surprise=True)

    # Split
    bl, el = splitter(latent)

    # Quantize
    bl_q, el_q = svc_quant(bl, el)

    # Decode BL-only (zero-padded)
    recon_bl = svc_dec.decode_bl(bl_q)

    # Decode BL+EL (full fusion)
    recon_full = svc_dec.decode_full(bl_q, el_q)

    # Task loss: feature-space MSE on BL-only decode
    task_loss = F.mse_loss(task_probe(recon_bl), task_probe(frames))

    # Reconstruction loss: MSE + LPIPS on full decode
    mse = F.mse_loss(recon_full, frames)
    lpips = hpips_fn(recon_full, frames).mean() if hpips_fn is not None else 0.0
    recon_loss = mse + 0.3 * lpips

    # Total
    total = lambda_task * task_loss + lambda_recon * recon_loss

    # Stats
    bl_bpp = bl_q.numel() * 4 / (frames.shape[2] * frames.shape[3])  # 4-bit BL
    el_bpp = (el_q.numel() * 8) / (frames.shape[2] * frames.shape[3]) if el_q is not None else 0

    return {
        "total_loss": total,
        "task_loss": task_loss.item(),
        "recon_loss": recon_loss.item(),
        "mse": mse.item(),
        "lpips": lpips.item() if isinstance(lpips, torch.Tensor) else lpips,
        "bl_bpp": bl_bpp,
        "el_bpp": el_bpp,
        "recon_bl_psnr": 10 * np.log10(1.0 / max(mse.item(), 1e-10)),
    }


def train_svc(args):
    device = args.device
    print(f"Building models from {args.checkpoint}")
    models, latent_dim, base_dim = build_base_models(args.config, args.checkpoint, device)

    for name in ["splitter", "fuser", "decoder"]:
        models[name].train()
        models[name].to(device)

    # Optimizer
    trainable = (
        list(models["splitter"].parameters())
        + list(models["fuser"].parameters())
        + list(models["decoder"].parameters())
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    # Scheduler
    total_steps = args.steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Data
    loader = get_dataloader(args.data, args.image_size, args.batch_size)

    # Task probe (frozen ResNet-18 feature extractor)
    from lewm_vc.utils.task_probe import create_task_probe

    task_probe = create_task_probe("resnet18", multi_scale=True, device=device)
    task_probe.eval()

    # LPIPS
    hpips_fn = _make_lpips(device)

    # Output dir
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nSVC Training: {total_steps} steps, lr={args.lr}, lambda_task={args.lambda_task}")
    print(
        f"  Trainable: splitter ({sum(p.numel() for p in models['splitter'].parameters())} params), "
        f"fuser ({sum(p.numel() for p in models['fuser'].parameters())} params), "
        f"decoder ({sum(p.numel() for p in models['decoder'].parameters())} params)"
    )

    step = 0
    best_task_loss = float("inf")
    epoch = 0
    while step < total_steps:
        epoch += 1
        for batch in loader:
            if step >= total_steps:
                break

            losses = compute_svc_loss(
                models,
                batch,
                device,
                task_probe,
                hpips_fn,
                lambda_task=args.lambda_task,
                lambda_recon=args.lambda_recon,
            )

            optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 100 == 0:
                print(
                    f"  step={step:5d}  loss={losses['total_loss'].item():.4f}  "
                    f"task={losses['task_loss']:.6f}  recon={losses['recon_loss']:.4f}  "
                    f"BL_PSNR={losses['recon_bl_psnr']:.1f}  BL_bpp={losses['bl_bpp']:.4f}"
                )

            if step % 500 == 0:
                # Save checkpoint
                ckpt = {
                    "step": step,
                    "splitter": models["splitter"].state_dict(),
                    "fuser": models["fuser"].state_dict(),
                    "decoder": models["decoder"].state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "task_loss": losses["task_loss"],
                }
                torch.save(ckpt, out / f"svc_step_{step}.pt")

                if losses["task_loss"] < best_task_loss:
                    best_task_loss = losses["task_loss"]
                    torch.save(ckpt, out / "svc_best.pt")
                    print(f"    -> NEW BEST (task_loss={best_task_loss:.6f})")

    print(f"\nSVC training complete. Best saved to {out / 'svc_best.pt'}")
    print(f"  Final BL PSNR: {losses['recon_bl_psnr']:.1f} dB")
    print(f"  BL BPP: {losses['bl_bpp']:.4f}")
    print(f"  Task loss: {best_task_loss:.6f}")


# ---------------------------------------------------------------------------
# LPIPS helper
# ---------------------------------------------------------------------------


def _make_lpips(device):
    """Load LPIPS perceptual loss (VGG-16 based)."""
    try:
        import lpips

        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device)
        lpips_fn.eval()
        print("  LPIPS perceptual loss enabled")
        return lpips_fn
    except Exception:
        print("  [warning] LPIPS not available, using MSE only")
        return lambda x, y: torch.tensor(0.0, device=device)


# ---------------------------------------------------------------------------
# Evaluation (quick validation on a single clip)
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(models, clip_path: str, device: str, max_frames: int = 50):
    """Quick validation: encode clip, measure BL and full PSNR."""
    from PIL import Image

    frames = sorted(Path(clip_path).glob("*.png"))[:max_frames]
    print(f"\nValidating on {len(frames)} frames from {clip_path}")

    psnr_bl_list, psnr_full_list, bl_bpp_list = [], [], []

    for fp in frames:
        img = (
            torch.from_numpy(np.array(Image.open(fp).resize((256, 256)).convert("RGB")))
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
            / 255.0
        )
        latent, _ = models["encoder"](img, return_surprise=True)
        bl, el = models["splitter"](latent)
        bl_q, el_q = models["svc_quant"](bl, el)
        recon_bl = models["svc_decoder"].decode_bl(bl_q)
        recon_full = models["svc_decoder"].decode_full(bl_q, el_q)

        mse_bl = F.mse_loss(recon_bl, img).item()
        mse_full = F.mse_loss(recon_full, img).item()
        psnr_bl_list.append(10 * np.log10(1.0 / max(mse_bl, 1e-10)))
        psnr_full_list.append(10 * np.log10(1.0 / max(mse_full, 1e-10)))
        bl_bpp_list.append(bl_q.numel() * 4 / (256 * 256))

    print(f"  BL-only:       PSNR={np.mean(psnr_bl_list):.1f} dB  BPP={np.mean(bl_bpp_list):.4f}")
    print(f"  BL+EL (full):  PSNR={np.mean(psnr_full_list):.1f} dB")
    return {"bl_psnr": np.mean(psnr_bl_list), "full_psnr": np.mean(psnr_full_list)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train SVC dual-layer codec")
    parser.add_argument("--checkpoint", required=True, help="Base codec best.pt")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--data", default="datasets/virat/frames", help="Training frames")
    parser.add_argument("--output", default="svc_checkpoints")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-task", type=float, default=10.0, help="Task loss weight")
    parser.add_argument(
        "--lambda-recon", type=float, default=1.0, help="Reconstruction loss weight"
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validate", default=None, help="Validate on a clip after training")
    args = parser.parse_args()

    train_svc(args)

    if args.validate:
        models, _, _ = build_base_models(args.config, args.checkpoint, args.device)
        # Load trained SVC weights
        best_path = Path(args.output) / "svc_best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=args.device)
            models["splitter"].load_state_dict(ckpt["splitter"])
            models["fuser"].load_state_dict(ckpt["fuser"])
            models["decoder"].load_state_dict(ckpt["decoder"])
            print(f"Loaded SVC weights from {best_path}")
        validate(models, args.validate, args.device)


if __name__ == "__main__":
    main()
