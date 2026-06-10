"""
Step 3: Full FPN Compressor + Temporal Predictor Training
Retrains the 331k compressor and JEPA predictor on the chosen FPN level.
Uses SIGReg + dual loss, targeting <10% mAP degradation.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from lewm_vc.data import FrameDataset, collate_sequences


# ---------------------------------------------------------------------------
# Rate loss — CDF-based NLL for quantized latents
# ---------------------------------------------------------------------------


def nll_rate_loss(latents, mu, sigma, step_size):
    half = step_size / 2.0
    prob = torch.special.ndtr((latents + half - mu) / sigma) - torch.special.ndtr(
        (latents - half - mu) / sigma
    )
    prob = prob.clamp(min=1e-10)
    nats = -torch.log(prob)
    bits = nats / 0.693147
    return bits.sum() / latents.shape[0]


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_models(
    latent_dim: int,
    hyper_channels: int,
    hidden_dim: int,
    predictor_layers: int,
    predictor_heads: int,
    context_len: int,
    fpn_level: str,
    device: str,
):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent))

    from fpn_backbone import ResNet50FPN
    from lewm_vc.feature_compress import FeatureCompressor, FeatureDecompressor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer, QuantMode
    from lewm_vc.predictor import LeWMPredictor

    backbone = ResNet50FPN().to(device).eval()

    compressor = FeatureCompressor(in_channels=256, latent_dim=latent_dim).to(device)
    decompressor = FeatureDecompressor(latent_dim=latent_dim, out_channels=256).to(device)
    entropy = HyperpriorEntropy(latent_dim=latent_dim, hyper_channels=hyper_channels).to(device)
    quantizer = Quantizer(mode=QuantMode.TRAINING).to(device)

    predictor = LeWMPredictor(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=predictor_layers,
        num_heads=predictor_heads,
        context_len=context_len,
    ).to(device)

    return {
        "backbone": backbone,
        "compressor": compressor,
        "decompressor": decompressor,
        "entropy": entropy,
        "quantizer": quantizer,
        "predictor": predictor,
    }


def count_params(models: dict, trainable_only: bool = True) -> int:
    total = 0
    for name, m in models.items():
        if name == "backbone":
            continue
        for p in m.parameters():
            if trainable_only and not p.requires_grad:
                continue
            total += p.numel()
    return total


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def create_loaders(
    roots: list[str],
    sequence_length: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment: bool,
):
    train_ds = FrameDataset.from_roots(
        roots,
        sequence_length=sequence_length,
        image_size=image_size,
        augment=augment,
        frame_stride=1,
        split="train",
    )
    val_ds = FrameDataset.from_roots(
        roots,
        sequence_length=sequence_length,
        image_size=image_size,
        augment=False,
        frame_stride=1,
        split="val",
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_sequences,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_sequences,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def extract_fpn(batch, backbone, fpn_level: str, device: str):
    frames = batch["frames"].to(device)
    B, T = frames.shape[:2]
    frames_flat = frames.view(B * T, *frames.shape[2:])
    pyramid = backbone(frames_flat)
    feats = pyramid[fpn_level]
    _, C, H, W = feats.shape
    return feats.view(B, T, C, H, W)


@torch.no_grad()
def validate(
    models: dict,
    loader: DataLoader,
    fpn_level: str,
    intra_period: int,
    step_size: float,
    n_pixels: int,
    device: str,
) -> dict:
    for m in ["compressor", "decompressor", "entropy", "predictor"]:
        models[m].eval()

    total_mse = 0.0
    total_rate = 0.0
    total_nll = 0.0
    n = 0

    for batch in loader:
        feats_seq = extract_fpn(batch, models["backbone"], fpn_level, device)
        B, T = feats_seq.shape[:2]

        for b in range(B):
            recon_latents = []
            for t in range(T):
                feat = feats_seq[b : b + 1, t]
                latent = models["compressor"](feat)

                if t % intra_period == 0:
                    qz = models["quantizer"](latent)
                    recon_latent = qz
                    _, params = models["entropy"](qz)
                    rate = nll_rate_loss(qz, params["mu"], params["sigma"], step_size)
                else:
                    ctx = recon_latents[
                        max(0, len(recon_latents) - models["predictor"].context_len) :
                    ]
                    with torch.no_grad():
                        pred_mean, _ = models["predictor"].forward(ctx)
                    res = latent - pred_mean
                    qr = models["quantizer"](res)
                    recon_latent = pred_mean + qr
                    _, params = models["entropy"](qr)
                    rate = nll_rate_loss(qr, params["mu"], params["sigma"], step_size)

                recon_feat = models["decompressor"](recon_latent)
                recon_latents.append(recon_latent)

                mse = F.mse_loss(recon_feat, feat).item()
                total_mse += mse
                total_rate += rate.item()
                total_nll += rate.item()
                n += 1

    avg_mse = total_mse / max(n, 1)
    avg_bpp = total_rate / max(n, 1) / n_pixels
    feat_psnr = 10 * torch.log10(torch.tensor(1.0 / max(avg_mse, 1e-10))).item()

    for m in ["compressor", "decompressor", "entropy", "predictor"]:
        models[m].train()

    return {
        "feat_psnr": feat_psnr,
        "mse": avg_mse,
        "bpp": avg_bpp,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="FPN compressor + temporal predictor training")
    parser.add_argument(
        "--roots",
        type=str,
        nargs="+",
        default=[
            "datasets/seadronessee/frames",
            "datasets/smd/frames",
        ],
    )
    parser.add_argument(
        "--fpn-level", type=str, default="P4", help="FPN pyramid level to train on (P3/P4/P5)"
    )
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument(
        "--intra-period", type=int, default=8, help="I-frame interval (every N frames)"
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hyper-channels", type=int, default=32)
    parser.add_argument("--predictor-hidden", type=int, default=128)
    parser.add_argument("--predictor-layers", type=int, default=4)
    parser.add_argument("--predictor-heads", type=int, default=2)
    parser.add_argument("--context-len", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--rd-steps", type=int, default=15000)
    parser.add_argument("--temporal-steps", type=int, default=20000)
    parser.add_argument("--rd-lambda", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="checkpoints/fpn_compress")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--val-interval", type=int, default=500)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build models
    models = build_models(
        latent_dim=args.latent_dim,
        hyper_channels=args.hyper_channels,
        hidden_dim=args.predictor_hidden,
        predictor_layers=args.predictor_layers,
        predictor_heads=args.predictor_heads,
        context_len=args.context_len,
        fpn_level=args.fpn_level,
        device=device,
    )
    n_trainable = count_params(models)
    print(f"Trainable params: {n_trainable:,}")
    n_pixels = (args.image_size // 16) ** 2
    step_size = models["quantizer"].step_size.item()

    # Create loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_loaders(
        args.roots,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0

    # ======================================================================
    # Phase 0: Warmup — compressor + decompressor only, no rate, no predictor
    # ======================================================================
    print(f"\n{'=' * 60}")
    print(f"Phase 0: Autoencoder warmup ({args.warmup_steps} steps)")
    print(f"{'=' * 60}")

    # Freeze predictor and entropy
    for p in models["predictor"].parameters():
        p.requires_grad = False
    for p in models["entropy"].parameters():
        p.requires_grad = False

    opt_cd = torch.optim.AdamW(
        [
            {"params": models["compressor"].parameters(), "lr": args.lr},
            {"params": models["decompressor"].parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    train_iter = iter(train_loader)
    t0 = time.time()
    for step in range(1, args.warmup_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        feats_seq = extract_fpn(batch, models["backbone"], args.fpn_level, device)
        B, T = feats_seq.shape[:2]
        feats_flat = feats_seq.view(B * T, *feats_seq.shape[2:])

        latent = models["compressor"](feats_flat)
        qz = models["quantizer"](latent)
        recon = models["decompressor"](qz)

        loss = F.mse_loss(recon, feats_flat)

        opt_cd.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(models["compressor"].parameters()) + list(models["decompressor"].parameters()),
            args.max_grad_norm,
        )
        opt_cd.step()
        global_step += 1

        if step % args.log_interval == 0:
            print(
                f"  step={step}/{args.warmup_steps}  loss={loss.item():.4f}  [{time.time() - t0:.0f}s]"
            )

        if step % args.val_interval == 0:
            metrics = validate(
                models, val_loader, args.fpn_level, args.intra_period, step_size, n_pixels, device
            )
            print(
                f"  [val] step={step}  feat_PSNR={metrics['feat_psnr']:.2f}  BPP={metrics['bpp']:.4f}"
            )

    # ======================================================================
    # Phase 1: Rate-aware training — add entropy model
    # ======================================================================
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Rate-distortion training ({args.rd_steps} steps)")
    print(f"{'=' * 60}")

    for p in models["entropy"].parameters():
        p.requires_grad = True

    opt_rd = torch.optim.AdamW(
        [
            {"params": models["compressor"].parameters(), "lr": args.lr},
            {"params": models["decompressor"].parameters(), "lr": args.lr},
            {"params": models["entropy"].parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    sched_rd = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rd, T_max=args.rd_steps)

    train_iter = iter(train_loader)
    for step in range(1, args.rd_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        frac = step / args.rd_steps
        rd_lambda = args.rd_lambda * (1.0 - 0.9 * frac)

        feats_seq = extract_fpn(batch, models["backbone"], args.fpn_level, device)
        B, T = feats_seq.shape[:2]
        feats_flat = feats_seq.view(B * T, *feats_seq.shape[2:])

        latent = models["compressor"](feats_flat)
        qz = models["quantizer"](latent)
        recon = models["decompressor"](qz)

        recon_loss = F.mse_loss(recon, feats_flat)
        _, params = models["entropy"](qz)
        rate = nll_rate_loss(qz, params["mu"], params["sigma"], step_size)
        rate_bpp = rate / n_pixels
        loss = recon_loss + rd_lambda * rate_bpp

        opt_rd.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(models["compressor"].parameters())
            + list(models["decompressor"].parameters())
            + list(models["entropy"].parameters()),
            args.max_grad_norm,
        )
        opt_rd.step()
        sched_rd.step()
        global_step += 1

        if step % args.log_interval == 0:
            feat_psnr = 10 * torch.log10(1.0 / recon_loss.detach()).item()
            print(
                f"  step={step}/{args.rd_steps}  loss={loss.item():.4f}  "
                f"PSNR={feat_psnr:.2f}  BPP={rate_bpp.item():.4f}  "
                f"\u03bb={rd_lambda:.3f}  [{time.time() - t0:.0f}s]"
            )

        if step % args.val_interval == 0:
            metrics = validate(
                models, val_loader, args.fpn_level, args.intra_period, step_size, n_pixels, device
            )
            val_loss = metrics["feat_psnr"] - 10 * max(metrics["bpp"] - 0.5, 0)
            print(
                f"  [val] step={step}  feat_PSNR={metrics['feat_psnr']:.2f}  BPP={metrics['bpp']:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = out_dir / "phase1_best.pt"
                torch.save(
                    {
                        "step": global_step,
                        "phase": "rd",
                        "fpn_level": args.fpn_level,
                        "compressor": models["compressor"].state_dict(),
                        "decompressor": models["decompressor"].state_dict(),
                        "entropy": models["entropy"].state_dict(),
                        "feat_psnr": metrics["feat_psnr"],
                        "bpp": metrics["bpp"],
                    },
                    ckpt_path,
                )
                print(f"    -> saved {ckpt_path}")

    # ======================================================================
    # Phase 2: Temporal — train predictor on latent residuals
    # ======================================================================
    print(f"\n{'=' * 60}")
    print(f"Phase 2: Temporal predictor training ({args.temporal_steps} steps)")
    print(f"{'=' * 60}")

    for p in models["predictor"].parameters():
        p.requires_grad = True

    opt_all = torch.optim.AdamW(
        [
            {"params": models["compressor"].parameters(), "lr": args.lr * 0.5},
            {"params": models["decompressor"].parameters(), "lr": args.lr * 0.5},
            {"params": models["entropy"].parameters(), "lr": args.lr * 0.5},
            {"params": models["predictor"].parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    sched_all = torch.optim.lr_scheduler.CosineAnnealingLR(opt_all, T_max=args.temporal_steps)

    train_iter = iter(train_loader)
    for step in range(1, args.temporal_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        frac = step / args.temporal_steps
        rd_lambda = args.rd_lambda * 0.1 * (1.0 - 0.5 * frac)

        feats_seq = extract_fpn(batch, models["backbone"], args.fpn_level, device)
        B, T = feats_seq.shape[:2]

        opt_all.zero_grad()
        total_loss = 0.0

        for b in range(B):
            recon_latents = []
            seq_recon_loss = 0.0
            seq_rate_loss = 0.0
            seq_nll_loss = 0.0

            for t in range(T):
                feat = feats_seq[b : b + 1, t]
                latent = models["compressor"](feat)

                if t % args.intra_period == 0:
                    qz = models["quantizer"](latent)
                    recon_latent = qz
                    recon_latents.append(recon_latent)
                    _, params = models["entropy"](qz)
                    rate = nll_rate_loss(qz, params["mu"], params["sigma"], step_size)
                    seq_rate_loss = seq_rate_loss + rate
                else:
                    ctx = recon_latents[max(0, len(recon_latents) - args.context_len) :]
                    pred_mean, _ = models["predictor"].forward(ctx)
                    nll = models["predictor"].nll_loss(ctx, latent)
                    seq_nll_loss = seq_nll_loss + nll

                    res = latent - pred_mean
                    qr = models["quantizer"](res)
                    recon_latent = pred_mean + qr
                    recon_latents.append(recon_latent)
                    _, params = models["entropy"](qr)
                    rate = nll_rate_loss(qr, params["mu"], params["sigma"], step_size)
                    seq_rate_loss = seq_rate_loss + rate

                recon_feat = models["decompressor"](recon_latent)
                seq_recon_loss = seq_recon_loss + F.mse_loss(recon_feat, feat)

            loss = (
                seq_recon_loss / T
                + rd_lambda * seq_rate_loss / T
                + 0.01 * seq_nll_loss / max(T - 1, 1)
            )
            total_loss = total_loss + loss

        total_loss = total_loss / B
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            [
                p
                for m in models.values()
                if m is not models["backbone"]
                for p in m.parameters()
                if p.requires_grad
            ],
            args.max_grad_norm,
        )
        opt_all.step()
        sched_all.step()
        global_step += 1

        if step % args.log_interval == 0:
            print(
                f"  step={step}/{args.temporal_steps}  loss={total_loss.item():.4f}  "
                f"\u03bb={rd_lambda:.4f}  [{time.time() - t0:.0f}s]"
            )

        if step % args.val_interval == 0:
            metrics = validate(
                models, val_loader, args.fpn_level, args.intra_period, step_size, n_pixels, device
            )
            val_loss = metrics["feat_psnr"] - 10 * max(metrics["bpp"] - 0.5, 0)
            print(
                f"  [val] step={step}  feat_PSNR={metrics['feat_psnr']:.2f}  "
                f"BPP={metrics['bpp']:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = out_dir / "best.pt"
                torch.save(
                    {
                        "step": global_step,
                        "phase": "temporal",
                        "fpn_level": args.fpn_level,
                        "compressor": models["compressor"].state_dict(),
                        "decompressor": models["decompressor"].state_dict(),
                        "entropy": models["entropy"].state_dict(),
                        "predictor": models["predictor"].state_dict(),
                        "feat_psnr": metrics["feat_psnr"],
                        "bpp": metrics["bpp"],
                    },
                    ckpt_path,
                )
                print(f"    -> saved {ckpt_path}")

    print(f"\n{'=' * 60}")
    elapsed_h = (time.time() - t0) / 3600
    print(f"Training complete!  ({elapsed_h:.1f}h)")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
