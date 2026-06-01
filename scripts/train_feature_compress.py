import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def nll_rate_loss(latents, mu, sigma, step_size):
    half = step_size / 2.0
    prob = torch.special.ndtr((latents + half - mu) / sigma) - torch.special.ndtr(
        (latents - half - mu) / sigma
    )
    prob = prob.clamp(min=1e-10)
    nats = -torch.log(prob)
    bits = nats / 0.693147
    return bits.sum() / latents.shape[0]


def build_models(config, device):
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.feature_compress import (
        FeatureCompressor,
        FeatureDecompressor,
        ResNetFeatureExtractor,
    )
    from lewm_vc.quant import Quantizer, QuantMode

    mc = config.get("model", {})
    latent_dim = mc.get("latent_dim", 8)

    backbone = ResNetFeatureExtractor(backbone_name=mc.get("backbone", "resnet50"))
    feat_channels = backbone.feature_channels
    compressor = FeatureCompressor(
        in_channels=feat_channels,
        latent_dim=latent_dim,
    )
    decompressor = FeatureDecompressor(
        latent_dim=latent_dim,
        out_channels=feat_channels,
    )
    entropy_model = HyperpriorEntropy(
        latent_dim=latent_dim,
        hyper_channels=mc.get("entropy", {}).get("hyper_channels", 32),
        num_components=mc.get("entropy", {}).get("num_components", 2),
    )
    quantizer = Quantizer()
    quantizer.set_mode(QuantMode.TRAINING)

    models = {
        "backbone": backbone.to(device),
        "compressor": compressor.to(device),
        "decompressor": decompressor.to(device),
        "entropy_model": entropy_model.to(device),
        "quantizer": quantizer.to(device),
    }
    for m in models.values():
        m.train() if not isinstance(m, (Quantizer, ResNetFeatureExtractor)) else m.eval()
    return models


def create_loaders(config):
    from lewm_vc.data import FrameDataset, collate_sequences

    dc = config.get("data", {})
    roots = dc.get("roots", [])
    image_size = dc.get("image_size", 256)
    augment = dc.get("augment", True)
    tc = config.get("training", {})
    batch_size = tc.get("batch_size", 16)
    num_workers = tc.get("num_workers", 4)

    train_ds = FrameDataset.from_roots(
        roots,
        sequence_length=1,
        image_size=image_size,
        augment=augment,
        frame_stride=1,
        split="train",
    )
    val_ds = FrameDataset.from_roots(
        roots,
        sequence_length=1,
        image_size=image_size,
        augment=False,
        frame_stride=1,
        split="val",
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_sequences,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_sequences,
    )
    return train_loader, val_loader


def freeze_models(models, freeze_list):
    for name, m in models.items():
        requires_grad = name not in freeze_list
        for p in m.parameters():
            p.requires_grad = requires_grad
        print(f"  {name}: {'active' if requires_grad else 'frozen'}")


def main():
    parser = argparse.ArgumentParser(description="Feature compression training (MPEG VCM)")
    parser.add_argument("--config", default="configs/train_feature_compress.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    device = args.device

    print("Building models...")
    models = build_models(config, device)
    quantizer = models["quantizer"]
    step_size = quantizer.step_size.item()
    n_pixels = config.get("data", {}).get("image_size", 256) ** 2

    print("Creating data loaders...")
    train_loader, val_loader = create_loaders(config)

    phases = config.get("phases", {})
    global_step = 0

    for phase_key in sorted(phases.keys(), key=lambda k: int(k.replace("phase", ""))):
        pc = phases[phase_key]
        name = pc.get("name", phase_key)
        steps = pc["steps"]
        rate_weight = pc.get("rate_weight", 1.0)
        rd_lambda_start = pc.get("lambda", 0.05)
        rd_lambda_end = pc.get("lambda_end", rd_lambda_start)
        freeze_list = pc.get("freeze", [])

        print(f"\n{'=' * 60}")
        print(f"Phase {phase_key} ({name}) — {steps} steps")
        print(f"{'=' * 60}")

        freeze_models(models, freeze_list + ["quantizer", "backbone"])

        opt = torch.optim.AdamW(
            [
                {
                    "params": models["compressor"].parameters(),
                    "lr": config.get("training", {}).get("lr_compressor", 3e-4),
                },
                {
                    "params": models["decompressor"].parameters(),
                    "lr": config.get("training", {}).get("lr_decompressor", 3e-4),
                },
                {
                    "params": models["entropy_model"].parameters(),
                    "lr": config.get("training", {}).get("lr_entropy", 3e-4),
                },
            ],
            weight_decay=config.get("training", {}).get("weight_decay", 0.01),
        )

        trainable = sum(
            p.numel()
            for p in (
                list(models["compressor"].parameters())
                + list(models["decompressor"].parameters())
                + list(models["entropy_model"].parameters())
            )
            if p.requires_grad
        )
        print(f"  trainable params: {trainable:,}")

        t0 = time.time()
        train_iter = iter(train_loader)
        best_val = float("inf")

        for step in range(1, steps + 1):
            frac = step / steps
            rd_lambda = rd_lambda_start + (rd_lambda_end - rd_lambda_start) * frac

            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            frames = batch["frames"].to(device)
            frames = frames[:, 0]

            opt.zero_grad()

            with torch.no_grad():
                target_feats = models["backbone"](frames)

            latent = models["compressor"](target_feats)
            qz = models["quantizer"](latent)
            recon_feats = models["decompressor"](qz)

            task_loss = nn.functional.mse_loss(recon_feats, target_feats)
            loss = task_loss

            if rate_weight > 0:
                _, params = models["entropy_model"](qz)
                rate = nll_rate_loss(qz, params["mu"], params["sigma"], step_size)
                rate_bpp = rate / n_pixels
                loss = loss + rate_weight * rd_lambda * rate_bpp
            else:
                rate = torch.tensor(0.0, device=device)
                rate_bpp = torch.tensor(0.0, device=device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for m in ["compressor", "decompressor", "entropy_model"]
                    for p in models[m].parameters()
                    if p.requires_grad
                ],
                config.get("training", {}).get("max_grad_norm", 1.0),
            )
            opt.step()
            global_step += 1

            if step % config.get("logging", {}).get("log_interval", 20) == 0:
                feat_psnr = 10 * torch.log10(1.0 / task_loss.detach()).item()
                bpp_str = f"  BPP={rate_bpp.item():.4f}" if rate_weight > 0 else ""
                lam_str = (
                    f"  λ={rd_lambda:.4f}"
                    if rate_weight > 0 and rd_lambda_start != rd_lambda_end
                    else ""
                )
                print(
                    f"  step={step}/{steps}  loss={loss.item():.4f}  feat_PSNR={feat_psnr:.2f}{bpp_str}{lam_str}  [{time.time() - t0:.0f}s]"
                )

            if step % config.get("logging", {}).get("val_interval", 500) == 0:
                models["compressor"].eval()
                models["decompressor"].eval()
                models["entropy_model"].eval()
                v_task = 0.0
                v_rate = 0.0
                v_count = 0

                with torch.no_grad():
                    for vbatch in val_loader:
                        vf = vbatch["frames"].to(device)[:, 0]
                        vfeats = models["backbone"](vf)
                        vl = models["compressor"](vfeats)
                        vqz = models["quantizer"](vl)
                        vrecon = models["decompressor"](vqz)
                        v_task = v_task + nn.functional.mse_loss(vrecon, vfeats)
                        if rate_weight > 0:
                            _, vp = models["entropy_model"](vqz)
                            vr = nll_rate_loss(vqz, vp["mu"], vp["sigma"], step_size)
                            v_rate = v_rate + vr
                        v_count += 1

                avg_v_task = v_task / v_count
                v_feat_psnr = 10 * torch.log10(1.0 / avg_v_task).item()
                v_bpp = (v_rate / v_count / n_pixels).item() if rate_weight > 0 else 0.0

                val_loss = v_feat_psnr
                if rate_weight > 0 and v_bpp < 0.5:
                    val_loss = v_feat_psnr - 10 * (0.5 - v_bpp)

                print(f"  [val] step={step}  feat_PSNR={v_feat_psnr:.2f}  BPP={v_bpp:.4f}")

                if val_loss < best_val:
                    best_val = val_loss
                    ckpt_path = (
                        Path(
                            config.get("checkpoint", {}).get("dir", "checkpoints/feature_compress")
                        )
                        / "best.pt"
                    )
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "step": global_step,
                            "phase": phase_key,
                            "models": {
                                "compressor": models["compressor"].state_dict(),
                                "decompressor": models["decompressor"].state_dict(),
                                "entropy_model": models["entropy_model"].state_dict(),
                            },
                            "config": config,
                            "bpp": v_bpp,
                            "feat_psnr": v_feat_psnr,
                            "val_loss": val_loss,
                        },
                        ckpt_path,
                    )
                    print(f"    -> NEW BEST (feat_PSNR={v_feat_psnr:.2f}, BPP={v_bpp:.4f})")
                    print(f"       saved to {ckpt_path}")

    print(f"\n{'=' * 60}")
    print(f"Training complete! Best checkpoint at {ckpt_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
