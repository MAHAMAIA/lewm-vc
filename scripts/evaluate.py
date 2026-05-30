#!/usr/bin/env python3
"""
Evaluation script for LeWM-VC models.

Loads a trained checkpoint, runs on test frames, and reports:
  - PSNR, MS-SSIM, LPIPS (distortion)
  - BPP (bitrate from entropy model)
  - Task loss (ResNet-18 feature distance)
  - Temporal mode (--temporal): IPPP coding with predictor, separates I/P metrics

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/lambda_0.05/final.pt
    python scripts/evaluate.py --checkpoint checkpoints/lambda_0.05/step_50000.pt \\
        --data datasets/pevid/frames --output eval_results
    python scripts/evaluate.py --checkpoint ... --temporal --image-size 256 --num-frames 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

try:
    from skimage.metrics import structural_similarity as ssim

    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

try:
    import lpips

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


def _load_config(config_path: str) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def _discover_frames(root: str, max_frames: int | None = None) -> list[Path]:
    root_p = Path(root)
    all_frames = sorted(root_p.rglob("*.png"))
    if max_frames:
        step = max(1, len(all_frames) // max_frames)
        all_frames = all_frames[::step]
    return all_frames


def psnr(mse: float, max_val: float = 1.0) -> float:
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val**2 / mse)


def ms_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    if not HAS_SSIM:
        return 0.0
    img1_np = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.detach().cpu().numpy().transpose(1, 2, 0)
    return float(ssim(img1_np, img2_np, channel_axis=2, data_range=1.0))


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    config_path: str,
    data_root: str,
    output_dir: str,
    image_size: int,
    num_frames: int | None,
    device: str,
    temporal: bool = False,
):
    config = _load_config(config_path)
    model_cfg = config.get("model", {})
    latent_dim = model_cfg.get("latent_dim", 192)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    # Import model classes
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer
    from lewm_vc.utils.task_probe import create_task_probe

    # Build models
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

    # Load checkpoint
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
            print(f"  loaded {name}")
        else:
            print(f"  [warn] no state_dict for {name}")

    for m in [encoder, predictor, decoder, entropy_model, quantizer]:
        m.to(device)
        m.eval()

    # Task probe
    task_probe = create_task_probe("resnet18").to(device).eval()

    # Init LPIPS
    lpips_fn = None
    if HAS_LPIPS:
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    # Discover frames
    frames = _discover_frames(data_root, num_frames)
    if not frames:
        print(f"ERROR: no frames found in {data_root}")
        sys.exit(1)
    print(f"Evaluating on {len(frames)} frames from {data_root}")

    from PIL import Image

    results = []
    total_mse = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_bpp = 0.0
    total_task = 0.0
    count = 0

    if temporal:
        results = _evaluate_temporal(
            encoder,
            predictor,
            decoder,
            entropy_model,
            quantizer,
            frames,
            device,
            image_size,
            lpips_fn,
            task_probe,
        )
    else:
        results = _evaluate_intra(
            encoder,
            decoder,
            entropy_model,
            quantizer,
            frames,
            device,
            image_size,
            lpips_fn,
            task_probe,
        )

    # Aggregate
    count = len(results)
    avg_mse = sum(r["mse"] for r in results) / count
    avg_psnr_val = psnr(avg_mse)
    avg_ssim = sum(r["ssim"] for r in results) / count
    avg_lpips = sum(r["lpips"] for r in results) / count
    avg_bpp = sum(r["bpp"] for r in results) / count
    avg_task = sum(r["task_loss"] for r in results) / count
    i_bpp = sum(r["bpp"] for r in results if r["type"] == "I") / max(
        1, sum(1 for r in results if r["type"] == "I")
    )
    p_bpp = sum(r["bpp"] for r in results if r["type"] == "P") / max(
        1, sum(1 for r in results if r["type"] == "P")
    )

    summary = {
        "checkpoint": checkpoint_path,
        "mode": "temporal" if temporal else "intra",
        "num_frames": count,
        "image_size": image_size,
        "avg_mse": avg_mse,
        "avg_psnr": avg_psnr_val,
        "avg_ssim": avg_ssim,
        "avg_lpips": avg_lpips,
        "avg_bpp": avg_bpp,
        "avg_task_loss": avg_task,
        "i_frame_bpp": i_bpp,
        "p_frame_bpp": p_bpp,
        "p_i_bpp_ratio": p_bpp / i_bpp if i_bpp > 0 else 0,
    }

    print("\n=== Results ===")
    print(f"  Mode:      {'temporal IPPP' if temporal else 'intra-only'}")
    print(f"  PSNR:      {avg_psnr_val:.2f} dB")
    print(f"  MS-SSIM:   {avg_ssim:.4f}")
    print(f"  LPIPS:     {avg_lpips:.4f}")
    print(f"  BPP:       {avg_bpp:.4f}")
    if temporal:
        print(f"  I-frame BPP: {i_bpp:.4f}")
        print(f"  P-frame BPP: {p_bpp:.4f}  (I/P ratio: {p_bpp / i_bpp:.3f})")
    print(f"  Task Loss: {avg_task:.6f}")

    # Save JSON
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_frame.json").write_text(
        json.dumps(
            [
                {
                    "frame": r["frame"],
                    "type": r["type"],
                    "psnr": round(r["psnr"], 2),
                    "bpp": round(r["bpp"], 4),
                }
                for r in results
            ],
            indent=2,
        )
    )

    # HTML report
    table_rows = ""
    for r in results:
        table_rows += (
            f"<tr>"
            f"<td style='font-weight:700'>{r['type']}</td>"
            f"<td>{r['frame']}</td>"
            f"<td>{r['psnr']:.2f}</td>"
            f"<td>{r['bpp']:.4f}</td>"
            f"<td>{r['lpips']:.4f}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LeWM-VC Evaluation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 20px; background: #f8f9fa; color: #333; }}
h1 {{ color: #1a1a2e; }}
.summary {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
.card {{ background: white; padding: 16px 20px; border-radius: 8px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex: 1; min-width: 120px; }}
.card .num {{ font-size: 24px; font-weight: 700; color: #1a1a2e; }}
.card .lbl {{ font-size: 12px; color: #666; text-transform: uppercase; }}
table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px;
         overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th {{ background: #1a1a2e; color: white; padding: 8px 12px; text-align: left;
      font-size: 12px; text-transform: uppercase; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #eee; font-size: 13px; }}
tr:hover {{ background: #f0f4ff; }}
.footer {{ margin-top: 24px; font-size: 11px; color: #999; text-align: center; }}
</style>
</head>
<body>
<h1>LeWM-VC Evaluation Report</h1>
<p>Checkpoint: {checkpoint_path}</p>

<div class="summary">
  <div class="card"><div class="num">{avg_psnr_val:.2f}</div><div class="lbl">PSNR (dB)</div></div>
  <div class="card"><div class="num">{avg_bpp:.4f}</div><div class="lbl">Avg BPP</div></div>
  <div class="card"><div class="num">{i_bpp:.4f}</div><div class="lbl">I-frame BPP</div></div>
  <div class="card"><div class="num">{p_bpp:.4f}</div><div class="lbl">P-frame BPP</div></div>
  <div class="card"><div class="num">{p_bpp / i_bpp:.3f}</div><div class="lbl">I/P ratio</div></div>
  <div class="card"><div class="num">{avg_lpips:.4f}</div><div class="lbl">LPIPS</div></div>
</div>

<h2>Per-Frame Results</h2>
<div style="max-height:500px;overflow-y:auto;">
<table>
<thead><tr><th>Type</th><th>Frame</th><th>PSNR</th><th>BPP</th><th>LPIPS</th></tr></thead>
<tbody>{table_rows}</tbody>
</table>
</div>

<div class="footer">LeWM-VC Evaluation — MAHAMAIA Systems</div>
</body>
</html>"""
    (out / "report.html").write_text(html)
    print(f"\nReport: {out / 'report.html'}")


@torch.no_grad()
def _evaluate_intra(
    encoder,
    decoder,
    entropy_model,
    quantizer,
    frames,
    device,
    image_size,
    lpips_fn,
    task_probe,
) -> list[dict]:
    """Intra-frame evaluation: each frame encoded independently."""
    from PIL import Image

    results = []
    for i, fpath in enumerate(frames):
        img = Image.open(fpath).convert("RGB").resize((image_size, image_size))
        tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        latent, _ = encoder(tensor, return_surprise=True)
        quant_latent = quantizer(latent)
        recon = decoder(quant_latent)

        rate, _ = entropy_model(quant_latent)
        bpp = rate.item() / (tensor.shape[2] * tensor.shape[3])
        mse_val = F.mse_loss(recon, tensor).item()
        ssim_val = ms_ssim(recon.squeeze(0), tensor.squeeze(0))
        lpips_val = lpips_fn(recon, tensor).item() if lpips_fn else 0.0
        task_val = F.mse_loss(task_probe(recon), task_probe(tensor)).item()

        results.append(
            {
                "frame": fpath.name,
                "mse": mse_val,
                "psnr": psnr(mse_val),
                "ssim": ssim_val,
                "lpips": lpips_val,
                "bpp": bpp,
                "task_loss": task_val,
                "type": "I",
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(frames)}] I-frame  PSNR={psnr(mse_val):.2f}  BPP={bpp:.4f}")
    return results


@torch.no_grad()
def _evaluate_temporal(
    encoder,
    predictor,
    decoder,
    entropy_model,
    quantizer,
    frames,
    device,
    image_size,
    lpips_fn,
    task_probe,
) -> list[dict]:
    """Temporal IPPP evaluation with predictor + residual coding."""
    from PIL import Image

    results = []
    recon_latents = []
    context_len = getattr(predictor, "context_len", 4)

    for i, fpath in enumerate(frames):
        img = Image.open(fpath).convert("RGB").resize((image_size, image_size))
        tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        latent, _ = encoder(tensor, return_surprise=True)

        if i == 0:
            # I-frame: quantize raw latent, decode, store as context
            quant_latent = quantizer(latent)
            recon = decoder(quant_latent)
            rate, _ = entropy_model(quant_latent)
            frame_type = "I"
            recon_latents.append(quant_latent)
        else:
            # P-frame: predict from decoded latents, code residual, reconstruct
            ctx = recon_latents[max(0, len(recon_latents) - context_len) :]
            pred_mean, _ = predictor(ctx)
            residual = latent - pred_mean
            quant_residual = quantizer(residual)
            rate, _ = entropy_model(quant_residual)
            recon_latent = pred_mean + quant_residual
            recon = decoder(recon_latent)
            frame_type = "P"
            recon_latents.append(recon_latent)

        bpp = rate.item() / (tensor.shape[2] * tensor.shape[3])
        mse_val = F.mse_loss(recon, tensor).item()
        ssim_val = ms_ssim(recon.squeeze(0), tensor.squeeze(0))
        lpips_val = lpips_fn(recon, tensor).item() if lpips_fn else 0.0
        task_val = F.mse_loss(task_probe(recon), task_probe(tensor)).item()

        results.append(
            {
                "frame": fpath.name,
                "mse": mse_val,
                "psnr": psnr(mse_val),
                "ssim": ssim_val,
                "lpips": lpips_val,
                "bpp": bpp,
                "task_loss": task_val,
                "type": frame_type,
            }
        )

        if (i + 1) % 20 == 0:
            print(
                f"  [{i + 1}/{len(frames)}] {frame_type}-frame  PSNR={psnr(mse_val):.2f}  BPP={bpp:.4f}"
            )

    # Print summary by frame type
    i_frames = [r for r in results if r["type"] == "I"]
    p_frames = [r for r in results if r["type"] == "P"]
    if i_frames:
        i_bpp = sum(r["bpp"] for r in i_frames) / len(i_frames)
        print(f"\n  I-frames ({len(i_frames)}): avg BPP={i_bpp:.4f}")
    if p_frames:
        p_bpp = sum(r["bpp"] for r in p_frames) / len(p_frames)
        i_bpp_avg = sum(r["bpp"] for r in i_frames) / len(i_frames) if i_frames else 0.0
        print(
            f"  P-frames ({len(p_frames)}): avg BPP={p_bpp:.4f}  (I/P ratio={p_bpp / i_bpp_avg:.3f})"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LeWM-VC model")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--data", default="datasets/pevid/frames", help="Test data root")
    parser.add_argument("--output", default="eval_results")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--temporal", action="store_true", help="Use IPPP temporal coding with predictor"
    )
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.config,
        args.data,
        args.output,
        args.image_size,
        args.num_frames,
        args.device,
        temporal=args.temporal,
    )


if __name__ == "__main__":
    main()
