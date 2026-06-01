"""
Measure real BPP using zlib + torchac arithmetic coding.

Usage:
    python scripts/measure_bpp.py --checkpoint checkpoints/sentinel-p1-l60.0-809b54/lambda_60.0/best.pt
    python scripts/measure_bpp.py --checkpoint ... --data datasets/virat/frames --output bpp_results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    import lpips as lpips_lib

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


import zlib
import torchac


def build_gaussian_cdf(
    mu: torch.Tensor, sigma: torch.Tensor, step_size: float, num_levels: int = 256
) -> torch.Tensor:
    B, C, H, W = mu.shape
    device = mu.device
    half = num_levels // 2

    offsets = (
        torch.arange(0, num_levels + 1, dtype=torch.float32, device=device) - half - 0.5
    ) * step_size
    z = (offsets - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
    cdf_float = torch.special.ndtr(z)

    cdf_min = cdf_float[..., 0:1]
    cdf_max = cdf_float[..., -1:]
    cdf_float = (cdf_float - cdf_min) / (cdf_max - cdf_min + 1e-10)
    cdf_float[..., 0] = 0.0
    cdf_float[..., -1] = 1.0

    cdf_int = (cdf_float * 65535 + 0.5).to(torch.int32)
    cdf_int = torch.where(cdf_int > 32767, cdf_int - 65536, cdf_int).to(torch.int16)
    cdf_int[..., -1] = -1

    return cdf_int


def encode_with_torchac(
    quantized: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, step_size: float
) -> bytes:
    assert quantized.dim() == 4
    B, C, H, W = quantized.shape

    indices = torch.round(quantized / step_size).clamp(-128, 127).to(torch.int16) + 128
    indices = indices.to(torch.int16).cpu()

    mu_cpu = mu.detach().cpu()
    sigma_cpu = sigma.detach().cpu()
    cdf = build_gaussian_cdf(mu_cpu, sigma_cpu, step_size)

    encoded = torchac.encode_int16_normalized_cdf(cdf, indices)
    return encoded


def decode_with_torchac(
    encoded: bytes, mu: torch.Tensor, sigma: torch.Tensor, step_size: float, shape
) -> torch.Tensor:
    mu_cpu = mu.detach().cpu()
    sigma_cpu = sigma.detach().cpu()
    cdf = build_gaussian_cdf(mu_cpu, sigma_cpu, step_size)
    decoded = torchac.decode_int16_normalized_cdf(cdf, encoded).to(mu.device)

    values = (decoded.float() - 128) * step_size
    return values.reshape(shape)


def load_model(checkpoint_path: str, config_path: str, device: str = "cuda"):
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
            print(f"  loaded {name}")
        else:
            print(f"  [warn] no state_dict for {name}")

    for m in [encoder, predictor, decoder, entropy_model, quantizer]:
        m.to(device)
        m.eval()

    return encoder, predictor, decoder, entropy_model, quantizer


def quantized_to_indices(tensor: torch.Tensor, step_size: float) -> np.ndarray:
    arr = tensor.squeeze(0).cpu().numpy()
    indices = np.round(arr / step_size).clip(-128, 127).astype(np.int16) + 128
    return indices.astype(np.uint8)


def compress_with_zlib(tensor: torch.Tensor, step_size: float) -> bytes:
    indices = quantized_to_indices(tensor, step_size)
    return zlib.compress(indices.tobytes(), level=6)


def discover_frames(data_root: str, num_frames: int | None = None) -> list[Path]:
    root_p = Path(data_root)
    clip_dirs = sorted([d for d in root_p.iterdir() if d.is_dir()])
    if clip_dirs:
        first_clip = clip_dirs[0]
        all_frames = sorted(first_clip.rglob("*.png"))
    else:
        all_frames = sorted(root_p.rglob("*.png"))
    if num_frames and num_frames < len(all_frames):
        all_frames = all_frames[:num_frames]
    return all_frames


def psnr_from_mse(mse: float) -> float:
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


@torch.no_grad()
def measure_bpp(
    checkpoint_path: str,
    config_path: str,
    data_root: str,
    output_dir: str,
    image_size: int,
    num_frames: int | None,
    device: str,
):
    from PIL import Image
    from lewm_vc.utils.task_probe import create_task_probe

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    encoder, predictor, decoder, entropy_model, quantizer = load_model(
        checkpoint_path, config_path, device
    )

    step_size = quantizer.step_size.item()
    print(f"  Quantizer step_size: {step_size:.6f}")

    task_probe = create_task_probe("resnet18").to(device).eval()
    lpips_fn = lpips_lib.LPIPS(net="vgg", verbose=False).to(device).eval() if HAS_LPIPS else None

    frames = discover_frames(data_root, num_frames)
    print(f"Measuring BPP on {len(frames)} frames from {data_root}")

    recon_latents = []
    context_len = getattr(predictor, "context_len", 4)
    results = []
    total_zlib_bytes = 0
    total_torchac_bytes = 0
    latent_h = image_size // 16
    latent_w = image_size // 16

    for i, fpath in enumerate(frames):
        img = Image.open(fpath).convert("RGB").resize((image_size, image_size))
        tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        latent, _ = encoder(tensor, return_surprise=True)

        if i == 0:
            quant_latent = quantizer(latent)
            recon = decoder(quant_latent)
            frame_type = "I"
            recon_latents.append(quant_latent)

            zlib_bytes = compress_with_zlib(quant_latent, step_size)
            _, params = entropy_model(quant_latent)
            ac_bytes = encode_with_torchac(quant_latent, params["mu"], params["sigma"], step_size)
            kl_rate = entropy_model.gaussian_kl(quant_latent, params["mu"], params["sigma"])
            kl_bpp = kl_rate.sum().item() / (tensor.shape[2] * tensor.shape[3])
        else:
            ctx = recon_latents[max(0, len(recon_latents) - context_len) :]
            pred_mean, _ = predictor(ctx)
            residual = latent - pred_mean
            quant_residual = quantizer(residual)
            recon_latent = pred_mean + quant_residual
            recon = decoder(recon_latent)
            frame_type = "P"
            recon_latents.append(recon_latent)

            zlib_bytes = compress_with_zlib(quant_residual, step_size)
            _, params = entropy_model(quant_residual)
            ac_bytes = encode_with_torchac(quant_residual, params["mu"], params["sigma"], step_size)
            kl_rate = entropy_model.gaussian_kl(quant_residual, params["mu"], params["sigma"])
            kl_bpp = kl_rate.sum().item() / (tensor.shape[2] * tensor.shape[3])

        num_pixels = image_size * image_size
        zlib_bpp = len(zlib_bytes) * 8 / num_pixels
        ac_bpp = len(ac_bytes) * 8 / num_pixels
        total_zlib_bytes += len(zlib_bytes)
        total_torchac_bytes += len(ac_bytes)

        mse_val = F.mse_loss(recon, tensor).item()
        psnr_val = psnr_from_mse(mse_val)
        lpips_val = lpips_fn(recon, tensor).item() if lpips_fn else 0.0
        task_val = F.mse_loss(task_probe(recon), task_probe(tensor)).item()

        results.append(
            {
                "frame": fpath.name,
                "type": frame_type,
                "psnr": round(psnr_val, 2),
                "zlib_bpp": round(zlib_bpp, 6),
                "zlib_bytes": len(zlib_bytes),
                "torchac_bpp": round(ac_bpp, 6),
                "torchac_bytes": len(ac_bytes),
                "kl_bpp": round(kl_bpp, 6),
                "mse": round(mse_val, 8),
                "lpips": round(lpips_val, 4),
                "task_loss": round(task_val, 6),
            }
        )

        if (i + 1) % 10 == 0:
            print(
                f"  [{i + 1}/{len(frames)}] {frame_type} PSNR={psnr_val:.2f}  zlibBPP={zlib_bpp:.4f}  acBPP={ac_bpp:.6f}  klBPP={kl_bpp:.6f}"
            )

    avg_psnr = sum(r["psnr"] for r in results) / len(results)
    avg_lpips = sum(r["lpips"] for r in results) / len(results)
    avg_task = sum(r["task_loss"] for r in results) / len(results)

    def avg_metric(key, filter_type=None):
        vals = [r[key] for r in results if filter_type is None or r["type"] == filter_type]
        return sum(vals) / len(vals) if vals else 0

    summary = {
        "checkpoint": checkpoint_path,
        "mode": "temporal IPPP",
        "num_frames": len(results),
        "image_size": image_size,
        "avg_psnr": round(avg_psnr, 2),
        "avg_lpips": round(avg_lpips, 4),
        "avg_task_loss": round(avg_task, 6),
        "zlib": {
            "avg_bpp": round(avg_metric("zlib_bpp"), 6),
            "i_bpp": round(avg_metric("zlib_bpp", "I"), 6),
            "p_bpp": round(avg_metric("zlib_bpp", "P"), 6),
            "total_bytes": total_zlib_bytes,
        },
        "torchac": {
            "avg_bpp": round(avg_metric("torchac_bpp"), 6),
            "i_bpp": round(avg_metric("torchac_bpp", "I"), 6),
            "p_bpp": round(avg_metric("torchac_bpp", "P"), 6),
            "total_bytes": total_torchac_bytes,
        },
        "kl_bpp": round(avg_metric("kl_bpp"), 6),
    }
    summary["zlib"]["p_i_ratio"] = (
        round(summary["zlib"]["p_bpp"] / summary["zlib"]["i_bpp"], 3)
        if summary["zlib"]["i_bpp"] > 0
        else 0
    )
    summary["torchac"]["p_i_ratio"] = (
        round(summary["torchac"]["p_bpp"] / summary["torchac"]["i_bpp"], 3)
        if summary["torchac"]["i_bpp"] > 0
        else 0
    )

    raw_bpp = 192 * latent_h * latent_w * 8 / (image_size * image_size)
    summary["compression_vs_raw"] = {
        "raw_bpp": round(raw_bpp, 2),
        "zlib_ratio": round(raw_bpp / summary["zlib"]["avg_bpp"], 1)
        if summary["zlib"]["avg_bpp"] > 0
        else 0,
        "torchac_ratio": round(raw_bpp / summary["torchac"]["avg_bpp"], 1)
        if summary["torchac"]["avg_bpp"] > 0
        else 0,
    }

    print("\n=== BPP Results ===")
    print(f"  PSNR:     {avg_psnr:.2f} dB")
    print(f"  LPIPS:    {avg_lpips:.4f}")
    print(f"  Task:     {avg_task:.6f}")
    print(f"  Raw BPP:  {raw_bpp:.2f}")
    print(f"  KL  BPP:  {summary['kl_bpp']:.6f}")
    print(f"  zlib BPP:")
    print(
        f"    Avg:    {summary['zlib']['avg_bpp']:.4f}  (I:{summary['zlib']['i_bpp']:.4f} P:{summary['zlib']['p_bpp']:.4f} ratio={summary['zlib']['p_i_ratio']:.3f})"
    )
    print(f"    Ratio:  {summary['compression_vs_raw']['zlib_ratio']:.1f}x vs raw")
    print(f"  torchac BPP (arithmetic coding):")
    print(
        f"    Avg:    {summary['torchac']['avg_bpp']:.6f}  (I:{summary['torchac']['i_bpp']:.6f} P:{summary['torchac']['p_bpp']:.6f} ratio={summary['torchac']['p_i_ratio']:.3f})"
    )
    print(f"    Ratio:  {summary['compression_vs_raw']['torchac_ratio']:.1f}x vs raw")

    (out / "bpp_results.json").write_text(json.dumps(summary, indent=2))
    (out / "per_frame.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}/")


def main():
    parser = argparse.ArgumentParser(description="Measure real BPP using zlib bitstream")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--data", default="datasets/virat/frames")
    parser.add_argument("--output", default="bpp_results")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    measure_bpp(
        args.checkpoint,
        args.config,
        args.data,
        args.output,
        args.image_size,
        args.num_frames,
        args.device,
    )


if __name__ == "__main__":
    main()
