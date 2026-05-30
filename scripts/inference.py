"""
Inference script for LeWM-VC temporal codec.

Encodes a video sequence (frames) using the IPPP temporal pipeline and
decodes back to frames. Saves reconstructed frames and a bitstream log.

Usage:
    python scripts/inference.py --checkpoint path/to/best.pt --input datasets/virat/frames/VID_0001 --output out_dir
    python scripts/inference.py --checkpoint path/to/best.pt --input frames/ --output out_dir --temporal
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image


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


def discover_frames(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_dir():
        return sorted(p.rglob("*.png"))
    elif p.suffix in (".png", ".jpg", ".jpeg"):
        return [p]
    else:
        raise ValueError(f"Input must be a directory of PNGs or a single image: {input_path}")


def psnr(mse: float) -> float:
    return 10 * np.log10(1.0 / max(mse, 1e-10))


@torch.no_grad()
def encode_decode_sequence(
    encoder,
    predictor,
    decoder,
    entropy_model,
    quantizer,
    frame_paths,
    device,
    image_size,
    temporal: bool,
    output_dir: str,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    recon_dir = out / "recon"
    recon_dir.mkdir(exist_ok=True)

    recon_latents = []
    context_len = getattr(predictor, "context_len", 4)
    bitstream = []
    results = []

    for i, fpath in enumerate(frame_paths):
        img = Image.open(fpath).convert("RGB").resize((image_size, image_size))
        tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        latent, _ = encoder(tensor, return_surprise=True)

        if not temporal or i == 0:
            quant_latent = quantizer(latent)
            recon = decoder(quant_latent)
            rate, _ = entropy_model(quant_latent)
            frame_type = "I"
            if temporal:
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

        bpp = rate.item() / (tensor.shape[2] * tensor.shape[3])
        mse_val = F.mse_loss(recon, tensor).item()

        # Save reconstruction
        recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
        recon_path = recon_dir / f"frame_{i:04d}_{frame_type}.png"
        Image.fromarray(recon_np).save(recon_path)

        bitstream.append(
            {
                "frame": i,
                "type": frame_type,
                "bpp": round(bpp, 4),
                "bits": round(bpp * image_size * image_size, 0),
            }
        )
        results.append(
            {
                "frame": fpath.name,
                "recon": str(recon_path),
                "type": frame_type,
                "psnr": round(psnr(mse_val), 2),
                "bpp": round(bpp, 4),
                "mse": round(mse_val, 6),
            }
        )

        if (i + 1) % 20 == 0:
            print(
                f"  [{i + 1}/{len(frame_paths)}] {frame_type} PSNR={psnr(mse_val):.2f} BPP={bpp:.4f}"
            )

    # Save bitstream log
    total_bits = sum(b["bits"] for b in bitstream)
    bitstream_info = {
        "total_frames": len(bitstream),
        "total_bits": int(total_bits),
        "total_bpp": round(total_bits / (image_size * image_size * len(bitstream)), 4),
        "stream": bitstream,
    }
    (out / "bitstream.json").write_text(json.dumps(bitstream_info, indent=2))

    # Summary stats
    i_frames = [r for r in results if r["type"] == "I"]
    p_frames = [r for r in results if r["type"] == "P"]
    i_bpp_avg = sum(r["bpp"] for r in i_frames) / len(i_frames) if i_frames else 0
    p_bpp_avg = sum(r["bpp"] for r in p_frames) / len(p_frames) if p_frames else 0
    avg_psnr_val = sum(r["psnr"] for r in results) / len(results)

    summary = {
        "mode": "temporal" if temporal else "intra",
        "num_frames": len(results),
        "avg_psnr": round(avg_psnr_val, 2),
        "i_frame_bpp": round(i_bpp_avg, 4),
        "p_frame_bpp": round(p_bpp_avg, 4),
        "p_i_ratio": round(p_bpp_avg / i_bpp_avg, 3) if i_bpp_avg > 0 else 0,
        "total_bits": int(total_bits),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Summary ===")
    print(f"  Mode:     {'temporal' if temporal else 'intra'}")
    print(f"  Frames:   {len(results)}")
    print(f"  PSNR:     {avg_psnr_val:.2f} dB")
    print(f"  I BPP:    {i_bpp_avg:.4f}")
    if temporal:
        print(f"  P BPP:    {p_bpp_avg:.4f}  (P/I ratio: {p_bpp_avg / i_bpp_avg:.3f})")
    print(f"  Total:    {int(total_bits):,} bits")
    print(f"  Reconstructions: {recon_dir}/")
    print(f"  Bitstream log:   {out / 'bitstream.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="LeWM-VC inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--input", required=True, help="Input frames directory or single image")
    parser.add_argument("--output", default="inference_output")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temporal", action="store_true", help="Use IPPP temporal coding")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    encoder, predictor, decoder, entropy_model, quantizer = load_model(
        args.checkpoint, args.config, args.device
    )

    frames = discover_frames(args.input)
    if args.max_frames:
        frames = frames[: args.max_frames]
    print(f"Encoding {len(frames)} frames from {args.input}")

    encode_decode_sequence(
        encoder,
        predictor,
        decoder,
        entropy_model,
        quantizer,
        frames,
        args.device,
        args.image_size,
        temporal=args.temporal,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
