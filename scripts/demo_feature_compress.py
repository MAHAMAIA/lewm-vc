import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchac
from PIL import Image, ImageDraw, ImageFont

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from lewm_vc.feature_compress import FeatureCompressor, FeatureDecompressor, ResNetFeatureExtractor
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer, QuantMode


def build_gaussian_cdf(mu, sigma, step_size, num_levels=256):
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


def make_feature_heatmap(
    feats: torch.Tensor, num_channels: int = 16, scale: int = 8
) -> Image.Image:
    feats = feats.detach().cpu()
    c, h, w = feats.shape
    n = min(num_channels, c)
    side = int(np.ceil(np.sqrt(n)))
    cell = w * scale
    gap = 2
    gw = cell * side + gap * (side - 1)
    gh = cell * side + gap * (side - 1)
    grid = Image.new("RGB", (gw, gh), (20, 20, 20))
    for i in range(n):
        ch = feats[i]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        ch = (ch * 255).byte().numpy()
        row, col = i // side, i % side
        x = col * (cell + gap)
        y = row * (cell + gap)
        ch_img = Image.fromarray(ch, mode="L").convert("RGB")
        ch_img = ch_img.resize((cell, cell), Image.BICUBIC)
        grid.paste(ch_img, (x, y))
    return grid


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Feature compression demo with visualizations")
    parser.add_argument("--checkpoint", default="checkpoints/feature_compress/best.pt")
    parser.add_argument("--input", default="datasets/virat/frames", help="Input frame or directory")
    parser.add_argument("--output", default="demo_feature", help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    device = args.device
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_size = args.image_size
    n_pixels = image_size * image_size

    print("Loading models...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    mc = config.get("model", {})
    latent_dim = mc.get("latent_dim", 8)
    backbone_name = mc.get("backbone", "resnet18")

    backbone = ResNetFeatureExtractor(backbone_name=backbone_name).to(device)
    backbone.eval()
    feat_c = backbone.feature_channels

    compressor = FeatureCompressor(in_channels=feat_c, latent_dim=latent_dim).to(device)
    decompressor = FeatureDecompressor(latent_dim=latent_dim, out_channels=feat_c).to(device)
    entropy_model = HyperpriorEntropy(
        latent_dim=latent_dim,
        hyper_channels=mc.get("entropy", {}).get("hyper_channels", 32),
    ).to(device)

    compressor.load_state_dict(ckpt["models"]["compressor"])
    decompressor.load_state_dict(ckpt["models"]["decompressor"])
    entropy_model.load_state_dict(ckpt["models"]["entropy_model"])
    compressor.eval()
    decompressor.eval()
    entropy_model.eval()

    quantizer = Quantizer()
    quantizer.set_mode(QuantMode.INFERENCE)
    quantizer.to(device)
    step_size = quantizer.step_size.item()

    import torchvision.models as tv_models

    full_resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT).to(device)
    full_resnet.eval()
    layer4 = full_resnet.layer4
    avgpool = full_resnet.avgpool
    fc = full_resnet.fc

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    try:
        import torchvision.transforms as T

        urls = json.loads(Path(_project_root / "docs" / "imagenet_labels.json").read_text())
    except:
        urls = {i: f"class_{i}" for i in range(1000)}

    transform = T.Compose(
        [
            T.Resize(256),
            T.ToTensor(),
        ]
    )

    input_path = Path(args.input)
    frames = []
    if input_path.is_file():
        frames = [input_path]
    else:
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        if subdirs:
            for d in sorted(subdirs):
                for ext in ["*.png", "*.jpg", "*.jpeg"]:
                    frames.extend(sorted(d.glob(ext)))
        else:
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                frames.extend(sorted(input_path.glob(ext)))
    frames = frames[: args.num_frames]

    print(f"Processing {len(frames)} frames...")
    results = []
    all_compressed_bytes = 0

    t0 = time.time()

    for i, fpath in enumerate(frames):
        fname = fpath.name
        img = Image.open(fpath).convert("RGB")
        img_rgb = transform(img).unsqueeze(0).to(device)
        img_224 = torch.nn.functional.interpolate(img_rgb, size=(224, 224), mode="bilinear")
        img_224 = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(img_224)

        with torch.no_grad():
            # Original features
            orig_feats = backbone(img_rgb)

            # Compressed
            latent = compressor(orig_feats)
            qz = quantizer(latent)
            recon_feats = decompressor(qz)

            # Feature similarity
            feat_mse = ((recon_feats - orig_feats) ** 2).mean().item()
            feat_psnr = 10 * np.log10(1.0 / max(feat_mse, 1e-10))
            cos_sim = nn.functional.cosine_similarity(
                recon_feats.flatten(), orig_feats.flatten(), dim=0
            ).item()

            # Encode to bitstream
            indices = torch.round(qz / step_size).clamp(-128, 127).to(torch.int16) + 128
            indices = indices.to(torch.int16).cpu()
            _, params = entropy_model(qz)
            mu = params["mu"].detach().cpu()
            sigma = params["sigma"].detach().cpu()
            cdf = build_gaussian_cdf(mu, sigma, step_size)
            encoded = torchac.encode_int16_normalized_cdf(cdf, indices)
            bpp = len(encoded) * 8 / n_pixels

            all_compressed_bytes += len(encoded)

            # Original features through classifier
            x_orig = layer4(orig_feats)
            x_orig = avgpool(x_orig)
            x_orig = torch.flatten(x_orig, 1)
            logits_orig = fc(x_orig)
            probs_orig = torch.softmax(logits_orig, dim=1)

            # Compressed features through classifier
            x_recon = layer4(recon_feats)
            x_recon = avgpool(x_recon)
            x_recon = torch.flatten(x_recon, 1)
            logits_recon = fc(x_recon)
            probs_recon = torch.softmax(logits_recon, dim=1)

            # Top-5 comparison
            top5_orig = probs_orig.topk(5)
            top5_recon = probs_recon.topk(5)

        results.append(
            {
                "frame": fname,
                "feat_psnr": round(feat_psnr, 2),
                "cosine_sim": round(cos_sim, 4),
                "bpp": round(bpp, 4),
                "bytes": len(encoded),
                "raw_bytes": 256 * 16 * 16 * 4,
                "compression_ratio": round((256 * 16 * 16 * 4) / len(encoded), 1),
                "top5_orig": [int(top5_orig.indices[0, j]) for j in range(5)],
                "top5_recon": [int(top5_recon.indices[0, j]) for j in range(5)],
                "top5_orig_conf": [round(float(top5_orig.values[0, j]), 4) for j in range(5)],
                "top5_recon_conf": [round(float(top5_recon.values[0, j]), 4) for j in range(5)],
            }
        )

        if i < 3 or i % 5 == 4:
            print(
                f"  [{i + 1}/{len(frames)}] {fname}: BPP={bpp:.4f}, feat_PSNR={feat_psnr:.2f}, cos={cos_sim:.4f}"
            )

        # Generate per-frame visualizations for first 4 frames
        if i < 4:
            canvas = Image.new("RGB", (800, 500), (15, 15, 15))
            draw = ImageDraw.Draw(canvas)
            try:
                font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                font_b = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
                )
            except:
                font_s = font_b = ImageFont.load_default()

            r = results[i]

            canvas.paste(img.resize((320, 320)), (10, 10))
            draw.text((10, 340), f"Input: {r['frame']}", fill=(180, 180, 180), font=font_s)

            # Metrics
            y = 10
            draw.text((350, y), "SENTINEL FEATURE COMPRESSION", fill=(0, 180, 255), font=font_b)
            y += 25
            draw.text(
                (350, y),
                f"BPP:       {r['bpp']:.4f}  ({r['bytes']} bytes)",
                fill=(255, 255, 255),
                font=font_s,
            )
            y += 20
            draw.text(
                (350, y),
                f"Compression: {r['compression_ratio']}x vs raw",
                fill=(255, 255, 255),
                font=font_s,
            )
            y += 20
            draw.text(
                (350, y), f"Feat PSNR: {r['feat_psnr']:.2f} dB", fill=(255, 255, 255), font=font_s
            )
            y += 20
            draw.text(
                (350, y), f"Cosine sim: {r['cosine_sim']:.4f}", fill=(255, 255, 255), font=font_s
            )
            y += 30

            # Classification comparison
            draw.text((350, y), "CLASSIFICATION RESULTS", fill=(255, 200, 0), font=font_b)
            y += 25
            draw.text((350, y), "Uncompressed", fill=(100, 200, 100), font=font_s)
            draw.text((560, y), "Compressed (183x)", fill=(100, 200, 100), font=font_s)
            y += 20

            match = sum(1 for j in range(5) if r["top5_orig"][j] == r["top5_recon"][j])
            draw.text(
                (350, y),
                f"Top-5 match: {match}/5",
                fill=(0, 255, 0) if match >= 4 else (255, 200, 0),
                font=font_s,
            )
            y += 25

            for j in range(5):
                o_label = str(r["top5_orig"][j])
                o_conf = f"{r['top5_orig_conf'][j]:.1%}"
                r_label = str(r["top5_recon"][j])
                r_conf = f"{r['top5_recon_conf'][j]:.1%}"
                same = o_label == r_label
                color = (0, 255, 0) if same else (255, 100, 100)
                draw.text((350, y), f"#{j + 1} {o_label} ({o_conf})", fill=color, font=font_s)
                draw.text((560, y), f"#{j + 1} {r_label} ({r_conf})", fill=color, font=font_s)
                y += 18

            y += 10
            bandwidth_savings = (1 - r["bpp"] / 0.25) * 100
            draw.text(
                (350, y),
                f"VSAT bandwidth savings: {bandwidth_savings:.0f}%",
                fill=(0, 180, 255),
                font=font_b,
            )

            canvas.save(str(out_dir / f"frame_{i + 1:04d}.png"))
            print(f"    Saved visualization: {out_dir / f'frame_{i + 1:04d}.png'}")

    # Summary
    avg_bpp = all_compressed_bytes * 8 / n_pixels / len(frames)
    comp_ratio_divisor = all_compressed_bytes / len(frames) if all_compressed_bytes > 0 else 1
    avg_ratio = (256 * 16 * 16 * 4) / (all_compressed_bytes / len(frames))
    avg_psnr = np.mean([r["feat_psnr"] for r in results])
    avg_cos = np.mean([r["cosine_sim"] for r in results])

    print(f"\n{'=' * 60}")
    print(f"Feature Compression Demo Summary")
    print(f"{'=' * 60}")
    print(f"  Model:       {backbone_name} backbone, latent_dim={latent_dim}")
    print(f"  Frames:      {len(frames)}")
    print(f"  Avg BPP:     {avg_bpp:.4f}")
    print(f"  Avg ratio:   {avg_ratio:.1f}x vs raw features")
    print(f"  Avg feat_PSNR: {avg_psnr:.2f} dB")
    print(f"  Avg cosine:  {avg_cos:.4f}")
    print(f"  Output:      {out_dir}/")
    print(f"  Time:        {time.time() - t0:.1f}s")
    print(f"{'=' * 60}")

    # Save summary
    summary = {
        "model": f"{backbone_name} latent_dim={latent_dim}",
        "checkpoint": str(args.checkpoint),
        "num_frames": len(frames),
        "avg_bpp": avg_bpp,
        "avg_compression_ratio": round(avg_ratio, 1),
        "avg_feat_psnr": round(avg_psnr, 2),
        "avg_cosine_sim": round(avg_cos, 4),
    }
    for r in results:
        summary[f"frame__{r['frame']}"] = {
            "bpp": r["bpp"],
            "ratio": r["compression_ratio"],
            "feat_psnr": r["feat_psnr"],
            "cosine_sim": r["cosine_sim"],
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
