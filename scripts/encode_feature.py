import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchac
from PIL import Image

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from lewm_vc.feature_compress import FeatureCompressor, ResNetFeatureExtractor
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Encode frames to feature-compressed bitstream")
    parser.add_argument("--checkpoint", default="checkpoints/feature_compress/best.pt")
    parser.add_argument("--input", default="datasets/virat/frames", help="Input image or directory")
    parser.add_argument("--output", default="bitstream.bin", help="Output bitstream path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-frames", type=int, default=32, help="Max frames to encode")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    device = args.device

    # Load checkpoint
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
    entropy_model = HyperpriorEntropy(
        latent_dim=latent_dim,
        hyper_channels=mc.get("entropy", {}).get("hyper_channels", 32),
    ).to(device)

    compressor.load_state_dict(ckpt["models"]["compressor"])
    entropy_model.load_state_dict(ckpt["models"]["entropy_model"])
    compressor.eval()
    entropy_model.eval()

    quantizer = Quantizer()
    quantizer.set_mode(QuantMode.INFERENCE)
    quantizer.to(device)
    step_size = quantizer.step_size.item()

    # Discover input frames
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

    print(f"Encoding {len(frames)} frames...")
    image_size = args.image_size
    n_pixels = image_size * image_size

    # Encode each frame
    total_bytes = 0
    all_encoded = []
    frame_metadata = []

    t0 = time.time()

    for i, fpath in enumerate(frames):
        img = Image.open(fpath).convert("RGB").resize((image_size, image_size))
        tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        # Features
        feats = backbone(tensor)
        latent = compressor(feats)
        qz = quantizer(latent)

        # Indices for torchac
        indices = torch.round(qz / step_size).clamp(-128, 127).to(torch.int16) + 128
        indices = indices.to(torch.int16).cpu()

        # Entropy params
        _, params = entropy_model(qz)
        mu = params["mu"].detach().cpu()
        sigma = params["sigma"].detach().cpu()
        cdf = build_gaussian_cdf(mu, sigma, step_size)

        # Encode
        encoded = torchac.encode_int16_normalized_cdf(cdf, indices)
        total_bytes += len(encoded)

        bpp = len(encoded) * 8 / n_pixels
        frame_metadata.append(
            {
                "frame": fpath.name,
                "bytes": len(encoded),
                "bpp": round(bpp, 6),
                "shape": list(qz.shape),
            }
        )
        all_encoded.append(encoded)

        if (i + 1) % 10 == 0 or i < 3:
            print(f"  [{i + 1}/{len(frames)}] {fpath.name}: {len(encoded)} bytes, BPP={bpp:.4f}")

    avg_bpp = total_bytes * 8 / n_pixels / len(frames)
    print(f"\n  Total: {len(frames)} frames, {total_bytes} bytes, avg BPP={avg_bpp:.4f}")

    # Write bitstream
    header = {
        "backbone": backbone_name,
        "latent_dim": latent_dim,
        "feature_channels": feat_c,
        "image_size": image_size,
        "num_frames": len(frames),
        "step_size": step_size,
        "frame_metadata": frame_metadata,
    }
    header_bytes = json.dumps(header).encode("utf-8")

    with open(args.output, "wb") as f:
        f.write(struct.pack(">I", len(header_bytes)))
        f.write(header_bytes)
        for enc in all_encoded:
            f.write(struct.pack(">I", len(enc)))
            f.write(enc)

    total_file_size = 4 + len(header_bytes) + sum(4 + len(e) for e in all_encoded)
    print(f"\n  Bitstream saved: {args.output} ({total_file_size} bytes)")
    print(f"  Overhead: {((total_file_size - total_bytes) / total_file_size) * 100:.1f}% (header)")
    print(f"  Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
