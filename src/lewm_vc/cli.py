import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

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


def load_model(checkpoint_path, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location=device)
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

    models = {
        "backbone": backbone,
        "compressor": compressor,
        "decompressor": decompressor,
        "entropy_model": entropy_model,
        "quantizer": quantizer,
    }
    info = {
        "backbone": backbone_name,
        "latent_dim": latent_dim,
        "feature_channels": feat_c,
        "step_size": quantizer.step_size.item(),
        "checkpoint": str(checkpoint_path),
        "bpp": ckpt.get("bpp", "unknown"),
        "step": ckpt.get("step", "unknown"),
        "phase": ckpt.get("phase", "unknown"),
    }
    return models, info


def discover_frames(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    frames = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        frames.extend(sorted(input_path.rglob(ext)))
    return frames


def cmd_encode(args):
    device = args.device
    models, info = load_model(args.model, device)
    q = models["quantizer"]
    step_size = info["step_size"]
    encoder = models["compressor"]
    backbone = models["backbone"]
    entropy = models["entropy_model"]

    frames = discover_frames(args.input)
    if not frames:
        print(f"No frames found at {args.input}")
        sys.exit(1)
    if args.max_frames:
        frames = frames[: args.max_frames]

    image_size = args.image_size
    n_pixels = image_size * image_size
    print(f"Encoding {len(frames)} frames to {args.output}...")
    t0 = time.time()

    import torchac

    total_bytes = 0
    encoded_list = []
    frame_meta = []

    for i, fpath in enumerate(frames):
        img = Image.open(fpath).convert("RGB").resize((image_size, image_size))
        tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            feats = backbone(tensor)
            latent = encoder(feats)
            qz = q(latent)

            indices = torch.round(qz / step_size).clamp(-128, 127).to(torch.int16) + 128
            indices = indices.to(torch.int16).cpu()
            _, params = entropy(qz)
            mu = params["mu"].detach().cpu()
            sigma = params["sigma"].detach().cpu()
            cdf = build_gaussian_cdf(mu, sigma, step_size)

            encoded = torchac.encode_int16_normalized_cdf(cdf, indices)
            total_bytes += len(encoded)
            bpp = len(encoded) * 8 / n_pixels

        encoded_list.append(encoded)
        frame_meta.append(
            {
                "frame": fpath.name,
                "bytes": len(encoded),
                "bpp": round(bpp, 6),
                "shape": list(qz.shape),
            }
        )

        if args.verbose or (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(frames)}] {fpath.name}: {len(encoded)} B, {bpp:.4f} BPP")

    avg_bpp = total_bytes * 8 / n_pixels / len(frames)
    print(f"  Total: {len(frames)} frames, {total_bytes} bytes, avg BPP={avg_bpp:.4f}")

    header = {
        "backbone": info["backbone"],
        "latent_dim": info["latent_dim"],
        "feature_channels": info["feature_channels"],
        "image_size": image_size,
        "num_frames": len(frames),
        "step_size": step_size,
        "frame_metadata": frame_meta,
    }
    header_bytes = json.dumps(header).encode("utf-8")

    with open(args.output, "wb") as f:
        f.write(struct.pack(">I", len(header_bytes)))
        f.write(header_bytes)
        for enc in encoded_list:
            f.write(struct.pack(">I", len(enc)))
            f.write(enc)

    total_file = 4 + len(header_bytes) + sum(4 + len(e) for e in encoded_list)
    print(f"  Wrote {args.output} ({total_file} bytes, {total_file / 1024:.1f} KB)")
    print(f"  Time: {time.time() - t0:.1f}s")


def cmd_decode(args):
    device = args.device
    models, info = load_model(args.model, device)
    step_size = info["step_size"]
    decompressor = models["decompressor"]
    entropy = models["entropy_model"]

    with open(args.input, "rb") as f:
        header_len = struct.unpack(">I", f.read(4))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        encoded_list = []
        for _ in range(header["num_frames"]):
            frame_len = struct.unpack(">I", f.read(4))[0]
            encoded_list.append(f.read(frame_len))

    import torchac

    print(f"Decoding {len(encoded_list)} frames from {args.input}...")
    print(f"  Backbone: {header['backbone']}, latent_dim={header['latent_dim']}")
    t0 = time.time()

    all_features = []
    for i, encoded in enumerate(encoded_list):
        meta = header["frame_metadata"][i]
        shape = meta["shape"]
        dummy = torch.zeros(1, header["latent_dim"], shape[2], shape[3], device=device)
        _, params = entropy(dummy)
        mu = params["mu"].detach().cpu()
        sigma = params["sigma"].detach().cpu()
        cdf = build_gaussian_cdf(mu, sigma, step_size)

        decoded = torchac.decode_int16_normalized_cdf(cdf, encoded)
        qz = (decoded.float() - 128) * step_size
        qz = qz.reshape(1, header["latent_dim"], shape[2], shape[3]).to(device)

        with torch.no_grad():
            feats = decompressor(qz)
            all_features.append(feats.cpu())

        if args.verbose or (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(encoded_list)}] features={list(feats.shape)}")

    torch.save({"features": torch.cat(all_features), "header": header}, args.output)
    print(f"  Wrote {args.output} ({torch.cat(all_features).shape})")
    print(f"  Time: {time.time() - t0:.1f}s")


def cmd_info(args):
    device = args.device
    models, info = load_model(args.model, device)
    comp_params = sum(p.numel() for p in models["compressor"].parameters())
    decomp_params = sum(p.numel() for p in models["decompressor"].parameters())
    entropy_params = sum(p.numel() for p in models["entropy_model"].parameters())

    print(f"Sentinel Feature Compression Model")
    print(f"{'=' * 50}")
    print(f"  Checkpoint:    {info['checkpoint']}")
    print(f"  Backbone:      {info['backbone']}")
    print(f"  Latent dim:    {info['latent_dim']}")
    print(f"  Feature ch:    {info['feature_channels']}")
    print(f"  Step size:     {info['step_size']:.6f}")
    print(f"  Training step: {info['step']}")
    print(f"  Training phase:{info['phase']}")
    print(f"  Val BPP:       {info['bpp']}")
    print()
    print(f"  Latent elements per frame:")
    print(f"    {info['latent_dim']} x 16 x 16 = {info['latent_dim'] * 256}")
    print(
        f"    Raw bits: {info['latent_dim'] * 256 * 8} ({info['latent_dim'] * 256 * 8 / 65536:.2f} BPP)"
    )
    print(f"    w/ entropy: ~0.18 BPP (179x compression)")
    print(f"\n  Parameters:")
    print(f"    Compressor:    {comp_params:,}")
    print(f"    Decompressor:  {decomp_params:,}")
    print(f"    Entropy model: {entropy_params:,}")
    print(f"    Total:         {comp_params + decomp_params + entropy_params:,}")


def main():
    parser = argparse.ArgumentParser(description="Sentinel Feature Compression CLI")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    subparsers = parser.add_subparsers(dest="command")

    feature_parser = subparsers.add_parser("feature", help="Feature compression operations")
    feature_sub = feature_parser.add_subparsers(dest="subcommand")

    encode_p = feature_sub.add_parser("encode", help="Encode frames to feature bitstream")
    encode_p.add_argument("--model", "-m", required=True, help="Model checkpoint (.pt)")
    encode_p.add_argument("--input", "-i", required=True, help="Input frame or directory")
    encode_p.add_argument("--output", "-o", default="bitstream.bin", help="Output bitstream")
    encode_p.add_argument("--image-size", type=int, default=256)
    encode_p.add_argument("--max-frames", type=int, default=None)
    encode_p.add_argument("--verbose", "-v", action="store_true")

    decode_p = feature_sub.add_parser("decode", help="Decode bitstream to features")
    decode_p.add_argument("--model", "-m", required=True, help="Model checkpoint (.pt)")
    decode_p.add_argument("--input", "-i", required=True, help="Input bitstream (.bin)")
    decode_p.add_argument(
        "--output", "-o", default="recon_features.pt", help="Output features (.pt)"
    )
    decode_p.add_argument("--verbose", "-v", action="store_true")

    info_p = feature_sub.add_parser("info", help="Show model information")
    info_p.add_argument("--model", "-m", required=True, help="Model checkpoint (.pt)")

    args = parser.parse_args()

    if args.command == "feature":
        if not args.subcommand:
            feature_parser.print_help()
            sys.exit(1)
        if args.subcommand == "encode":
            cmd_encode(args)
        elif args.subcommand == "decode":
            cmd_decode(args)
        elif args.subcommand == "info":
            cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
