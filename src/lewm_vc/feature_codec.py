#!/usr/bin/env python3
"""
Feature-domain video codec CLI — pilot deployment tool.

Encodes ResNet50-FPN P4 feature maps using v3 deep compressor (tanh),
frame-differencing predictor, and separate intra/residual entropy models.

The checkpoint is a single .pt file containing:
    compressor       — DeepCompressor state dict
    decompressor     — DeepDecompressor state dict
    entropy_model    — HyperpriorEntropy (intra frames)
    residual_entropy — HyperpriorEntropy (frame-diff residuals, optional)

Usage:
    python -m lewm_vc.feature_codec encode -m model.pt -i frames/ -o stream.bin
    python -m lewm_vc.feature_codec decode -m model.pt -i stream.bin -o recon.pt
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.feature_compress import DeepCompressor, DeepDecompressor
from lewm_vc.quant import QuantMode, Quantizer

STEP_SIZE = 2.0 / 256


# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def _build_gaussian_cdf(mu, sigma, num_levels=256):
    """Build quantised Gaussian CDF tables used by torchac."""
    B, C, H, W = mu.shape
    device = mu.device
    half = num_levels // 2
    offsets = (
        torch.arange(0, num_levels + 1, dtype=torch.float32, device=device) - half - 0.5
    ) * STEP_SIZE
    z = (offsets - mu.unsqueeze(-1)) / sigma.unsqueeze(-1)
    cdf = torch.special.ndtr(z)
    cdf_min = cdf[..., 0:1]
    cdf_max = cdf[..., -1:]
    cdf = (cdf - cdf_min) / (cdf_max - cdf_min + 1e-10)
    cdf[..., 0] = 0.0
    cdf[..., -1] = 1.0
    cdf_int = (cdf * 65535 + 0.5).to(torch.int32)
    cdf_int = torch.where(cdf_int > 32767, cdf_int - 65536, cdf_int).to(torch.int16)
    cdf_int[..., -1] = -1
    return cdf_int


# ═══════════════════════════════════════════════════════════════════
# model loading
# ═══════════════════════════════════════════════════════════════════


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device)

    w = ckpt["compressor"]["head.weight"]
    latent_dim = w.shape[0]
    in_ch = ckpt["compressor"]["stem.weight"].shape[1]
    mid_ch = ckpt["compressor"]["stem.weight"].shape[0]

    # Infer hyper_channels from any entropy-model Conv2d key
    hc = 32
    for k in ckpt.get("entropy_model", {}):
        if hasattr(ckpt["entropy_model"][k], "shape") and len(ckpt["entropy_model"][k].shape) == 4:
            hc = ckpt["entropy_model"][k].shape[0]
            break

    compressor = DeepCompressor(in_ch, latent_dim, mid_ch).to(device)
    decompressor = DeepDecompressor(latent_dim, in_ch, mid_ch).to(device)
    entropy_intra = HyperpriorEntropy(latent_dim, hc).to(device)

    compressor.load_state_dict(ckpt["compressor"])
    decompressor.load_state_dict(ckpt["decompressor"])
    entropy_intra.load_state_dict(ckpt["entropy_model"])

    residual = None
    if "residual_entropy" in ckpt:
        residual = HyperpriorEntropy(latent_dim, hc).to(device)
        residual.load_state_dict(ckpt["residual_entropy"])

    compressor.eval()
    decompressor.eval()
    entropy_intra.eval()
    if residual:
        residual.eval()

    return compressor, decompressor, entropy_intra, residual


# ═══════════════════════════════════════════════════════════════════
# encode
# ═══════════════════════════════════════════════════════════════════


def cmd_encode(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    compressor, decompressor, entropy_intra, entropy_residual = load_checkpoint(args.model, device)
    quantizer = Quantizer(mode=QuantMode.INFERENCE).to(device)

    # FPN backbone
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "..", "plans", "mi300x-training-sprint")
    )
    from fpn_backbone import ResNet50FPN

    backbone = ResNet50FPN().to(device).eval()

    import torchac

    # find frames
    input_dir = Path(args.input)
    frames = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if args.max_frames:
        frames = frames[: args.max_frames]
    print(f"Frames: {len(frames)}  image_size={args.image_size}  intra_period={args.intra_period}")

    encoded_list = []  # per-frame rANS byte blobs
    frame_meta = []  # per-frame metadata for header

    prev_latent = None
    total_bytes = 0
    t0 = time.time()

    for idx, fp in enumerate(frames):
        img = _load_image(fp, args.image_size).unsqueeze(0).to(device)

        with torch.no_grad():
            pyramid = backbone(img)
            feat = pyramid["P4"]  # [1, 256, 16, 16]

        latent = compressor(feat)  # [1, 16, 16, 16]  in [-1,1]

        is_iframe = (prev_latent is None) or (idx % args.intra_period == 0)

        if is_iframe:
            q = quantizer(latent)
            _, p = entropy_intra(q)
            frame_type = "I"
        else:
            residual = latent - prev_latent
            q = quantizer(residual)
            ent = entropy_residual if entropy_residual else entropy_intra
            _, p = ent(q)
            frame_type = "P"
            prev_latent = latent  # decoder reconstructs ẑ_t = prev_latent + qr

        # symbols for arithmetic coder: round to step, shift to [0,255]
        symbols = torch.round(q / STEP_SIZE).to(torch.int32) + 128
        symbols = symbols.clamp(0, 255).cpu().to(torch.int16)

        cdf = _build_gaussian_cdf(p["mu"], p["sigma"])

        blob = torchac.encode_int16_normalized_cdf(cdf, symbols)
        nb = len(blob)
        total_bytes += nb
        encoded_list.append(blob)

        frame_meta.append(
            {
                "type": frame_type,
                "file": fp.name,
                "bytes": nb,
                "bpp": nb * 8 / (args.image_size * args.image_size * 3),
            }
        )

        if (idx + 1) % 50 == 0 or idx == len(frames) - 1:
            elapsed = time.time() - t0
            fps = (idx + 1) / max(elapsed, 0.001)
            print(f"  frame {idx + 1:4d}/{len(frames)}  {frame_type}  {nb:6d} B  {fps:5.1f} fps")

        prev_latent = latent

    elapsed = time.time() - t0
    n_pixels = args.image_size * args.image_size
    avg_bpp = total_bytes * 8 / n_pixels / len(frames)

    # ── write bitstream ─────────────────────────────────────────────
    header = {
        "image_size": args.image_size,
        "num_frames": len(frames),
        "step_size": STEP_SIZE,
        "frame_meta": frame_meta,
    }
    header_bytes = json.dumps(header).encode("utf-8")

    with open(args.output, "wb") as f:
        f.write(struct.pack(">I", len(header_bytes)))
        f.write(header_bytes)
        for blob in encoded_list:
            f.write(struct.pack(">I", len(blob)))
            f.write(blob)

    print(
        f"\n  Encoded {len(frames)} frames in {elapsed:.1f}s ({len(frames) / max(elapsed, 0.001):.1f} fps)"
    )
    print(
        f"  Total: {total_bytes:,} B  BPP(image)={avg_bpp:.4f}  "
        f"BPP(feature)={total_bytes * 8 / (len(frames) * 256):.2f}"
    )
    print(f"  Written: {args.output} ({os.path.getsize(args.output):,} B)")


# ═══════════════════════════════════════════════════════════════════
# decode
# ═══════════════════════════════════════════════════════════════════


def cmd_decode(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    compressor, decompressor, entropy_intra, entropy_residual = load_checkpoint(args.model, device)

    import torchac

    # ── read header ──────────────────────────────────────────────────
    with open(args.input, "rb") as f:
        hdr_len = struct.unpack(">I", f.read(4))[0]
        header = json.loads(f.read(hdr_len).decode("utf-8"))
        meta = header["frame_meta"]
        blobs = []
        for _ in range(header["num_frames"]):
            frm_len = struct.unpack(">I", f.read(4))[0]
            blobs.append(f.read(frm_len))

    step = header["step_size"]
    print(f"Decoding {len(blobs)} frames…  image_size={header['image_size']}  step={step:.6f}")
    t0 = time.time()

    # We need a compressor for the decode path (to reconstruct prev_latent).
    # FPN backbone not loaded — decoder only needs decompressor + entropy.

    all_features = []
    prev_latent = None

    for i, blob in enumerate(blobs):
        frame_type = meta[i]["type"]
        # Re-run entropy model to get CDF tables (uses the dummy-latent trick)
        latent_dim = len(ckpt_load_latent_dim(args.model, device))
        dummy = torch.zeros(1, latent_dim, 16, 16, device=device)

        if frame_type == "I":
            _, p = entropy_intra(dummy)
        else:
            ent = entropy_residual if entropy_residual else entropy_intra
            _, p = ent(dummy)

        cdf = _build_gaussian_cdf(p["mu"], p["sigma"])
        syms = torchac.decode_int16_normalized_cdf(cdf, blob)
        q = (syms.float().to(device) - 128) * step
        q = q.reshape(1, latent_dim, 16, 16)

        with torch.no_grad():
            if frame_type == "I" or prev_latent is None:
                feats = decompressor(q)
                # recompute latent for next frame prediction
                prev_latent = compressor(feats)
            else:
                reconstructed_latent = prev_latent + q
                feats = decompressor(reconstructed_latent)
                prev_latent = compressor(feats)

        all_features.append(feats.cpu())

    elapsed = time.time() - t0
    stacked = torch.cat(all_features)  # [N, 256, 16, 16]
    torch.save({"features": stacked, "header": header}, args.output)
    print(f"  Decoded {stacked.shape[0]} frames → {args.output} ({elapsed:.1f}s)")


def ckpt_load_latent_dim(path, device):
    """Quick read of latent_dim without loading full model."""
    ckpt = torch.load(path, map_location=device)
    return ckpt["compressor"]["head.weight"].shape[0]


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Feature-domain video codec (Deep + FPN)")
    sub = parser.add_subparsers(dest="command")

    enc = sub.add_parser("encode", help="Frames → bitstream")
    enc.add_argument("-m", "--model", required=True, help="Checkpoint path (.pt)")
    enc.add_argument("-i", "--input", required=True, help="Directory of frames")
    enc.add_argument("-o", "--output", default="stream.bin", help="Output bitstream")
    enc.add_argument("--image-size", type=int, default=256)
    enc.add_argument("--intra-period", type=int, default=4, help="I-frame interval")
    enc.add_argument("--max-frames", type=int, default=None)

    dec = sub.add_parser("decode", help="Bitstream → features")
    dec.add_argument("-m", "--model", required=True, help="Checkpoint path (.pt)")
    dec.add_argument("-i", "--input", required=True, help="Input bitstream")
    dec.add_argument("-o", "--output", default="recon_features.pt", help="Output .pt file")

    args = parser.parse_args()
    if args.command == "encode":
        cmd_encode(args)
    elif args.command == "decode":
        cmd_decode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
