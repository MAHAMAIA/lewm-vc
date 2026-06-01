import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from lewm_vc.feature_compress import FeatureCompressor, FeatureDecompressor, ResNetFeatureExtractor
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer, QuantMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def nll_rate_loss(latents, mu, sigma, step_size):
    half = step_size / 2.0
    prob = torch.special.ndtr((latents + half - mu) / sigma) - torch.special.ndtr(
        (latents - half - mu) / sigma
    )
    prob = prob.clamp(min=1e-10)
    nats = -torch.log(prob)
    bits = nats / 0.693147
    return bits.sum() / latents.shape[0]


def find_images(root: Path, max_images: int):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    # Check if subdirs (ImageNet format)
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if subdirs:
        per_dir = max(1, max_images // len(subdirs))
        for d in sorted(subdirs):
            for ext in exts:
                images.extend(d.glob(ext))
            if len(images) >= max_images:
                break
        images = sorted(images[:max_images])
    else:
        for ext in exts:
            images.extend(root.glob(ext))
        images = sorted(images[:max_images])
    return images


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate feature compression quality")
    parser.add_argument("--checkpoint", default="checkpoints/feature_compress/best.pt")
    parser.add_argument("--data", default="datasets/virat/frames")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=500)
    args = parser.parse_args()

    device = args.device

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    mc = config.get("model", {})
    latent_dim = mc.get("latent_dim", 8)
    backbone_name = mc.get("backbone", "resnet18")

    print(f"  Backbone: {backbone_name}, latent_dim={latent_dim}")

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

    # Load images
    data_root = Path(args.data)
    if not data_root.is_dir():
        print(f"Data root not found: {data_root}")
        sys.exit(1)

    images = find_images(data_root, args.max_images)
    print(f"  Found {len(images)} images")

    transform = T.Compose(
        [
            T.Resize(256),
            T.ToTensor(),
        ]
    )

    # Metrics
    total_feat_mse = 0.0
    total_cosine = 0.0
    total_nll_bits = 0.0
    total_pixels = 0
    n_pixels = 256 * 256
    count = 0
    batch_size = 32

    print(f"\nEvaluating on {len(images)} images...")
    t0 = time.time()

    for i in range(0, len(images), batch_size):
        batch_paths = images[i : i + batch_size]
        batch = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch.append(transform(img))
        imgs = torch.stack(batch).to(device)

        # Forward
        feats = backbone(imgs)
        latent = compressor(feats)
        qz = quantizer(latent)
        recon = decompressor(qz)

        # Feature MSE / PSNR
        feat_mse = ((recon - feats) ** 2).mean(dim=(1, 2, 3))
        total_feat_mse += feat_mse.sum().item()

        # Cosine similarity (per-sample mean)
        cos = nn.functional.cosine_similarity(recon.flatten(1), feats.flatten(1))
        total_cosine += cos.sum().item()

        # NLL rate
        _, params = entropy_model(qz)
        rate = nll_rate_loss(qz, params["mu"], params["sigma"], step_size)
        total_nll_bits += rate.item() * imgs.shape[0]

        count += imgs.shape[0]

        if (i // batch_size) % 10 == 0:
            print(f"  [{i}/{len(images)}]  feat_PSNR={10 * torch.log10(1.0 / feat_mse.mean()):.2f}")

    avg_mse = total_feat_mse / count
    avg_psnr = 10 * torch.log10(torch.tensor(1.0 / avg_mse)).item()
    avg_cos = total_cosine / count
    avg_bpp = total_nll_bits / count / n_pixels

    print(f"\n{'=' * 60}")
    print(f"Results ({Path(args.checkpoint).name})")
    print(f"Step {ckpt.get('step', '?')}, Phase {ckpt.get('phase', '?')}")
    print(f"{'=' * 60}")
    print(f"  Feature PSNR:   {avg_psnr:.2f} dB")
    print(f"  Feature MSE:    {avg_mse:.6f}")
    print(f"  Cosine sim:     {avg_cos:.4f}")
    print(f"  NLL BPP:        {avg_bpp:.4f}")
    print(f"  Latent elements: {latent_dim} x 16 x 16 = {latent_dim * 16 * 16}")
    print(f"  Images:         {count}")
    print(f"  Time:           {time.time() - t0:.0f}s")
    print(f"{'=' * 60}")

    # Interpret
    if avg_bpp <= 0.25:
        print("  TARGET: BPP within range (0.08-0.25) for satellite backhaul")
    elif avg_bpp <= 0.5:
        print("  CLOSE: BPP near target range, more training should bring it down")
    else:
        print(f"  HIGH: BPP={avg_bpp:.4f} above target, needs more training or smaller latent_dim")

    if avg_psnr >= 20:
        print("  GOOD: Feature fidelity sufficient for detection downstream")
    elif avg_psnr >= 15:
        print("  OK: Feature fidelity acceptable, detection may degrade slightly")
    else:
        print("  LOW: Feature fidelity may impact detection accuracy")


if __name__ == "__main__":
    main()
