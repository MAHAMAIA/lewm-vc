import argparse
import json
import struct
import sys
import time
from pathlib import Path

import torch
import torchac
from PIL import Image

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from lewm_vc.feature_compress import FeatureDecompressor, ResNetFeatureExtractor
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


def load_bitstream(path):
    with open(path, "rb") as f:
        header_len = struct.unpack(">I", f.read(4))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        encoded_frames = []
        for _ in range(header["num_frames"]):
            frame_len = struct.unpack(">I", f.read(4))[0]
            encoded_frames.append(f.read(frame_len))
    return header, encoded_frames


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Decode feature-compressed bitstream")
    parser.add_argument("--checkpoint", default="checkpoints/feature_compress/best.pt")
    parser.add_argument("--input", default="bitstream.bin", help="Input bitstream")
    parser.add_argument("--output", default="recon_features.pt", help="Output features (.pt)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--classify", action="store_true", help="Run classifier on reconstructed features"
    )
    args = parser.parse_args()

    device = args.device

    # Load bitstream header
    header, encoded_frames = load_bitstream(args.input)
    print(f"Bitstream: {header['num_frames']} frames, latent_dim={header['latent_dim']}")
    print(f"  Backbone: {header['backbone']}, image_size={header['image_size']}")

    # Load checkpoint for entropy model + decompressor
    print("Loading models...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    mc = config.get("model", {})

    latent_dim = header["latent_dim"]
    feat_c = header["feature_channels"]
    step_size = header["step_size"]

    decompressor = FeatureDecompressor(latent_dim=latent_dim, out_channels=feat_c).to(device)
    entropy_model = HyperpriorEntropy(
        latent_dim=latent_dim,
        hyper_channels=mc.get("entropy", {}).get("hyper_channels", 32),
    ).to(device)

    decompressor.load_state_dict(ckpt["models"]["decompressor"])
    entropy_model.load_state_dict(ckpt["models"]["entropy_model"])
    decompressor.eval()
    entropy_model.eval()

    # Restore the backbone for the classifier (optional)
    classifier = None
    if args.classify:
        import torchvision.models as tv_models

        backbone = ResNetFeatureExtractor(backbone_name=header["backbone"]).to(device)
        full_resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT).to(device)
        full_resnet.eval()
        classifier = {
            "layer4": full_resnet.layer4,
            "avgpool": full_resnet.avgpool,
            "fc": full_resnet.fc,
        }

    # Decode frames
    print(f"Decoding {len(encoded_frames)} frames...")
    t0 = time.time()
    all_features = []

    for i, encoded in enumerate(encoded_frames):
        meta = header["frame_metadata"][i]
        shape = meta["shape"]

        # Create dummy input for entropy model to get mu, sigma
        dummy = torch.zeros(1, latent_dim, shape[2], shape[3], device=device)
        _, params = entropy_model(dummy)
        mu = params["mu"].detach().cpu()
        sigma = params["sigma"].detach().cpu()
        cdf = build_gaussian_cdf(mu, sigma, step_size)

        # Decode
        decoded = torchac.decode_int16_normalized_cdf(cdf, encoded)
        qz = (decoded.float() - 128) * step_size
        qz = qz.reshape(1, latent_dim, shape[2], shape[3]).to(device)

        # Decompress to features
        feats = decompressor(qz)
        all_features.append(feats.cpu())

        if (i + 1) % 10 == 0 or i < 3:
            print(f"  [{i + 1}/{len(encoded_frames)}] features={list(feats.shape)}")

    # Save
    torch.save({"features": torch.cat(all_features), "header": header}, args.output)
    print(f"\n  Saved: {args.output} ({torch.cat(all_features).shape})")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Classify (optional)
    if classifier:
        import torchvision.transforms as T

        print("\n  Running classifier on reconstructed features...")
        transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        for i in range(min(5, len(all_features))):
            feats = all_features[i].to(device)
            x = classifier["layer4"](feats)
            x = classifier["avgpool"](x)
            x = torch.flatten(x, 1)
            logits = classifier["fc"](x)
            pred = logits.argmax(dim=1).item()
            print(f"    Frame {i}: class={pred}")


if __name__ == "__main__":
    main()
