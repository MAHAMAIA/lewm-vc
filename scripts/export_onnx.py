"""
Export LeWM-VC models to ONNX format for TensorRT deployment.

Exports each module independently:
  - encoder: [1, 3, 256, 256] → [1, 192, 16, 16] without surprise output
  - predictor: [1, 4, 192, 16, 16] → [1, 192, 16, 16] mean + std
  - decoder: [1, 192, 16, 16] → [1, 3, 256, 256]
  - entropy_model: [1, 192, 16, 16] → [1, 384, 16, 16] (mu + log_scale + log_weight)

Usage:
    python scripts/export_onnx.py --checkpoint path/to/best.pt --output-dir onnx_models
    python scripts/export_onnx.py --checkpoint path/to/best.pt --output-dir onnx_models --image-size 256
"""

import argparse
import sys
from pathlib import Path

import torch


def resolve_checkpoint(checkpoint_path: str | None) -> str:
    """Find a checkpoint: either explicit or auto-discover the newest best.pt."""
    if checkpoint_path:
        return checkpoint_path
    base = Path("checkpoints")
    if not base.exists():
        print("No checkpoint specified and no checkpoints/ directory found.")
        sys.exit(1)
    # Walk run directories looking for best.pt, newest first
    candidates = sorted(base.iterdir(), reverse=True)
    for run_dir in candidates:
        for sub in ["lambda_0.05", "lambda_0.02", "lambda_0.08", ""]:
            best = run_dir / sub / "best.pt"
            if best.exists():
                print(f"Auto-discovered: {best}")
                return str(best)
    print("No best.pt found in checkpoints/ — using untrained model")
    return ""


def build_models(config_path: str | None):
    """Build model instances from config or defaults."""
    import yaml

    if config_path and Path(config_path).exists():
        config = yaml.safe_load(open(config_path))
        model_cfg = config.get("model", {})
    else:
        model_cfg = {}

    latent_dim = model_cfg.get("latent_dim", 192)
    patch_size = model_cfg.get("patch_size", 16)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy

    encoder = LeWMEncoder(
        latent_dim=latent_dim,
        patch_size=patch_size,
        hidden_dim=model_cfg.get("encoder", {}).get("hidden_dim", 192),
        num_layers=model_cfg.get("encoder", {}).get("num_layers", 6),
        num_heads=model_cfg.get("encoder", {}).get("num_heads", 3),
        semantic_surprise=True,
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
    return encoder, predictor, decoder, entropy_model


def export_encoder(
    encoder: torch.nn.Module,
    output_dir: Path,
    image_size: int,
    latent_dim: int,
    device: str,
):
    """Export encoder to ONNX.

    Input:  [1, 3, H, W]  — RGB float32, values in [0, 1]
    Output: [1, D, H/16, W/16] — latent tensor (no surprise)
    """
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    encoder.to(device).eval()

    class EncoderWrapper(torch.nn.Module):
        """Wraps encoder forward, discarding the optional surprise output."""

        def forward(self, x):
            return encoder(x, return_surprise=True)[0]

    wrapped = EncoderWrapper()
    path = output_dir / "encoder.onnx"
    torch.onnx.export(
        wrapped,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["latent"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  encoder: {path}")


def export_predictor(
    predictor: torch.nn.Module,
    output_dir: Path,
    latent_dim: int,
    context_len: int,
    h_latent: int,
    w_latent: int,
    device: str,
):
    """Export predictor to ONNX.

    Input:  [1, context_len, D, H_lat, W_lat] — context frames stacked
    Output: mean [1, D, H_lat, W_lat], std [1, D, H_lat, W_lat]
    """
    dummy_ctx = torch.randn(1, context_len, latent_dim, h_latent, w_latent).to(device)
    predictor.to(device).eval()

    class PredictorWrapper(torch.nn.Module):
        def forward(self, ctx):
            # Split [1, T, D, H, W] back into list of [1, D, H, W]
            ctx_list = [ctx[:, i] for i in range(ctx.shape[1])]
            mean, std = predictor(ctx_list)
            return mean, std

    wrapped = PredictorWrapper()
    path = output_dir / "predictor.onnx"
    torch.onnx.export(
        wrapped,
        dummy_ctx,
        str(path),
        input_names=["context"],
        output_names=["mean", "std"],
        dynamic_axes={"context": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  predictor: {path}")


def export_decoder(
    decoder: torch.nn.Module,
    output_dir: Path,
    latent_dim: int,
    image_size: int,
    device: str,
):
    """Export decoder to ONNX.

    Input:  [1, D, H/16, W/16] — quantized latent or recon_latent
    Output: [1, 3, H, W] — reconstructed RGB frame
    """
    dummy = torch.randn(1, latent_dim, image_size // 16, image_size // 16).to(device)
    decoder.to(device).eval()

    path = output_dir / "decoder.onnx"
    torch.onnx.export(
        decoder,
        dummy,
        str(path),
        input_names=["latent"],
        output_names=["reconstruction"],
        dynamic_axes={"latent": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  decoder: {path}")


def export_entropy(
    entropy_model: torch.nn.Module,
    output_dir: Path,
    latent_dim: int,
    h_latent: int,
    w_latent: int,
    device: str,
):
    """Export entropy model to ONNX.

    Input:  [1, D, H/16, W/16] — quantized latent or residual
    Output: [1, 3*D, H/16, W/16] — concatenated mu, log_scale, log_weight
    """
    dummy = torch.randn(1, latent_dim, h_latent, w_latent).to(device)
    entropy_model.to(device).eval()

    class EntropyWrapper(torch.nn.Module):
        def forward(self, x):
            mu, scale, log_weight = entropy_model(x)
            return torch.cat([mu, scale, log_weight], dim=1)

    wrapped = EntropyWrapper()
    path = output_dir / "entropy_model.onnx"
    torch.onnx.export(
        wrapped,
        dummy,
        str(path),
        input_names=["quantized"],
        output_names=["entropy_params"],
        dynamic_axes={"quantized": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  entropy_model: {path}")


def main():
    parser = argparse.ArgumentParser(description="Export LeWM-VC models to ONNX")
    parser.add_argument("--checkpoint", default=None, help="Path to best.pt")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--output-dir", default="onnx_models", help="Output directory")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"Exporting models to {output_dir} on {device}")
    encoder, predictor, decoder, entropy_model = build_models(args.config)

    # Load checkpoint if available
    ckpt_path = resolve_checkpoint(args.checkpoint)
    if ckpt_path:
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
        print("WARNING: no checkpoint loaded — exporting untrained model")

    h_lat = args.image_size // 16
    w_lat = args.image_size // 16
    context_len = getattr(predictor, "context_len", 4)

    print("\nExporting encoder...")
    export_encoder(encoder, output_dir, args.image_size, encoder.latent_dim, device)

    print("Exporting predictor...")
    export_predictor(predictor, output_dir, encoder.latent_dim, context_len, h_lat, w_lat, device)

    print("Exporting decoder...")
    export_decoder(decoder, output_dir, encoder.latent_dim, args.image_size, device)

    print("Exporting entropy_model...")
    export_entropy(entropy_model, output_dir, encoder.latent_dim, h_lat, w_lat, device)

    print(f"\nDone. {len(list(output_dir.glob('*.onnx')))} ONNX files in {output_dir}")
    for p in sorted(output_dir.glob("*.onnx")):
        size_mb = p.stat().st_size / 1_000_000
        print(f"  {p.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
