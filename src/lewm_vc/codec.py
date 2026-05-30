"""
LeWM-VC Codec Interface — Python bridge for FFmpeg C plugin.

Provides get_encoder() and get_decoder() functions called from
FFmpeg's C plugin (ffmpeg/lewm_vc_encoder.c) via embedded Python.

The C plugin imports this module via Python.h, calls get_encoder() to
obtain an encoder object, and calls encoder.encode() per frame.

Usage (from C / FFmpeg plugin):
    import lewm_vc.codec
    enc = lewm_vc.codec.get_encoder("lewmvc")
    bitstream = enc.encode(width, height, yuv420_bytes)
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Lazy-loaded model singleton
_ENCODER = None
_DECODER = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CHECKPOINT_DIRS = [
    "checkpoints",
    "/workspace/le-maia/checkpoints",
]
_H, _W = 256, 256


def _find_best_checkpoint() -> str | None:
    """Find the most recent best.pt in checkpoint directories."""
    for base in _CHECKPOINT_DIRS:
        p = Path(base)
        if not p.exists():
            continue
        for run_dir in sorted(p.iterdir(), reverse=True):
            for lbda_dir in ["lambda_0.05", "lambda_0.02", "lambda_0.08"]:
                best = run_dir / lbda_dir / "best.pt"
                if best.exists():
                    return str(best)
            best = run_dir / "best.pt"
            if best.exists():
                return str(best)
            step_files = sorted(run_dir.rglob("step_*.pt"), reverse=True)
            if step_files:
                return str(step_files[0])
    return None


def _build_models(latent_dim=192):
    """Build and load models from the best available checkpoint."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lewm_vc import LeWMDecoder, LeWMEncoder, LeWMPredictor
    from lewm_vc.entropy import HyperpriorEntropy
    from lewm_vc.quant import Quantizer

    model_cfg = {}
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

    ckpt_path = _find_best_checkpoint()
    if ckpt_path:
        print(f"[lewm_vc.codec] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=_DEVICE, weights_only=False)
        model_state = ckpt.get("models", ckpt)
        for name, m in [
            ("encoder", encoder),
            ("predictor", predictor),
            ("decoder", decoder),
            ("entropy_model", entropy_model),
            ("quantizer", quantizer),
        ]:
            sd = model_state.get(name)
            if sd is not None:
                m.load_state_dict(sd, strict=False)
    else:
        print("[lewm_vc.codec] WARNING: no checkpoint found, using untrained model")

    for m in [encoder, predictor, decoder, entropy_model, quantizer]:
        m.to(_DEVICE)
        m.eval()

    return {
        "encoder": encoder,
        "predictor": predictor,
        "decoder": decoder,
        "entropy_model": entropy_model,
        "quantizer": quantizer,
    }


def _rgb_to_yuv420(rgb: np.ndarray) -> bytes:
    """Convert RGB [H,W,3] uint8 to YUV420P byte string (FFmpeg format)."""
    import cv2

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420).tobytes()


def _yuv420_to_rgb(yuv: bytes, h: int, w: int) -> np.ndarray:
    """Convert YUV420P bytes to RGB [H,W,3] uint8."""
    import cv2

    yuv_np = np.frombuffer(yuv, dtype=np.uint8).reshape(h * 3 // 2, w)
    return cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_I420)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class LEWMEncoder:
    """Encoder wrapper called from FFmpeg C plugin via embedded Python.

    Produces real .lewm bitstream (NAL units with zlib-compressed latents).

    Usage (from C):
        encoder = get_encoder("lewmvc")
        packet = encoder.encode(width, height, yuv420_bytes)
    """

    def __init__(self, models: dict):
        self.enc = models["encoder"]
        self.pred = models["predictor"]
        self.entropy = models["entropy_model"]
        self.quant = models["quantizer"]
        self.dec = models["decoder"]
        self.context_len = getattr(self.pred, "context_len", 4)
        self.recon_latents: list[torch.Tensor] = []
        self.frame_count = 0
        self.total_bits = 0.0

    def reset(self):
        self.recon_latents = []
        self.frame_count = 0
        self.total_bits = 0.0

    def encode(self, width: int, height: int, yuv420_bytes: bytes) -> bytes:
        """Encode one frame, return NAL unit bytes.

        NAL format: [sync(0x4C) | type(1) | size(2 big-endian) | payload]

        Returns:
            NAL unit bytes with zlib-compressed quantized latent/residual.
        """
        rgb = _yuv420_to_rgb(yuv420_bytes, height, width)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(_DEVICE) / 255.0
        if tensor.shape[2] != _H or tensor.shape[3] != _W:
            tensor = torch.nn.functional.interpolate(tensor, size=(_H, _W), mode="bilinear")

        with torch.no_grad():
            latent, _ = self.enc(tensor, return_surprise=True)

            if len(self.recon_latents) == 0:
                quant_latent = self.quant(latent)
                recon = self.dec(quant_latent)
                rate, _ = self.entropy(quant_latent)
                is_iframe = True
                self.recon_latents.append(quant_latent)
                to_compress = quant_latent.cpu().numpy()
            else:
                ctx = self.recon_latents[max(0, len(self.recon_latents) - self.context_len) :]
                pred_mean, _ = self.pred(ctx)
                residual = latent - pred_mean
                quant_residual = self.quant(residual)
                rate, _ = self.entropy(quant_residual)
                recon_latent = pred_mean + quant_residual
                recon = self.dec(recon_latent)
                is_iframe = False
                self.recon_latents.append(recon_latent)
                to_compress = quant_residual.cpu().numpy()

        self.total_bits += rate.item()
        self.frame_count += 1

        from lewm_vc.bitstream import encode_frame as _ec_encode

        return _ec_encode(is_iframe, to_compress)

    def get_stats(self) -> dict:
        return {
            "frames_encoded": self.frame_count,
            "total_bits": self.total_bits,
            "total_bytes": self.total_bits / 8,
        }


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class LEWMDecoder:
    """Decoder wrapper called from FFmpeg C plugin."""

    def __init__(self, models: dict):
        self.pred = models["predictor"]
        self.dec = models["decoder"]
        self.quant = models["quantizer"]
        self.context_len = getattr(self.pred, "context_len", 4)
        self.recon_latents: list[torch.Tensor] = []

    def reset(self):
        self.recon_latents = []

    def decode(self, packet: bytes) -> bytes:
        """Decode one NAL unit packet, return YUV420P bytes."""
        from lewm_vc.bitstream import unpack_nal, decode_frame as _ec_decode, NALType

        nal_type, _, _ = unpack_nal(packet, 0)
        latent_shape = (192, _H // 16, _W // 16)
        quant_np = _ec_decode(packet, latent_shape)
        quant_tensor = torch.from_numpy(quant_np).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            if nal_type == NALType.I_FRAME:
                recon = self.dec(quant_tensor)
                self.recon_latents.append(quant_tensor)
            else:
                ctx = self.recon_latents[max(0, len(self.recon_latents) - self.context_len) :]
                pred_mean, _ = self.pred(ctx)
                recon_latent = pred_mean + quant_tensor
                recon = self.dec(recon_latent)
                self.recon_latents.append(recon_latent)

        recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()
        recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
        if recon_np.shape[0] != _H or recon_np.shape[1] != _W:
            import cv2

            recon_np = cv2.resize(recon_np, (_W, _H), interpolation=cv2.INTER_LINEAR)
        return _rgb_to_yuv420(recon_np)


# ---------------------------------------------------------------------------
# Module-level interface (called from C plugin)
# ---------------------------------------------------------------------------

_models = None


def get_encoder(name: str = "lewmvc"):
    """Called from FFmpeg C plugin via Python.h.

    Returns a Python object with .encode(width, height, yuv_bytes) method.
    """
    global _models
    if _models is None:
        _models = _build_models()
    return LEWMEncoder(_models)


def get_decoder(name: str = "lewmvc"):
    """Called from FFmpeg C plugin via Python.h.

    Returns a Python object with .decode(packet_bytes) method.
    """
    global _models
    if _models is None:
        _models = _build_models()
    return LEWMDecoder(_models)
