"""
Shared utilities for LeWM-VC reproduction experiments.

Provides dataset loading, metric computation, checkpoint management,
and the standard 256×256 frame pipeline used across all experiments.
"""

import os
import glob
import torch
import numpy as np
import cv2
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_M1 = ROOT / "checkpoints_milestone1"
CHECKPOINT_M2 = ROOT / "checkpoints_milestone2"
FRAME_SIZE = (256, 256)
FRAME_AREA = FRAME_SIZE[0] * FRAME_SIZE[1]


# ── Dataset Loading ─────────────────────────────────────────────
def get_pevid_paths(split="train") -> list[str]:
    """Return list of PEViD-HD .mpg file paths for the given split."""
    pevid_dir = DATASET_DIR / "pevid-hd"
    if not pevid_dir.exists():
        raise FileNotFoundError(
            f"PEViD-HD not found at {pevid_dir}. Run: bash experiment/01_download_data.sh"
        )
    paths = sorted(glob.glob(str(pevid_dir / "*.mpg")))
    if not paths:
        raise FileNotFoundError(f"No .mpg files in {pevid_dir}")
    if split == "train":
        return paths[:2]
    elif split == "test":
        return paths[2:3]
    return paths


def get_uvg_paths() -> list[str]:
    """Return list of UVG .mp4 file paths."""
    uvg_dir = DATASET_DIR / "uvg"
    if not uvg_dir.exists():
        raise FileNotFoundError(f"UVG dataset not found at {uvg_dir}")
    paths = sorted(glob.glob(str(uvg_dir / "*.mp4")))
    if not paths:
        raise FileNotFoundError(f"No .mp4 files in {uvg_dir}")
    return paths


def load_frames(
    video_path: str,
    max_frames: int = 100,
    target_size: tuple[int, int] = FRAME_SIZE,
) -> torch.Tensor:
    """Load video frames as normalized float RGB tensor [T, 3, H, W]."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)
    cap.release()
    return torch.from_numpy(np.stack(frames)).float()


# ── Metrics ─────────────────────────────────────────────────────
def compute_psnr(recon: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR in dB between two [B, C, H, W] tensors in [0, 1]."""
    mse = torch.mean((recon - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_bpp_from_entropy(
    latent: torch.Tensor,
    entropy_model: torch.nn.Module,
    quant_step: float = 2.0 / 255.0,
) -> float:
    """Estimate bits-per-pixel using the entropy model's cross-entropy bound."""
    with torch.no_grad():
        quantized = torch.round(latent / quant_step) * quant_step
        rates = entropy_model(quantized)
        nats = rates.sum().item()
        bits = nats / np.log(2)
    num_pixels = latent.shape[0] * FRAME_AREA
    return bits / (num_pixels * 3)


# ── Checkpoint Loading ──────────────────────────────────────────
def load_checkpoint(path: str | Path, model: torch.nn.Module, device: str = "cuda") -> None:
    """Load state dict into model, handling missing keys gracefully."""
    state = torch.load(str(path), map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)}")


# ── Frame Encoding ──────────────────────────────────────────────
@torch.no_grad()
def encode_frames(
    frames: torch.Tensor,
    encoder: torch.nn.Module,
    device: str = "cuda",
    batch_size: int = 8,
) -> torch.Tensor:
    """Encode a batch of frames [T, C, H, W] into latents [T, D, H', W']."""
    encoder.to(device).eval()
    T = frames.shape[0]
    latents = []
    for i in range(0, T, batch_size):
        batch = frames[i : i + batch_size].to(device)
        latents.append(encoder(batch))
    return torch.cat(latents, dim=0)


@torch.no_grad()
def decode_frames(
    latents: torch.Tensor,
    decoder: torch.nn.Module,
    device: str = "cuda",
    target_size: tuple[int, int] = FRAME_SIZE,
) -> torch.Tensor:
    """Decode latents [T, D, H', W'] back to frames [T, 3, H, W]."""
    decoder.to(device).eval()
    decoded = decoder(latents.to(device), target_size=target_size)
    return decoded.cpu()
