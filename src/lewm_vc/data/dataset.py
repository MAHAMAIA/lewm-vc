"""
LeWM-VC Dataset Module

Loads preprocessed PNG frames for training. Supports both single-frame
(intra-frame autoencoder) and sequence sampling (JEPA temporal predictor).

Expected directory structure after running download scripts:
    datasets/
    ├── pevid/frames/<clip_name>/frame_0001.png
    ├── virat/frames/<video_name>/frame_0001.png
    └── sfu/frames/<sequence_name>/frame_0001.png

Usage:
    # Auto-discover clips from a directory
    ds = FrameDataset.from_root("datasets/virat/frames", sequence_length=16)

    # Manual spec with train/val/test split
    spec = find_clips(["datasets/virat/frames", "datasets/pevid/frames"])
    train_ds = FrameDataset(spec["train"], sequence_length=16, augment=True)
    val_ds = FrameDataset(spec["val"], sequence_length=16)
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ClipSpec:
    """Describes a single clip directory of preprocessed frames."""

    path: str
    num_frames: int
    dataset: str = ""
    split: Literal["train", "val", "test"] = "train"


def find_clips(
    root_dirs: list[str],
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> list[ClipSpec]:
    """
    Scan directories for clip folders containing frame images.

    Each subdirectory containing image files is treated as one clip.
    Returns a list of ClipSpec sorted by clip name.
    """
    clips = []
    seen = set()
    for root in root_dirs:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            print(f"  [warning] directory not found: {root}")
            continue
        for entry in sorted(os.listdir(root)):
            clip_path = os.path.join(root, entry)
            if not os.path.isdir(clip_path):
                continue
            if clip_path in seen:
                continue
            seen.add(clip_path)
            frame_files = sorted(
                [f for f in os.listdir(clip_path) if f.lower().endswith(extensions)]
            )
            if not frame_files:
                continue
            clips.append(
                ClipSpec(
                    path=clip_path,
                    num_frames=len(frame_files),
                    dataset=os.path.basename(os.path.dirname(root)),
                )
            )
    clips.sort(key=lambda c: c.path)
    return clips


def split_clips(
    clips: list[ClipSpec],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[ClipSpec]]:
    """
    Split clips into train/val/test by clip (not by frame).
    Ensures no frame leakage between splits.
    """
    rng = random.Random(seed)
    shuffled = list(clips)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    result = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    for split_name, split_clips in result.items():
        for c in split_clips:
            c.split = split_name  # type: ignore
    return result


def write_spec(clips: list[ClipSpec], path: str):
    """Write a list of ClipSpec to a JSON lines file (one clip per line)."""
    with open(path, "w") as f:
        for c in clips:
            f.write(
                json.dumps(
                    {
                        "path": c.path,
                        "num_frames": c.num_frames,
                        "dataset": c.dataset,
                        "split": c.split,
                    }
                )
                + "\n"
            )


def load_spec(path: str) -> list[ClipSpec]:
    """Load ClipSpecs from a JSON lines file."""
    clips = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            clips.append(ClipSpec(**d))
    return clips


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


def _build_transforms(augment: bool, image_size: int = 256):
    """Build composition of torchvision transforms."""
    import torchvision.transforms.v2 as T  # noqa: N812

    transforms = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]

    if augment:
        transforms.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
                T.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            ]
        )

    transforms.append(T.Resize((image_size, image_size), antialias=True))
    return T.Compose(transforms)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FrameDataset(Dataset):
    """
    Dataset for preprocessed video frames as PNGs.

    Two modes:
      - sequence_length=1: returns single random frames (intra-frame / autoencoder)
      - sequence_length>1: returns contiguous sequences (JEPA temporal training)

    Each item is a dict: {"frames": [T, 3, H, W], "clip": str, "frame_start": int}

    Args:
        clips: List of ClipSpec describing available clips.
        sequence_length: Number of consecutive frames per sample.
        image_size: Spatial size to resize frames to.
        augment: Apply training augmentations.
        frame_rate: Frames per second stride (1 = every frame, 2 = every other frame, etc.).
    """

    def __init__(
        self,
        clips: list[ClipSpec],
        sequence_length: int = 1,
        image_size: int = 256,
        augment: bool = False,
        frame_stride: int = 1,
    ):
        super().__init__()
        self.clips = clips
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.transforms = _build_transforms(augment, image_size)

        # Build a flat index: for each clip, list of valid starting frame indices
        self._index: list[tuple[int, int]] = []  # (clip_idx, start_frame)
        for clip_idx, clip in enumerate(clips):
            stride = max(1, frame_stride)
            max_start = max(0, clip.num_frames - (sequence_length - 1) * stride - 1)
            for start in range(0, max_start + 1, stride):
                self._index.append((clip_idx, start))

        if not self._index:
            print(
                "  [warning] FrameDataset: no valid samples (clips may be too short for sequence_length)"
            )

    @classmethod
    def from_root(
        cls,
        root_dir: str,
        sequence_length: int = 1,
        image_size: int = 256,
        augment: bool = False,
        frame_stride: int = 1,
        split: Literal["train", "val", "test"] | None = None,
        find_kwargs: dict | None = None,
    ) -> "FrameDataset":
        """
        Create a dataset by auto-discovering clips under a root directory.

        Args:
            root_dir: Path to directory containing clip subdirectories.
            sequence_length: Number of consecutive frames per sample.
            image_size: Spatial size to resize frames to.
            augment: Apply training augmentations.
            frame_stride: Frame sampling stride.
            split: If provided, splits clips and returns only the given split.
            find_kwargs: Additional kwargs for find_clips().
        """
        clips = find_clips([root_dir], **(find_kwargs or {}))
        if split:
            splits = split_clips(clips)
            clips = splits[split]
            print(f"  {split}: {len(clips)} clips")
        total_frames = sum(c.num_frames for c in clips)
        print(f"  total frames: {total_frames}")
        return cls(
            clips,
            sequence_length=sequence_length,
            image_size=image_size,
            augment=augment,
            frame_stride=frame_stride,
        )

    @classmethod
    def from_roots(
        cls,
        root_dirs: list[str],
        sequence_length: int = 1,
        image_size: int = 256,
        augment: bool = False,
        frame_stride: int = 1,
        split: Literal["train", "val", "test"] | None = None,
    ) -> "FrameDataset":
        """Create dataset from multiple root directories."""
        clips = find_clips(root_dirs)
        if split:
            splits = split_clips(clips)
            clips = splits[split]
            print(f"  {split}: {len(clips)} clips")
        total_frames = sum(c.num_frames for c in clips)
        print(f"  total frames: {total_frames}")
        return cls(
            clips,
            sequence_length=sequence_length,
            image_size=image_size,
            augment=augment,
            frame_stride=frame_stride,
        )

    def __len__(self) -> int:
        return len(self._index) if self._index else 0

    def _load_frame(self, clip_path: str, frame_idx: int) -> torch.Tensor:
        """Load a single frame by index."""
        frame_path = os.path.join(clip_path, f"frame_{frame_idx + 1:04d}.png")
        img = Image.open(frame_path).convert("RGB")
        return self.transforms(img)

    def __getitem__(self, idx: int) -> dict:
        clip_idx, start = self._index[idx]
        clip = self.clips[clip_idx]

        frames = []
        for i in range(self.sequence_length):
            frame = self._load_frame(clip.path, start + i)
            frames.append(frame)

        return {
            "frames": torch.stack(frames, dim=0),  # [T, 3, H, W]
            "clip": os.path.basename(clip.path),
            "frame_start": start,
            "dataset": clip.dataset,
        }


def collate_sequences(batch: list[dict]) -> dict:
    """
    Collate function for DataLoader.
    Stacks frames into [B, T, 3, H, W] and collects metadata.
    """
    frames = torch.stack([item["frames"] for item in batch], dim=0)
    return {
        "frames": frames,
        "clip": [item["clip"] for item in batch],
        "frame_start": [item["frame_start"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
    }
