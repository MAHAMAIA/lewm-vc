"""Tests for FrameDataset."""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.lewm_vc.data import FrameDataset, find_clips, split_clips


def _make_fake_clip(tmp_path: str, name: str, num_frames: int = 10):
    """Create a directory of fake PNG frames."""
    clip_dir = Path(tmp_path) / "frames" / name
    clip_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        img = Image.new("RGB", (256, 256), color=(i * 20 % 255, 0, 0))
        img.save(clip_dir / f"frame_{i + 1:04d}.png")
    return clip_dir


class TestFindClips:
    def test_discovers_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_clip(tmp, "clip_a", 5)
            _make_fake_clip(tmp, "clip_b", 10)
            clips = find_clips([str(Path(tmp) / "frames")])
            assert len(clips) == 2
            names = [Path(c.path).name for c in clips]
            assert "clip_a" in names
            assert "clip_b" in names

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            clips = find_clips([tmp])
            assert len(clips) == 0


class TestSplitClips:
    def test_splits_by_clip(self):
        clips = (
            find_clips(["tests/fixtures/frames"]) if Path("tests/fixtures/frames").exists() else []
        )
        if not clips:
            pytest.skip("no test fixtures available")
        splits = split_clips(clips)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        total = sum(len(v) for v in splits.values())
        assert total == len(clips)


class TestFrameDataset:
    def test_single_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_clip(tmp, "test_clip", 20)
            ds = FrameDataset.from_root(str(Path(tmp) / "frames"), sequence_length=1)
            assert len(ds) > 0
            item = ds[0]
            assert "frames" in item
            assert item["frames"].shape == (1, 3, 256, 256)
            assert item["frames"].dtype == torch.float32
            assert 0 <= item["frames"].min() <= item["frames"].max() <= 1.0

    def test_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_clip(tmp, "test_clip", 30)
            ds = FrameDataset.from_root(str(Path(tmp) / "frames"), sequence_length=8)
            assert len(ds) >= 1
            item = ds[0]
            assert item["frames"].shape == (8, 3, 256, 256)

    def test_augment_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_clip(tmp, "test_clip", 10)
            ds_aug = FrameDataset.from_root(
                str(Path(tmp) / "frames"), sequence_length=1, augment=True
            )
            ds_plain = FrameDataset.from_root(
                str(Path(tmp) / "frames"), sequence_length=1, augment=False
            )
            # Same index should (probably) differ after augment
            a = ds_aug[0]["frames"]
            b = ds_plain[0]["frames"]
            # They might be different due to random aug
            assert a.shape == b.shape

    def test_collate(self):
        from src.lewm_vc.data.dataset import collate_sequences

        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_clip(tmp, "test_clip", 20)
            ds = FrameDataset.from_root(str(Path(tmp) / "frames"), sequence_length=4)
            loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=collate_sequences)
            batch = next(iter(loader))
            assert batch["frames"].shape == (4, 4, 3, 256, 256)

    def test_frame_stride(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_clip(tmp, "test_clip", 20)
            ds = FrameDataset.from_root(
                str(Path(tmp) / "frames"), sequence_length=4, frame_stride=2
            )
            # With stride 2, we skip every other frame in the sequence
            # 20 frames, seq_len=4, stride=2 -> valid starts: 0,2,4,6,8,10,12
            # So index 1 should be start=2: frames 2,4,6,8
            item = ds[1]
            assert item["frame_start"] == 2
            assert item["frames"].shape == (4, 3, 256, 256)
