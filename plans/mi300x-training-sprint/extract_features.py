"""
Step 1: Feature Extraction Pipeline
Extracts P3/P4/P5 features from ResNet50-FPN for SeaDronesSee + SMD.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from lewm_vc.data.dataset import FrameDataset, find_clips, split_clips, write_spec
from fpn_backbone import ResNet50FPN


def extract_and_save(
    backbone: ResNet50FPN,
    loader: DataLoader,
    output_dir: str,
    pyramid_levels: list[str],
    device: str,
    split: str,
):
    """Extract FPN features from frames and save to disk."""
    output_dir = Path(output_dir)
    for level in pyramid_levels:
        (output_dir / level / split).mkdir(parents=True, exist_ok=True)

    metadata = []
    idx = 0
    backbone = backbone.to(device)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {split}"):
            frames = batch["frames"].to(device)  # [B, T, 3, H, W]
            b, t = frames.shape[:2]
            frames_flat = frames.view(b * t, *frames.shape[2:])

            feats = backbone(frames_flat)

            for level in pyramid_levels:
                f = feats[level]  # [B*T, 256, H_l, W_l]
                for i in range(b * t):
                    clip_name = batch["clip"][i // t]
                    frame_num = batch["frame_start"][i // t] + (i % t)
                    save_path = output_dir / level / split / f"feat_{idx:08d}.pt"
                    torch.save(
                        {
                            "features": f[i].cpu().half(),
                            "clip": clip_name,
                            "frame": frame_num,
                            "scale_level": level,
                            "split": split,
                        },
                        save_path,
                    )
                    idx += 1

            for i in range(b):
                metadata.append(
                    {
                        "clip": batch["clip"][i],
                        "frame_start": batch["frame_start"][i],
                        "dataset": batch["dataset"][i],
                        "split": split,
                    }
                )

    meta_path = output_dir / f"metadata_{split}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {len(metadata)} samples to {meta_path}")
    print(f"Total feature tensors saved: {idx}")


def main():
    parser = argparse.ArgumentParser(description="Extract FPN features for training")
    parser.add_argument(
        "--seadronessee",
        type=str,
        default="datasets/seadronessee/frames",
        help="SeaDronesSee v2 frames directory",
    )
    parser.add_argument(
        "--smd",
        type=str,
        default="datasets/smd/frames",
        help="Singapore Maritime Dataset frames directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/fpn_features",
        help="Output directory for extracted features",
    )
    parser.add_argument(
        "--levels",
        type=str,
        nargs="+",
        default=["P3", "P4", "P5"],
        help="Pyramid levels to extract",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Loading ResNet50-FPN backbone on {device}...")
    backbone = ResNet50FPN().to(device).eval()

    roots = []
    if os.path.isdir(args.seadronessee):
        roots.append(args.seadronessee)
        print(f"Found SeaDronesSee: {args.seadronessee}")
    if os.path.isdir(args.smd):
        roots.append(args.smd)
        print(f"Found SMD: {args.smd}")

    if not roots:
        print("ERROR: No dataset directories found.")
        print(f"Checked: {args.seadronessee}, {args.smd}")
        sys.exit(1)

    print("Discovering clips and creating video-aware splits...")
    all_clips = find_clips(roots)
    print(f"Found {len(all_clips)} clips total")
    splits = split_clips(all_clips, train_ratio=0.7, val_ratio=0.15)

    clip_spec_path = Path(args.output) / "clip_spec.jsonl"
    clip_spec_path.parent.mkdir(parents=True, exist_ok=True)
    write_spec(all_clips, str(clip_spec_path))
    print(f"Saved clip spec to {clip_spec_path}")

    for split_name in ["train", "val", "test"]:
        clips = splits[split_name]
        if not clips:
            print(f"  {split_name}: 0 clips, skipping")
            continue
        print(f"\n{'=' * 60}")
        print(f"Processing {split_name} split ({len(clips)} clips)")
        print(f"{'=' * 60}")

        ds = FrameDataset(
            clips,
            sequence_length=1,
            image_size=args.image_size,
            augment=(split_name == "train"),
            frame_stride=1,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=(split_name == "train"),
        )

        extract_and_save(
            backbone=backbone,
            loader=loader,
            output_dir=args.output,
            pyramid_levels=args.levels,
            device=device,
            split=split_name,
        )

    print(f"\nDone! Features saved to {args.output}/")
    print(f"Structure: {args.output}/{{P3,P4,P5}}/{{train,val,test}}/feat_*.pt")


if __name__ == "__main__":
    main()
