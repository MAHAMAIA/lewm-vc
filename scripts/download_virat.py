#!/usr/bin/env python3
"""
Download and preprocess VIRAT Ground Dataset.

Downloads the monolithic collection zip from Kitware (~111 GB), then extracts and
converts all videos to 256x256 PNG frames via ffmpeg. Raw MP4s are deleted
after conversion to minimize disk usage. The zip is kept for future use.

Usage:
    python scripts/download_virat.py                           # all videos, all frames
    python scripts/download_virat.py --max-videos 5            # first 5 videos only
    python scripts/download_virat.py --max-frames 300          # limit per video
    python scripts/download_virat.py --resume                  # skip processed videos

Output: datasets/virat/frames/<video_name>/frame_XXXX.png
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

DOWNLOAD_URL = "https://data.kitware.com/api/v1/collection/56f56db28d777f753209ba9f/download"
PROGRESS_FILE = "virat_progress.json"
DEFAULT_MAX_VIDEOS = 0  # 0 = all videos
DEFAULT_MAX_FRAMES = 0  # 0 = all frames


def get_folder_size_gb(path: str) -> float:
    total = 0
    for dp, _, fn in os.walk(path):
        for f in fn:
            fp = os.path.join(dp, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / 1e9


def load_progress(out_dir: str) -> set[str]:
    path = os.path.join(out_dir, PROGRESS_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f))
    return set()


def save_progress(out_dir: str, processed: set[str]):
    path = os.path.join(out_dir, PROGRESS_FILE)
    with open(path, "w") as f:
        json.dump(sorted(processed), f)


def download_zip(dest: str):
    """Download the monolithic VIRAT collection zip via wget with resume."""
    if os.path.exists(dest):
        existing = os.path.getsize(dest)
        if existing > 1e6:
            # Check if zip is valid
            try:
                with zipfile.ZipFile(dest) as zf:
                    names = zf.namelist()
                print(
                    f"  Valid zip exists ({existing / 1e6:.0f} MB, {len(names)} files), skipping download"
                )
                return
            except zipfile.BadZipFile:
                print(f"  Partial download found ({existing / 1e6:.0f} MB), resuming...")
        else:
            os.remove(dest)
    print("  Downloading VIRAT archive (~111 GB)...")
    result = subprocess.run(
        ["wget", "-c", "-q", "--show-progress", "-O", dest, DOWNLOAD_URL],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  wget failed:\n{result.stderr}")
        sys.exit(1)
    size_mb = os.path.getsize(dest) / 1e6
    print(f"  Downloaded: {size_mb:.0f} MB")
    # Verify zip integrity
    try:
        with zipfile.ZipFile(dest) as zf:
            names = zf.namelist()
        print(f"  Valid zip with {len(names)} files")
    except zipfile.BadZipFile as e:
        print(f"  Downloaded file is corrupt: {e}")
        print("  The Kitware download may have been interrupted. Try re-running --resume.")
        sys.exit(1)


def convert_to_frames(video_path: str, frames_dir: str, video_name: str, max_frames: int) -> int:
    """Convert video to 256x256 PNG frames via ffmpeg. Returns frame count.
    If max_frames <= 0, extracts all frames."""
    tmp_dir = os.path.join(os.path.dirname(video_path), f"_tmp_{video_name}")
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        "scale=256:256,fps=30",
    ]
    if max_frames > 0:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [
        "-q:v",
        "1",
        f"{tmp_dir}/frame_%04d.png",
        "-y",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ffmpeg error:\n{result.stderr.strip()}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return 0

    clip_out = os.path.join(frames_dir, video_name)
    os.makedirs(clip_out, exist_ok=True)
    for png in sorted(os.listdir(tmp_dir)):
        shutil.move(os.path.join(tmp_dir, png), os.path.join(clip_out, png))
    shutil.rmtree(tmp_dir)

    return len([f for f in os.listdir(clip_out) if f.endswith(".png")])


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess VIRAT Ground Dataset")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <project_root>/datasets/virat/)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=DEFAULT_MAX_VIDEOS,
        help=f"Max videos to process (default: {DEFAULT_MAX_VIDEOS})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"Max frames per video (default: {DEFAULT_MAX_FRAMES})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-processed videos (tracked by progress file)",
    )
    parser.add_argument(
        "--disk-budget",
        type=float,
        default=0,
        help="Stop when output exceeds this many GB (0 = no limit)",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        out_dir = os.path.join(project_root, "datasets", "virat")
    else:
        out_dir = os.path.abspath(args.out_dir)

    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    zip_path = os.path.join(out_dir, "VIRAT.zip")
    tmp_extract = os.path.join(out_dir, "_tmp_videos")

    processed = load_progress(out_dir) if args.resume else set()

    # Step 1: Download zip
    print("=== Step 1: Download VIRAT archive ===")
    download_zip(zip_path)
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1e6:
        print("  Download failed or file too small, aborting")
        sys.exit(1)

    # Step 2: Extract and process videos one at a time
    frame_limit = args.max_frames if args.max_frames > 0 else "all"
    print(f"\n=== Step 2: Process videos ({frame_limit} frames each) ===")
    os.makedirs(tmp_extract, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_videos = sorted(n for n in zf.namelist() if n.endswith(".mp4"))
        if not all_videos:
            print("  No .mp4 files found in archive!")
            sys.exit(1)

        print(f"  Total videos in archive: {len(all_videos)}")
        already_done = len(processed)
        target = len(all_videos) if args.max_videos == 0 else min(args.max_videos, len(all_videos))
        print(f"  Already processed: {already_done}")
        print(f"  Will process:      {max(0, target - already_done)} more\n")

        for video_path in all_videos:
            name = os.path.splitext(os.path.basename(video_path))[0]
            if name in processed:
                continue
            if len(processed) >= target:
                break

            # Check disk budget
            if args.disk_budget > 0:
                usage = get_folder_size_gb(out_dir)
                if usage >= args.disk_budget:
                    print(f"  [stop] Disk budget reached ({usage:.1f} GB)")
                    break

            # Check if frames already exist
            clip_out = os.path.join(frames_dir, name)
            if os.path.exists(clip_out):
                existing = [f for f in os.listdir(clip_out) if f.endswith(".png")]
                if existing:
                    print(
                        f"  [{len(processed) + 1}/{target}] {name} — {len(existing)} frames exist"
                    )
                    processed.add(name)
                    save_progress(out_dir, processed)
                    continue

            # Extract single video from archive
            print(f"  [{len(processed) + 1}/{target}] {name}...", end=" ", flush=True)
            zf.extract(video_path, tmp_extract)
            raw_local = os.path.join(tmp_extract, video_path)

            # Convert to frames
            n_frames = convert_to_frames(raw_local, frames_dir, name, args.max_frames)

            # Delete raw MP4
            os.remove(raw_local)

            if n_frames > 0:
                print(f"{n_frames} frames")
                processed.add(name)
                save_progress(out_dir, processed)
            else:
                print("FAILED")

    # Cleanup (keep the zip for future runs)
    if os.path.exists(tmp_extract):
        shutil.rmtree(tmp_extract, ignore_errors=True)

    # Summary
    total_frames = 0
    video_count = 0
    for clip in sorted(os.listdir(frames_dir)):
        clip_path = os.path.join(frames_dir, clip)
        if os.path.isdir(clip_path):
            n = len([f for f in os.listdir(clip_path) if f.endswith(".png")])
            total_frames += n
            video_count += 1
    print(f"\n{'=' * 40}")
    print(f"Videos:    {video_count}")
    print(f"Frames:    {total_frames}")
    print(f"Total GB:  {get_folder_size_gb(out_dir):.1f}")
    print(f"Location:  {out_dir}")

    if video_count > 0:
        print(f"\nProgress file: {os.path.join(out_dir, PROGRESS_FILE)}")
        print("Resume with: python scripts/download_virat.py --resume")


if __name__ == "__main__":
    main()
