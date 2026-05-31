#!/usr/bin/env python3
"""
Download and preprocess PEViD-HD dataset.

Downloads .mpg clips from EPFL FTP, extracts frames as 256x256 PNGs organized
by clip name. Deletes raw .mpg after conversion to save space.

Output: datasets/pevid-hd/frames/<clip_name>/frame_XXXX.png
        datasets/pevid-hd/raw_mpg/  (deleted after conversion)

Usage:
    python scripts/download_pevid.py                      # default: datasets/pevid-hd/
    python scripts/download_pevid.py --out-dir /mnt/data/pevid
    python scripts/download_pevid.py --max-frames 200      # limit frames per clip
    python scripts/download_pevid.py --skip-download       # only convert already-downloaded files
    python scripts/download_pevid.py --download-only       # only download, no conversion

Requirements: ffmpeg on PATH
"""

import argparse
import os
import shutil
import subprocess
import sys
from ftplib import FTP

FTP_HOST = "tremplin.epfl.ch"
FTP_USER = "datasets@mmspgdata.epfl.ch"
FTP_PASS = "ohsh9jah4T"
FTP_DIR = "/PEViD/PEViD-HD"

DEFAULT_MAX_FRAMES = 400  # 16 seconds x 25 fps


def get_folder_size_gb(path: str) -> float:
    total = 0
    for dp, _, fn in os.walk(path):
        for f in fn:
            fp = os.path.join(dp, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / 1e9


def download_all(raw_dir: str) -> list[str]:
    """Download .mpg files from EPFL FTP. Returns list of downloaded filenames."""
    os.makedirs(raw_dir, exist_ok=True)
    existing = {f for f in os.listdir(raw_dir) if f.endswith(".mpg")}

    print(f"Connecting to {FTP_HOST}...")
    ftp = FTP(FTP_HOST)
    ftp.login(user=FTP_USER, passwd=FTP_PASS)
    ftp.cwd(FTP_DIR)

    files = []
    ftp.retrlines("LIST", lambda line: files.append(line.split()[-1]))
    mpg_files = sorted(f for f in files if f.endswith(".mpg"))

    if not mpg_files:
        print("No .mpg files found on FTP server.")
        ftp.quit()
        sys.exit(1)

    print(f"Found {len(mpg_files)} clips on FTP server.")
    downloaded = []
    for fname in mpg_files:
        local_path = os.path.join(raw_dir, fname)
        if fname in existing and os.path.getsize(local_path) > 0:
            print(f"  [skip] {fname}")
            downloaded.append(fname)
            continue
        print(f"  [download] {fname}...", flush=True)
        with open(local_path, "wb") as fp:
            ftp.retrbinary(f"RETR {fname}", fp.write)
        downloaded.append(fname)

    ftp.quit()
    print(f"Downloaded {len(downloaded)} files to {raw_dir}")
    return downloaded


def process_clips(raw_dir: str, frames_dir: str, max_frames: int):
    """Convert each .mpg to PNG frames 256x256, delete raw after processing."""
    mpg_files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".mpg"))
    total = len(mpg_files)
    processed = 0

    if not mpg_files:
        print("No .mpg files found in", raw_dir)
        return

    print(f"\nConverting {total} clips to frames...")
    for idx, fname in enumerate(mpg_files, 1):
        clip_name = os.path.splitext(fname)[0]
        clip_out = os.path.join(frames_dir, clip_name)
        raw_path = os.path.join(raw_dir, fname)

        # Already processed
        if os.path.exists(clip_out):
            existing_frames = [f for f in os.listdir(clip_out) if f.endswith(".png")]
            if existing_frames:
                print(
                    f"  [{idx}/{total}] {clip_name} — {len(existing_frames)} frames exist, skipping"
                )
                os.remove(raw_path)
                processed += 1
                continue

        # Convert to frames via tmp dir
        tmp_dir = os.path.join(raw_dir, f"_tmp_{clip_name}")
        os.makedirs(tmp_dir, exist_ok=True)

        print(f"  [{idx}/{total}] {clip_name}...", end=" ", flush=True)
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                raw_path,
                "-vf",
                "scale=256:256,fps=25",
                "-frames:v",
                str(max_frames),
                "-q:v",
                "1",
                f"{tmp_dir}/frame_%04d.png",
                "-y",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"ffmpeg error:\n{result.stderr.strip()}")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        # Move to final location
        os.makedirs(clip_out, exist_ok=True)
        for png in sorted(os.listdir(tmp_dir)):
            shutil.move(os.path.join(tmp_dir, png), os.path.join(clip_out, png))
        shutil.rmtree(tmp_dir)

        # Delete raw MPG to save space
        os.remove(raw_path)

        n_frames = len([f for f in os.listdir(clip_out) if f.endswith(".png")])
        print(f"{n_frames} frames")
        processed += 1

    # Cleanup raw dir if empty
    remaining = [f for f in os.listdir(raw_dir) if f.endswith(".mpg")]
    if not remaining:
        shutil.rmtree(raw_dir, ignore_errors=True)
        print(f"Removed temporary directory: {raw_dir}")

    print(f"\nProcessed {processed}/{total} clips.")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess PEViD-HD")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <project_root>/datasets/pevid-hd/)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"Max frames per clip (default: {DEFAULT_MAX_FRAMES})",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip FTP download, only process existing raw files",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download raw .mpg files, skip conversion",
    )
    args = parser.parse_args()

    # Default to project-root-relative path
    if args.out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        out_dir = os.path.join(project_root, "datasets", "pevid-hd")
    else:
        out_dir = os.path.abspath(args.out_dir)

    raw_dir = os.path.join(out_dir, "raw_mpg")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Output:    {out_dir}")
    print(f"Max frames per clip: {args.max_frames}")

    if os.path.exists(out_dir):
        print(f"Disk used: {get_folder_size_gb(out_dir):.1f} GB")

    # Download
    if not args.skip_download:
        download_all(raw_dir)
    else:
        if not os.path.exists(raw_dir) or not any(f.endswith(".mpg") for f in os.listdir(raw_dir)):
            print("No raw .mpg files found. Remove --skip-download or place files in:", raw_dir)
            sys.exit(1)
        print("Skipping FTP download (--skip-download)")

    # Convert
    if not args.download_only:
        process_clips(raw_dir, frames_dir, args.max_frames)
    else:
        print("Skipping frame extraction (--download-only)")

    # Summary
    total_frames = 0
    clip_count = 0
    for clip in sorted(os.listdir(frames_dir)):
        clip_path = os.path.join(frames_dir, clip)
        if os.path.isdir(clip_path):
            n = len([f for f in os.listdir(clip_path) if f.endswith(".png")])
            total_frames += n
            clip_count += 1
    print(f"\n{'=' * 40}")
    print(f"Clips:     {clip_count}")
    print(f"Frames:    {total_frames}")
    print(f"Total GB:  {get_folder_size_gb(out_dir):.1f}")
    print(f"Location:  {out_dir}")

    # Citation
    print("\nCitation: P. Korshunov and T. Ebrahimi, 'PEViD: privacy evaluation video dataset',")
    print("SPIE Applications of Digital Image Processing XXXVI, volume 8856, San Diego, Aug. 2013.")


if __name__ == "__main__":
    main()
