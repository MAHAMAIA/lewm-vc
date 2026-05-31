#!/usr/bin/env python3
"""
Download SFU-HW-Objects-v1 dataset from Mendeley Data.

MPEG VCM CTC dataset for object detection benchmarking. Contains YUV sequences
organized by resolution class (A-D). Converts to 256x256 PNG frames via ffmpeg.

Usage:
    python scripts/download_sfu.py                            # default: datasets/sfu/
    python scripts/download_sfu.py --max-frames 300           # limit frames per sequence
    python scripts/download_sfu.py --out-dir /mnt/data/sfu
    python scripts/download_sfu.py --list-only                # show available files
    python scripts/download_sfu.py --resume                   # resume from progress

Output: datasets/sfu/frames/<sequence_name>/frame_XXXX.png
Total: ~5-10 GB, 23 sequences across classes A-D
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

MENDELEY_API = "https://data.mendeley.com/api/datasets"
DATASET_ID = "hwm673bv4m"
PROGRESS_FILE = "sfu_progress.json"
DEFAULT_MAX_FRAMES = 300

# Resolution map for YUV to PNG conversion
SEQUENCE_INFO = {
    # Class A (2560x1600)
    "Traffic": {"res": "2560x1600", "fps": 30, "class": "A"},
    "PeopleOnStreet": {"res": "2560x1600", "fps": 30, "class": "A"},
    # Class B (1920x1080)
    "Cactus": {"res": "1920x1080", "fps": 50, "class": "B"},
    "Kimono": {"res": "1920x1080", "fps": 24, "class": "B"},
    "ParkScene": {"res": "1920x1080", "fps": 24, "class": "B"},
    "BasketballDrive": {"res": "1920x1080", "fps": 50, "class": "B"},
    "BQTerrace": {"res": "1920x1080", "fps": 60, "class": "B"},
    # Class C (832x480)
    "BasketballDrill": {"res": "832x480", "fps": 50, "class": "C"},
    "BQMall": {"res": "832x480", "fps": 60, "class": "C"},
    "PartyScene": {"res": "832x480", "fps": 50, "class": "C"},
    "RaceHorsesC": {"res": "832x480", "fps": 30, "class": "C"},
    # Class D (416x240)
    "BasketballPass": {"res": "416x240", "fps": 50, "class": "D"},
    "BlowingBubbles": {"res": "416x240", "fps": 50, "class": "D"},
    "BQSquare": {"res": "416x240", "fps": 60, "class": "D"},
    "RaceHorsesD": {"res": "416x240", "fps": 30, "class": "D"},
}


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


def mendeley_get(endpoint: str) -> dict | list:
    url = f"{MENDELEY_API}/{DATASET_ID}/{endpoint}"
    req = Request(
        url, headers={"User-Agent": "sentinel-downloader/1.0", "Accept": "application/json"}
    )
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def list_files() -> list[dict]:
    """List all files in the Mendeley dataset."""
    print(f"Fetching SFU-HW-Objects-v1 file listing from Mendeley...")
    data = mendeley_get("files")
    files = data if isinstance(data, list) else data.get("data", data.get("files", [data]))
    print(f"  Found {len(files)} files")
    return files


def find_download_urls(files: list[dict]) -> list[dict]:
    """Get download URLs for YUV/raw video files."""
    downloads = []
    for f in files:
        name = f.get("file_name", f.get("name", ""))
        file_id = f.get("file_id", f.get("id", ""))
        label = f.get("label", os.path.splitext(name)[0])

        # Match known sequences
        matched_seq = None
        for seq_name in SEQUENCE_INFO:
            if seq_name.lower() in name.lower() or seq_name.lower() in label.lower():
                matched_seq = seq_name
                break

        if matched_seq:
            downloads.append(
                {
                    "name": matched_seq,
                    "filename": name,
                    "file_id": file_id,
                    "info": SEQUENCE_INFO[matched_seq],
                    "url": f"{MENDELEY_API}/{DATASET_ID}/files/{file_id}" if file_id else None,
                }
            )

    return downloads


def download_file(url: str, dest: str):
    """Download a single file."""
    req = Request(url, headers={"User-Agent": "sentinel-downloader/1.0"})
    with urlopen(req, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    sys.stdout.write(
                        f"\r    Downloading... {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)"
                    )
                    sys.stdout.flush()
    print()


def yuv_to_frames(
    yuv_path: str, frames_dir: str, seq_name: str, res: str, fps: int, max_frames: int
):
    """Convert YUV to PNG frames via ffmpeg."""
    tmp_dir = os.path.join(os.path.dirname(yuv_path), f"_tmp_{seq_name}")
    os.makedirs(tmp_dir, exist_ok=True)

    width, height = res.split("x")

    result = subprocess.run(
        [
            "ffmpeg",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            res,
            "-r",
            str(fps),
            "-i",
            yuv_path,
            "-vf",
            f"scale=256:256",
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
        print(f"    ffmpeg error:\n{result.stderr.strip()}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return 0

    clip_out = os.path.join(frames_dir, seq_name)
    os.makedirs(clip_out, exist_ok=True)
    for png in sorted(os.listdir(tmp_dir)):
        shutil.move(os.path.join(tmp_dir, png), os.path.join(clip_out, png))
    shutil.rmtree(tmp_dir)

    return len([f for f in os.listdir(clip_out) if f.endswith(".png")])


def main():
    parser = argparse.ArgumentParser(description="Download SFU-HW-Objects-v1")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <project_root>/datasets/sfu/)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=f"Max frames per sequence (default: {DEFAULT_MAX_FRAMES})",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List available files and exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved progress",
    )
    args = parser.parse_args()

    if args.out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        out_dir = os.path.join(project_root, "datasets", "sfu")
    else:
        out_dir = os.path.abspath(args.out_dir)

    videos_dir = os.path.join(out_dir, "videos")
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    processed = load_progress(out_dir) if args.resume else set()

    files = list_files()
    sequences = find_download_urls(files)

    if args.list_only:
        print(f"\n{'Sequence':25s} {'Class':6s} {'Resolution':12s} {'FPS':5s}")
        print("-" * 50)
        for seq in sorted(sequences, key=lambda s: s["info"]["class"]):
            info = seq["info"]
            print(f"{seq['name']:25s} {info['class']:6s} {info['res']:12s} {info['fps']:5d}")
        return

    print(f"\nOutput:  {out_dir}")
    print(f"Found {len(sequences)} sequences")
    print(f"Already done: {len(processed)}\n")

    for idx, seq in enumerate(sequences):
        name = seq["name"]
        clip_out = os.path.join(frames_dir, name)
        raw_path = os.path.join(videos_dir, seq["filename"].replace(".yuv", "_raw.yuv"))

        # Already processed
        if os.path.exists(clip_out):
            existing = [f for f in os.listdir(clip_out) if f.endswith(".png")]
            if existing:
                print(f"[{idx + 1}/{len(sequences)}] {name} — {len(existing)} frames exist")
                processed.add(name)
                save_progress(out_dir, processed)
                continue

        print(f"[{idx + 1}/{len(sequences)}] {name} (class {seq['info']['class']})...")

        # Try downloading if we have a URL
        if seq["url"]:
            if not (os.path.exists(raw_path) and os.path.getsize(raw_path) > 1e6):
                try:
                    download_file(seq["url"], raw_path)
                except HTTPError as e:
                    print(f"    Download failed: {e}")
                    print("    You may need to download manually from:")
                    print(f"    https://data.mendeley.com/datasets/{DATASET_ID}")
                    print(f"    Then place the file in: {raw_path}")
                    continue
            else:
                print("    Already downloaded")

        if not os.path.exists(raw_path):
            print(f"    File not found at {raw_path}, skipping")
            continue

        # Convert to frames
        n_frames = yuv_to_frames(
            raw_path, frames_dir, name, seq["info"]["res"], seq["info"]["fps"], args.max_frames
        )

        if n_frames > 0:
            os.remove(raw_path)
            print(f"    {n_frames} frames, raw deleted")
            processed.add(name)
            save_progress(out_dir, processed)
        else:
            print(f"    Conversion failed, keeping raw")

    # Summary
    total_frames = 0
    seq_count = 0
    for clip in sorted(os.listdir(frames_dir)):
        clip_path = os.path.join(frames_dir, clip)
        if os.path.isdir(clip_path):
            n = len([f for f in os.listdir(clip_path) if f.endswith(".png")])
            total_frames += n
            seq_count += 1
    print(f"\n{'=' * 40}")
    print(f"Sequences:  {seq_count}")
    print(f"Frames:     {total_frames}")
    print(f"Total GB:   {get_folder_size_gb(out_dir):.1f}")
    print(f"Location:   {out_dir}")


if __name__ == "__main__":
    main()
