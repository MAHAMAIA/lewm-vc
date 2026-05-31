#!/usr/bin/env python3
"""
download_uvg.py — downloads UVG 1080p YUV files directly via HTTP, converts to MP4.

Usage:
    python scripts/download_uvg.py

Note: As of 2026, the ultravideo.fi URLs return 404. The UVG dataset may need
to be sourced from an alternative mirror.
"""

import os
import subprocess
import sys
from urllib.request import urlopen

UVG_FILES = {
    "Beauty_1920x1080_120fps_420_8bit_P420.yuv": "https://ultravideo.fi/files/Beauty_1920x1080_120fps_420_8bit_P420.yuv",
    "Bosphorus_1920x1080_120fps_420_8bit_P420.yuv": "https://ultravideo.fi/files/Bosphorus_1920x1080_120fps_420_8bit_P420.yuv",
    "HoneyBee_1920x1080_120fps_420_8bit_P420.yuv": "https://ultravideo.fi/files/HoneyBee_1920x1080_120fps_420_8bit_P420.yuv",
    "Jockey_1920x1080_120fps_420_8bit_P420.yuv": "https://ultravideo.fi/files/Jockey_1920x1080_120fps_420_8bit_P420.yuv",
    "ReadySetGo_1920x1080_120fps_420_8bit_P420.yuv": "https://ultravideo.fi/files/ReadySetGo_1920x1080_120fps_420_8bit_P420.yuv",
    "ShakeNDry_1920x1080_120fps_420_8bit_P420.yuv": "https://ultravideo.fi/files/ShakeNDry_1920x1080_120fps_420_8bit_P420.yuv",
    "YachtRide_1920x1080_120fps_420_8bit_P420.yuv": "https://ultravideo.fi/files/YachtRide_1920x1080_120fps_420_8bit_P420.yuv",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets", "uvg")


def download_file(url: str, dest_path: str) -> bool:
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000000:
        print(f"  Skipping {os.path.basename(dest_path)}")
        return True
    try:
        subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", dest_path, url], check=True)
        return True
    except FileNotFoundError:
        pass
    try:
        subprocess.run(
            ["curl", "-L", "-C", "-", "--progress-bar", "-o", dest_path, url], check=True
        )
        return True
    except FileNotFoundError:
        pass
    # Fallback: urllib
    try:
        print(f"  Downloading {os.path.basename(dest_path)} via urllib...")
        with urlopen(url, timeout=300) as resp, open(dest_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    sys.stdout.write(
                        f"\r    {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)"
                    )
                    sys.stdout.flush()
            print()
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def yuv_to_mp4(yuv_path: str, mp4_path: str) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        "1920x1080",
        "-r",
        "30",
        "-i",
        yuv_path,
        "-c:v",
        "libx264",
        "-crf",
        "0",
        "-preset",
        "ultrafast",
        mp4_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"Output: {DATASET_DIR}")
    for filename, url in UVG_FILES.items():
        yuv_path = os.path.join(DATASET_DIR, filename)
        mp4_path = yuv_path.replace(".yuv", ".mp4")
        if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 1000000:
            print(f"  [skip] {filename} (MP4 exists)")
            continue
        print(f"  {filename}...")
        if not download_file(url, yuv_path):
            continue
        if yuv_to_mp4(yuv_path, mp4_path):
            os.remove(yuv_path)
            print(f"    Done: {os.path.basename(mp4_path)}")
        else:
            print("    Saved YUV (ffmpeg conversion failed)")
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith((".yuv", ".mp4"))]
    print(f"\nDone: {len(files)} files in {DATASET_DIR}")


if __name__ == "__main__":
    main()
