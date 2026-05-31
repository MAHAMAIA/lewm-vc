"""
LeWM-VC Python Client

Simple Python client library for LeWM-VC server.
"""

import base64
import json
from pathlib import Path
from typing import Optional

import numpy as np
import requests


class LeWMClient:
    """
    Python client for LeWM-VC video codec API.

    Args:
        base_url: Base URL of the LeWM-VC server (default: http://localhost:5000)
        timeout: Request timeout in seconds (default: 30)
    """

    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[int] = None

    def health(self) -> dict:
        """Check server health."""
        resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def start_session(self) -> dict:
        """Start a new encoding session."""
        resp = requests.post(f"{self.base_url}/start", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data.get("session_id")
        return data

    def encode_frame(self, image: np.ndarray) -> dict:
        """
        Encode a single frame.

        Args:
            image: [H, W, 3] numpy array (RGB, 0-255)

        Returns:
            Encoding result with statistics
        """
        _, buffer = np.tobytes("", dtype=np.uint8)
        img_str = base64.b64encode(image).decode("utf-8")

        resp = requests.post(
            f"{self.base_url}/encode",
            json={"image": img_str},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def encode_file(self, path: str) -> dict:
        """
        Encode an image file.

        Args:
            path: Path to image file

        Returns:
            Encoding result
        """
        import cv2

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.encode_frame(img)

    def decode_frame(self, frame_num: int, target_size: Optional[tuple] = None) -> np.ndarray:
        """
        Decode an encoded frame.

        Args:
            frame_num: Frame number to decode
            target_size: Optional (H, W) target output size

        Returns:
            Decoded RGB frame as [H, W, 3] numpy array
        """
        import cv2

        payload = {"frame_num": frame_num}
        if target_size:
            payload["target_size"] = list(target_size)

        resp = requests.post(
            f"{self.base_url}/decode",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        img_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_stats(self) -> dict:
        """Get encoding statistics."""
        resp = requests.get(f"{self.base_url}/stats", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def reset(self) -> dict:
        """Reset the codec state."""
        resp = requests.post(f"{self.base_url}/reset", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def encode_video(self, frames: list[np.ndarray]) -> tuple[list[dict], dict]:
        """
        Encode a list of frames.

        Args:
            frames: List of [H, W, 3] numpy arrays (RGB, 0-255)

        Returns:
            Tuple of (encoded frames, stats)
        """
        self.start_session()

        encoded = []
        for frame in frames:
            result = self.encode_frame(frame)
            encoded.append(result)

        stats = self.get_stats()
        return encoded, stats

    def encode_rtsp(self, rtsp_url: str, num_frames: int = 100) -> tuple[list[dict], dict]:
        """
        Encode frames from an RTSP stream.

        Args:
            rtsp_url: RTSP stream URL
            num_frames: Number of frames to encode

        Returns:
            Tuple of (encoded frames, stats)
        """
        import cv2

        self.start_session()

        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {rtsp_url}")

        encoded = []
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.encode_frame(frame_rgb)
            encoded.append(result)

        cap.release()
        stats = self.get_stats()

        return encoded, stats


def main():
    """Demo usage."""
    client = LeWMClient()

    print("Checking server health...")
    health = client.health()
    print(f"  Status: {health['status']}")
    print(f"  Device: {health['device']}")

    print("\nStarting session...")
    session = client.start_session()
    print(f"  Session ID: {session['session_id']}")
    print(f"  Latent dim: {session['latent_dim']}")
    print(f"  GOP size: {session['gop_size']}")

    print("\nGetting stats...")
    stats = client.get_stats()
    print(f"  Frames: {stats['frames_processed']}")


if __name__ == "__main__":
    main()
