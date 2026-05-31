"""
LeWM-VC Example Usage

Examples demonstrating how to use the LeWM-VC video codec.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from codec import LeWMVideoCodec, compute_psnr


def example_basic_usage():
    """Basic encoding/decoding example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    codec = LeWMVideoCodec()

    frames = []
    video_path = Path(__file__).parent.parent / "test_video.mp4"
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        for i in range(16):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
    else:
        frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(16)]
        print(f"Using random frames (no test video found)")

    print(f"\nEncoding {len(frames)} frames...")
    encoded, stats = codec.encode_video(frames)

    print(f"\nEncoding Results:")
    print(f"  Total frames: {stats.total_frames}")
    print(f"  I-frames: {stats.i_frames}")
    print(f"  P-frames: {stats.p_frames}")
    print(f"  Total bits: {stats.total_bits:,}")
    print(f"  Total bytes: {stats.total_bytes:,}")
    print(f"  Avg bits/frame: {stats.avg_bits_per_frame:.1f}")
    print(f"  Encoding time: {stats.encoding_time_s:.3f}s")
    print(f"  FPS: {stats.fps:.1f}")

    print(f"\nDecoding frames...")
    decoded = codec.decode_video(encoded)

    psnrs = [compute_psnr(orig, dec) for orig, dec in zip(frames, decoded)]
    avg_psnr = np.mean(psnrs)
    print(f"\nQuality Results:")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Min PSNR: {min(psnrs):.2f} dB")
    print(f"  Max PSNR: {max(psnrs):.2f} dB")


def example_single_frame():
    """Encode/decode a single frame."""
    print("\n" + "=" * 60)
    print("Example 2: Single Frame Encoding")
    print("=" * 60)

    codec = LeWMVideoCodec()

    frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    print(f"\nInput frame shape: {frame.shape}")

    encoded = codec.encode_frame(frame)
    print(f"\nEncoded:")
    print(f"  Frame type: {encoded.frame_type}")
    print(f"  Bits used: {encoded.bits_used:,}")
    print(f"  Encoding time: {encoded.encoding_time_ms:.2f}ms")
    print(f"  Latent shape: {encoded.latent.shape}")

    decoded = codec.decode_frame(encoded)
    print(f"\nDecoded frame shape: {decoded.shape}")

    psnr = compute_psnr(frame, decoded)
    print(f"  PSNR: {psnr:.2f} dB")


def example_batch_processing():
    """Batch processing with rate control."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)

    codec = LeWMVideoCodec()

    num_batches = 4
    frames_per_batch = 8

    for batch_idx in range(num_batches):
        frames = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(frames_per_batch)
        ]

        encoded, stats = codec.encode_video(frames)

        print(f"\nBatch {batch_idx + 1}/{num_batches}:")
        print(f"  Frames: {stats.total_frames}")
        print(f"  Total bits: {stats.total_bits:,}")
        print(f"  Avg bits/frame: {stats.avg_bits_per_frame:.1f}")
        print(f"  FPS: {stats.fps:.1f}")


def example_api_server():
    """Example using the API server."""
    print("\n" + "=" * 60)
    print("Example 4: API Server Usage")
    print("=" * 60)

    print("""
To use the API server:

1. Start the server:
   docker run -p 5000:5000 \\
     -v $(pwd)/checkpoint:/app/checkpoint \\
     lewmvc-stream

2. Use the Python client:

```python
from src.client import LeWMClient

client = LeWMClient("http://localhost:5000")

# Start session
client.start_session()

# Encode frames
frames = [...]  # List of numpy arrays
for frame in frames:
    result = client.encode_frame(frame)

# Get stats
stats = client.get_stats()
print(f"Total frames: {stats['frames_processed']}")
print(f"Avg bits/frame: {stats['avg_bits_per_frame']:.1f}")

# Decode
decoded = client.decode_frame(0)
```

Or with RTSP stream:

```python
encoded, stats = client.encode_rtsp(
    "rtsp://camera:554/stream",
    num_frames=100
)
```
""")


def main():
    print("LeWM-VC Example Usage")
    print("=" * 60)

    example_basic_usage()
    example_single_frame()
    example_batch_processing()
    example_api_server()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
