# LeWM-VC: JEPA-Based Video Codec

**Learned Energy-based Model for Video Coding**

A deep learning-based video codec using Joint Embedding Predictive Architecture (JEPA) with Vision Transformer (ViT) architecture. LeWM-VC compresses video frames into a compact latent space, predicts future latents via a transformer predictor, and codes only the residual — achieving temporal compression without explicit motion vectors.

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **P-frame bitrate savings** | **62%** | vs all-intra on PEViD-HD (256x256) |
| **Encoding FPS** | **80+ fps** | On NVIDIA T4 GPU |
| **PSNR (all-intra)** | 25.13 dB | 0.23 BPP on PEViD-HD |
| **PSNR (temporal)** | 25.40 dB | 0.09 BPP on PEViD-HD |
| **Latent dimension** | 192 | 16x16 spatial grid |

## Quick Start

### One-Command Docker Run

```bash
docker run -p 5000:5000 \
  -v $(pwd)/checkpoint:/app/checkpoint \
  ghcr.io/mahamaia/lewmvc-stream:latest
```

Or using docker-compose:

```bash
git clone https://github.com/MAHAMAIA/lewmvc-stream.git
cd lewmvc-stream
docker-compose up
```

### Python API

```python
from src.codec import LeWMVideoCodec

codec = LeWMVideoCodec()

frames = [...]  # List of [H, W, 3] numpy arrays (RGB, 0-255)

encoded, stats = codec.encode_video(frames)

decoded_frames = codec.decode_video(encoded)
```

### RTSP Stream

```python
from src.client import LeWMClient

client = LeWMClient("http://localhost:5000")

encoded, stats = client.encode_rtsp(
    "rtsp://camera:554/stream",
    num_frames=100
)

print(f"Encoded {stats['frames_processed']} frames")
print(f"Avg bits/frame: {stats['avg_bits_per_frame']:.1f}")
```

## Features

- **JEPA Temporal Prediction**: Transformer-based predictor learns temporal dependencies without explicit motion vectors
- **Vector Quantization**: 256-codebook VQ for latent compression
- **Adaptive Bit Allocation**: I/P-frame coding with configurable GOP size
- **REST API**: HTTP endpoints for encoding, decoding, and streaming
- **RTSP Support**: Direct integration with camera streams
- **Docker Ready**: One-command deployment with GPU support

## Architecture

```
Input Frame (256x256 RGB)
       |
       v
  ┌─────────────┐
  │ ViT Encoder │  (6-layer transformer, 192-dim latent)
  └──────┬──────┘
         |
         v
   Latent (192 x 16 x 16)
         |
         v
  ┌──────────────┐
  │ JEPA Predictor│  (8-layer transformer, context=4)
  └──────┬───────┘
         |
         v
   Residual Coding
         |
         v
    Bitstream
```

**Key Components:**
- **Encoder**: ViT-Tiny (6 layers, 192-dim, 3 attention heads)
- **Predictor**: 8-layer transformer (256-dim, 4 heads, context length 4)
- **Quantizer**: 256-codebook vector quantization
- **Decoder**: 4-layer ConvTranspose with residual blocks

## Limitations

- **Domain-specific**: Trained primarily on surveillance video (PEViD-HD dataset)
- **Fixed resolution**: Currently optimized for 256x256 (must be multiple of 16)
- **No B-frame support**: Only IPPP pattern (GOP size configurable)
- **Research stage**: Not yet benchmarked against VVC/H.266
- **CPU inference**: GPU recommended for real-time performance

## Installation

### From Source

```bash
git clone https://github.com/MAHAMAIA/lewmvc-stream.git
cd lewmvc-stream
pip install -r requirements.txt

python -m src.server
```

### With GPU Support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
docker run --gpus all -p 5000:5000 ghcr.io/mahamaia/lewmvc-stream:latest
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/start` | POST | Start new encoding session |
| `/encode` | POST | Encode single frame |
| `/decode` | POST | Decode encoded frame |
| `/stats` | GET | Get encoding statistics |
| `/reset` | POST | Reset codec state |

## Benchmark Results

Tested on NVIDIA T4 GPU with PEViD-HD surveillance video:

```
Encoding Performance (256x256, batch=1):
  I-frame:  ~12ms
  P-frame:  ~8ms
  Overall:  80+ fps

Bitrate Comparison:
  All-intra:  0.23 BPP  (25.13 dB)
  Temporal:   0.09 BPP  (25.40 dB)
  Savings:    62%
```

## Citation

If you use LeWM-VC in your research, please cite:

```bibtex
@misc{lewmvc2026,
  author = {Preetam Mukherjee},
  title = {LeWM-VC: JEPA-Based Video Codec with Temporal Latent Prediction},
  year = {2026},
  url = {https://github.com/MAHAMAIA/lewmvc-stream}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related

- [Private Research Repo](https://github.com/MAHAMAIA/le-maia) - Full training code and experiments
- [arXiv Paper](#) - Coming soon