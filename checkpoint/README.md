# Checkpoints

This directory contains the trained model checkpoints.

## Available Checkpoints

| Checkpoint | Size | Description |
|------------|------|-------------|
| `ae_lambda_0.05_final.pt` | 28 MB | Autoencoder (Milestone 1) - Intra-frame compression |
| `entropy_lambda_0.05_final.pt` | 20 MB | Entropy model (Milestone 1) - Rate-distortion optimized |
| `temporal_final.pt` | 80 MB | Temporal codec (Milestone 2) - JEPA-based inter-frame compression |

## Download from GitHub Releases

All checkpoints are available for download from the [GitHub Releases](https://github.com/MAHAMAIA/lewm-vc/releases/tag/v0.1.0) page:

- [ae_lambda_0.05_final.pt](https://github.com/MAHAMAIA/lewm-vc/releases/download/v0.1.0/ae_lambda_0.05_final.pt)
- [entropy_lambda_0.05_final.pt](https://github.com/MAHAMAIA/lewm-vc/releases/download/v0.1.0/entropy_lambda_0.05_final.pt)
- [temporal_final.pt](https://github.com/MAHAMAIA/lewm-vc/releases/download/v0.1.0/temporal_final.pt)

```bash
# Download all checkpoints
wget https://github.com/MAHAMAIA/lewm-vc/releases/download/v0.1.0/ae_lambda_0.05_final.pt
wget https://github.com/MAHAMAIA/lewm-vc/releases/download/v0.1.0/entropy_lambda_0.05_final.pt
wget https://github.com/MAHAMAIA/lewm-vc/releases/download/v0.1.0/temporal_final.pt
```

## Usage

The `temporal_final.pt` checkpoint is used by default by `src/codec.py`:
```python
from src.codec import LeWMVideoCodec
codec = LeWMVideoCodec()  # Auto-loads checkpoint/temporal_final.pt
```