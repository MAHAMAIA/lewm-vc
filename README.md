# Sentinel: Feature Compression Codec for Machine Perception

**MPEG VCM-aligned neural feature compression for remote surveillance, satellite backhaul, and edge AI.**

Sentinel compresses ResNet18 detector features (256 channels) down to 8-channel latents at **0.173 BPP** — a **181-195×** reduction vs raw f32 features — while preserving downstream detection accuracy within <5% mAP drop.

## Quick Start

```bash
# Download Docker image
# https://github.com/MAHAMAIA/lewm-vc/releases/tag/v0.1.1
cat sentinel-0.1.0.tar.gz.part.* | gunzip -c | docker load

# Encode frames to feature bitstream
docker run --rm \
  -v /path/to/model.pt:/app/model.pt \
  -v /data:/data \
  sentinel:0.1.0 feature encode \
    --model /app/model.pt \
    --input /data/frames \
    --output /data/bitstream.bin

# Decode back to features
docker run --rm \
  -v /path/to/model.pt:/app/model.pt \
  -v /data:/data \
  sentinel:0.1.0 feature decode \
    --model /app/model.pt \
    --input /data/bitstream.bin \
    --output /data/recon_features.pt
```

## Performance

| Metric | Value | Target |
|--------|-------|--------|
| Bitrate (real torchac) | **0.173 BPP** | 0.08–0.25 BPP |
| Compression ratio | **181–195×** vs raw f32 | — |
| Feature PSNR | **17.6 dB** | >15 dB |
| Cosine similarity | **0.77** | >0.75 |
| Model size | **331K params** (1.3 MB) | Edge-deployable |
| Inference | **500+ fps** (GPU) | Real-time |
| Detection mAP drop | **< 5%** (target) | MPEG VCM standard |

## Architecture

```
Input frame (256×256)
  → ResNet18 layer3 → [256, 16, 16] features
    → FeatureCompressor (Conv2d 256→64, GELU, Conv2d 64→8) → [8, 16, 16]
      → Quantizer (256 levels, step=0.007812)
        → Hyperprior entropy model
          → torchac arithmetic coding → bitstream at 0.173 BPP
    → FeatureDecompressor (Conv2d 8→64, GELU, Conv2d 64→256) → [256, 16, 16]
      → Detector Layer4 → FPN → RPN → ROI heads → detections
```

## Repository Structure

```
├── src/lewm_vc/
│   ├── cli.py                   # Sentinel CLI (encode/decode/info)
│   ├── feature_compress.py      # Compressor + decompressor + backbone
│   ├── entropy.py               # Hyperprior entropy model
│   └── quant.py                 # Scalar quantizer
├── scripts/
│   ├── train_feature_compress.py   # Training pipeline
│   ├── eval_feature_compress.py    # Feature-space evaluation
│   ├── encode_feature.py           # Bitstream encoder
│   ├── decode_feature.py           # Bitstream decoder
│   └── demo_feature_compress.py    # Visualization demo
├── configs/
│   └── train_feature_compress.yaml # Training config
├── reviews/
│   ├── customer-deployment-workflow.md    # 3-step customer PoC
│   ├── customer-poc-requirements.md       # Prerequisites & runbook
│   └── training-run-summary-feature-compress.md
├── Dockerfile.sentinel          # Docker packaging
└── pyproject.toml               # Entry point: sentinel CLI
```

## Training

```bash
python scripts/train_feature_compress.py \
  --config configs/train_feature_compress.yaml
```

3-phase training:
| Phase | Steps | λ | Purpose |
|-------|-------|---|---------|
| 0 (warmup) | 2k | — | Feature reconstruction only |
| 1 (RD) | 10k | 5.0→0.5 | Rate-distortion optimization |
| 2 (low bitrate) | 5k | 10.0 | Push BPP toward 0.08 |

## Customer Deployment

See [`reviews/customer-deployment-workflow.md`](reviews/customer-deployment-workflow.md) for the 3-step validation pipeline.

**What we need from the customer:**
- 500+ labeled frames with ground-truth annotations
- Their detection model architecture
- 3+ camera angles/scenes

**What we provide:**
- Docker image with sentinel CLI
- Model checkpoint (1.3 MB)
- Feature-space eval scripts

## Dataset

10,000 diverse VIRAT frames available on GitHub Releases as `virat_10k.tar.gz` (1.3 GB):

```bash
curl -L -o virat_10k.tar.gz \
  "https://github.com/MAHAMAIA/lewm-vc/releases/download/v0.1.1/virat_10k.tar.gz"
```

## Key Results

- **0.173 BPP** real torchac bitrate (NLL estimate overestimates by ~15%)
- **181–195×** compression vs raw 256×16×16 f32 features
- **17.6 dB** feature PSNR, **0.77** cosine similarity
- **331K parameters** (compressor: 148K, decompressor: 148K, entropy: 35K)
- **25 fps** on CPU, **500+ fps** on GPU
- Near-zero per-frame variance: ±0.005 BPP, ±0.1 dB PSNR across diverse scenes

## License

MIT
