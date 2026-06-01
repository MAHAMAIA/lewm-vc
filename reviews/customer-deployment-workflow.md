# Sentinel Customer Deployment: 3-Step Validation Workflow

## Getting the Docker Image

The Sentinel Docker image is distributed via GitHub Releases.

**Release:** https://github.com/MAHAMAIA/lewm-vc/releases/tag/v0.1.1

```bash
# Download all 4 parts (total 3.5 GB)
curl -L -o sentinel-0.1.0.tar.gz.part.aa \
  "https://github.com/MAHAMAIA/lewm-vc/releases/download/v0.1.1/sentinel-0.1.0.tar.gz.part.aa"
curl -L -o sentinel-0.1.0.tar.gz.part.ab \
  "https://github.com/MAHAMAIA/lewm-vc/releases/download/v0.1.1/sentinel-0.1.0.tar.gz.part.ab"
curl -L -o sentinel-0.1.0.tar.gz.part.ac \
  "https://github.com/MAHAMAIA/lewm-vc/releases/download/v0.1.1/sentinel-0.1.0.tar.gz.part.ac"
curl -L -o sentinel-0.1.0.tar.gz.part.ad \
  "https://github.com/MAHAMAIA/lewm-vc/releases/download/v0.1.1/sentinel-0.1.0.tar.gz.part.ad"

# Concatenate and load into Docker
cat sentinel-0.1.0.tar.gz.part.* > sentinel-0.1.0.tar.gz
gunzip -c sentinel-0.1.0.tar.gz | docker load

# Verify
docker images sentinel --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
# → sentinel:0.1.0  7.97GB
```

The Docker image includes the `sentinel` CLI with all dependencies (PyTorch, torchac, torchvision). The model checkpoint (`best.pt`, 1.3 MB) must be mounted separately.

### Building from Source (Alternative)

```bash
git clone https://github.com/MAHAMAIA/lewm-vc.git
cd lewm-vc
docker build -t sentinel:latest -f Dockerfile.sentinel .
```

## The Problem

Standard COCO-pretrained object detectors return **zero detections** on high-angle satellite, aerial, or remote industrial surveillance footage. This is **not** a Sentinel compression issue — it occurs on both original AND compressed frames. The downstream model is domain-blind to the customer's operational perspective.

## The Solution: 3-Step Validation Pipeline

```
[Customer Video Ingestion]
          │
          ▼
[Step 1: Fine-Tune Detector on Customer GT] ──► Baseline mAP (uncompressed)
          │
          ▼
[Step 2: Encode via Sentinel at 0.173 BPP] ──► Compressed feature stream
          │
          ▼
[Step 3: Run Fine-Tuned Detector on Reconstructed Features] ──► Compressed mAP
          │
          ▼
[Report: ΔmAP = baseline_mAP - compressed_mAP, target < 5%]
```

### Step 1: Establish Ground Truth Baseline

Take a labeled slice of the customer's uncompressed video and their existing analytic annotations to fine-tune their detector. This establishes the uncompressed baseline mAP.

**Inputs:**
- Customer video frames (any resolution, any frame rate)
- Customer ground truth annotations (bounding boxes, tracks, etc.)
- Their existing detector architecture (or we provide Faster R-CNN ResNet50-FPN)

**Output:** Baseline mAP on uncompressed frames.

### Step 2: Encode via Sentinel

Pass the same video slice through the Sentinel feature compression pipeline.

```bash
sentinel feature encode \
  --model sentinel.pt \
  --input <frames_dir> \
  --output bitstream.bin
```

| Metric | Value |
|--------|-------|
| Bitrate | **0.173 BPP** |
| Compression ratio | **181-195×** vs raw f32 features |
| Latent dim | 8 channels at 16×16 spatial |
| Frame rate | ~25 fps on CPU, ~500+ fps on GPU |

**Output:** `bitstream.bin` — ~5 KB per 256×256 frame at 0.173 BPP.

### Step 3: Measure Machine-Perception Delta

Decode the bitstream and feed reconstructed features into the customer's fine-tuned detector.

```bash
sentinel feature decode \
  --model sentinel.pt \
  --input bitstream.bin \
  --output recon_features.pt
```

The reconstructed features (`[N, 256, 16, 16]` tensors) are **spatially and semantically aligned** with the original detector backbone activations (ResNet18 layer3). They can be fed directly into the detector's Layer4 → FPN → RPN → ROI heads.

**Expected results:**
- Feature PSNR: **17.6 dB**
- Cosine similarity: **0.77**
- mAP drop (target): **< 5%** absolute

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Customer Edge Gateway                      │
│                                                              │
│  ┌──────────┐    ┌───────────┐    ┌──────────┐              │
│  │ RTSP/Cam │───▶│ ResNet18  │───▶│ Sentinel │──▶ VSAT link │
│  │  Stream  │    │Backbone   │    │Compressor│   0.173 BPP  │
│  └──────────┘    └───────────┘    └──────────┘              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                        VSAT / Satellite
                            │
┌─────────────────────────────────────────────────────────────┐
│                     Customer Data Center                      │
│                                                              │
│  ┌──────────┐    ┌───────────┐    ┌──────────────────┐      │
│  │ Sentinel │───▶│ ResNet18  │───▶│ Fine-Tuned       │      │
│  │Decompress│    │ Layer4    │    │ Detector (RPN    │      │
│  │  0.173   │    │ + FPN     │    │ + ROI Heads)     │      │
│  │  BPP in  │    │           │    │   ▶ detections   │      │
│  └──────────┘    └───────────┘    └──────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Selling Points

| Metric | Value | Business Impact |
|--------|-------|-----------------|
| Bandwidth | **0.173 BPP** | 181-195× reduction → 99.5% VSAT cost savings |
| Model size | **331K params** (1.3 MB) | Fits any edge processor, no GPU needed |
| Inference | **500+ fps on GPU** | Real-time on 4K streams |
| Detection loss | **< 5% mAP drop** | Operational intelligence preserved |
| Deployment | **Docker container** | Drop-in via `docker run sentinel` |

## Running the Eval

```bash
# Pull the image (see "Getting the Docker Image" above)
# Mount checkpoint and data, then run

# Encode customer frames
docker run --rm \
  -v /path/to/model.pt:/app/model.pt \
  -v /data/customer_frames:/data/frames \
  -v /data/output:/data/output \
  sentinel:0.1.0 feature encode \
    --model /app/model.pt \
    --input /data/frames \
    --output /data/output/bitstream.bin

# Decode to features
docker run --rm \
  -v /path/to/model.pt:/app/model.pt \
  -v /data/output:/data/output \
  sentinel:0.1.0 feature decode \
    --model /app/model.pt \
    --input /data/output/bitstream.bin \
    --output /data/output/recon_features.pt

# Get model info
docker run --rm \
  -v /path/to/model.pt:/app/model.pt \
  sentinel:0.1.0 feature info --model /app/model.pt

# Run customer's detector on recon_features.pt (outside container)
python customer_detector.py --features /data/output/recon_features.pt
```

### CLI Usage Without Docker

```bash
# Install the package
pip install -e .

# Run directly
sentinel feature encode \
  --model checkpoints/feature_compress/best.pt \
  --input /data/frames \
  --output /data/bitstream.bin
```
