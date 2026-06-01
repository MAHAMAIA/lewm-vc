# Sentinel Customer PoC: Prerequisites & Runbook

## Overview

A customer proof-of-concept validates that Sentinel's feature compression (0.173 BPP, 181–195×) preserves their downstream detection accuracy within <5% mAP drop. This doc covers what's needed from the customer, what we provide, and the exact runbook.

## Prerequisites — What We Need From the Customer

### 1. Video Data
| Item | Format | Notes |
|------|--------|-------|
| Raw video stream | MP4, RTSP, or image sequence | Any resolution, any FPS. Representative of deployment environment. |
| Minimum samples | ≥500 labeled frames | Enough to fine-tune detector. More is better. |
| Scene diversity | ≥3 different scenes/camera angles | Ensures compression generalizes across their deployment. |

### 2. Ground Truth Annotations
| Item | Format | Notes |
|------|--------|-------|
| Bounding boxes | COCO JSON, YOLO TXT, or Pascal VOC XML | Object detection labels for each frame. |
| Object classes | List of class names + IDs | Customer's specific detection classes (vehicles, people, equipment, etc.). |
| Minimum annotations | ≥500 objects across classes | Enough to establish baseline mAP. |

### 3. Detection Model
| Item | Preferred | Alternative |
|------|-----------|-------------|
| Architecture | Faster R-CNN (ResNet50/ResNet18) | Any detector that can accept external backbone features |
| Framework | PyTorch | ONNX (requires export/import step) |
| Weights | Fine-tuned on customer data | COCO-pretrained (will underperform on deployment domain) |
| Training script | Customer provides or we fine-tune | We can fine-tune given annotations |

## What We Provide

| Artifact | Location | Description |
|----------|----------|-------------|
| **Model checkpoint** | `checkpoints/feature_compress/best.pt` (1.3 MB) | Phase 2 trained compressor (331K params) |
| **Docker image** | `sentinel:0.1.0` (7.97 GB) | CLI + all dependencies. Ready to run. |
| **Sentinel CLI** | `sentinel feature encode/decode/info` | Encode frames → bitstream; decode bitstream → features |
| **Eval script** | `scripts/vcm_eval_resnet18.py` | Feature-space Faster R-CNN eval with match/IoU metrics |
| **Integration guide** | `reviews/customer-deployment-workflow.md` | 3-step validation pipeline |

## Runbook

### Step 0: Setup

```bash
# On customer's GPU server (or our droplet)
docker pull sentinel:0.1.0  # or docker load < sentinel-0.1.0.tar.gz
```

### Step 1: Fine-Tune Detector (Establish Baseline mAP)

**Goal:** Establish uncompressed baseline on customer's domain.

```bash
# Customer provides: frames + annotations + detector training script
# We provide: baseline mAP computation
python train_detector.py \
  --data /data/customer_frames/ \
  --annotations /data/customer_annotations.json \
  --backbone resnet18 \
  --output /models/customer_detector.pt

# Evaluate baseline
python eval_detector.py \
  --model /models/customer_detector.pt \
  --data /data/customer_val/ \
  --annotations /data/customer_val_annotations.json
# → baseline_mAP
```

**Expected output:** Baseline mAP (e.g., 0.65 mAP@0.5).

### Step 2: Encode Frames via Sentinel

**Goal:** Compress customer frames to 0.173 BPP feature bitstream.

```bash
sentinel feature encode \
  --model checkpoints/feature_compress/best.pt \
  --input /data/customer_frames/ \
  --output /data/compressed_bitstream.bin

# Verify compression
sentinel feature info --model checkpoints/feature_compress/best.pt
```

**Expected output:** Bitstream at ~5 KB per 256×256 frame.

### Step 3: Reconstruct Features & Re-evaluate

**Goal:** Measure detection mAP on reconstructed features.

```bash
# Decode bitstream to feature tensors
sentinel feature decode \
  --model checkpoints/feature_compress/best.pt \
  --input /data/compressed_bitstream.bin \
  --output /data/recon_features.pt

# Run customer's fine-tuned detector on reconstructed features
# (features are [N, 256, 16, 16] tensors, spatially aligned with ResNet18 layer3)
python eval_detector_on_features.py \
  --model /models/customer_detector.pt \
  --features /data/recon_features.pt \
  --annotations /data/customer_val_annotations.json
# → compressed_mAP
```

**Expected output:** Compressed mAP.

### Step 4: Report

```bash
ΔmAP = baseline_mAP - compressed_mAP
```

| ΔmAP | Verdict |
|------|---------|
| < 2% | **Excellent** — compression is transparent to detector |
| 2–5% | **Good** — acceptable for production deployment |
| 5–10% | **Moderate** — may need higher bitrate or detector fine-tuning on compressed features |
| > 10% | **Poor** — investigate: latent dim too small? detector architecture mismatch? |

## Feature-Space Detection Integration

For detectors that accept external features (Faster R-CNN with ResNet backbone), the compressed features plug in directly:

```
┌─────────────────────────────────────────────┐
│         Customer Detector (Modified)          │
│                                               │
│  Input frame                                  │
│    → ResNet18 layer1, layer2 (shared)         │
│    → ResNet18 layer3                          │
│       ↓                                       │
│    [INTERCEPT HERE] ←── Sentinel recon features│
│       ↓                                       │
│    → ResNet18 layer4                          │
│    → FPN → RPN → ROI heads → detections       │
└─────────────────────────────────────────────┘
```

The `scripts/vcm_eval_resnet18.py` script demonstrates this integration for Faster R-CNN. Adapt to any ResNet-backbone detector by intercepting at `backbone.body.layer3`.

## Success Criteria

The PoC is successful if:
1. **Sentinel compresses customer frames at 0.15–0.20 BPP** (real torchac)
2. **Feature fidelity is ≥17 dB PSNR and ≥0.75 cosine similarity**
3. **ΔmAP < 5%** between uncompressed and compressed detection
4. **Customer can run inference at real-time rates** (≥30 fps on their hardware)

## Failure Modes & Mitigations

| Failure | Likely Cause | Mitigation |
|---------|-------------|------------|
| Detector returns 0 detections | COCO-pretrained detector on OOD domain | Fine-tune detector on customer data first |
| ΔmAP > 10% | Compressor latent dim too small for complex scenes | Increase latent_dim to 16 or 32; retrain |
| BPP > 0.25 | High-entropy scenes (e.g., dense foliage) | Enable temporal prediction (inter-frame coding) |
| Low fps | CPU inference on edge device | Use GPU; or reduce image size to 128×128 |

## Tools Reference

| Script | Purpose |
|--------|---------|
| `scripts/vcm_eval_resnet18.py` | Feature-space Faster R-CNN eval with detection match/IoU |
| `scripts/vcm_eval_final3.py` | ResNet50-native eval (requires 1024→256 adapter) |
| `scripts/demo_feature_compress.py` | Visualization demo with classification comparison |
| `scripts/eval_feature_compress.py` | Feature-space metrics (PSNR, cosine, NLL BPP) |
