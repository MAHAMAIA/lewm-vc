# Sentinel Training — Data & Curriculum Plan

## Current Checkpoint Status

> **Update (June 2026):** The checkpoints listed below were trained on random noise (stub `VideoDataset.__getitem__` bug) and are obsolete. The current run `sentinel-p1-l60.0-85e9d5` is training a new base codec from the partially-trained `5d285a` checkpoint with corrected λ=60, γ≈3e-5. By Phase 4, we will have a valid base codec trained on real VIRAT data. The `5d285a` run completed all phases but had catastrophic loss imbalance (98.5% JEPA) — that has been corrected in the current run.

| Checkpoint | Size | What It Is | Status |
|------------|------|------------|--------|
| `ae_lambda_0.05_final.pt` | 28 MB | Intra-frame autoencoder, single lambda=0.05 | ❌ Obsolete — trained on random noise |
| `entropy_lambda_0.05_final.pt` | 20 MB | Entropy model, single lambda=0.05 | ❌ Obsolete — trained on random noise |
| `temporal_final.pt` | 80 MB | JEPA temporal predictor | ❌ Obsolete — trained on random noise |
| `85e9d5` (current run) | ~300 MB | Base codec, λ=60, γ=3e-5, real VIRAT data | ✅ In Phase 1 — loss balanced, training healthy |

**Gap**: All checkpoints were trained on random noise, not actual video. Retraining on real surveillance footage is in progress with the current run.

---

## Training Gaps vs Shipping Requirements

| Need | Why | Priority |
|------|-----|----------|
| **JEPA compression validation** | No published benchmarks exist for JEPA as a video compression technique (only as a representation learning method). We are pioneering this application — there is no prior art to benchmark against. Success requires proving temporal coding gain (P-frames cheaper than I-frames) on real surveillance footage. | **P0** |
| **Dual-layer SVC** | `src/lewm_vc/svc.py` (LatentSplitter, LatentFuser, SVCDecoder) exists but never trained. BL/EL decomposition must be learned end-to-end | **P0** |
| **Multi-lambda** | Rate control needs 3-5 lambda points (0.02, 0.05, 0.08, 0.12, 0.20), not just 0.05 | **P1** |
| **QAT (Phase 2)** | INT8 quantization + TensorRT export required for Jetson Orin NX edge deployment | **P1** |

---

## Recommended Datasets

### Core Datasets (MPEG VCM CTC Alignment)

| Dataset | Task | Size | Why Relevant | Access |
|---------|------|------|--------------|--------|
| **VIRAT Video Dataset 2.0** | Object detection, anomaly detection, temporal modeling | ~80-120 GB | Best match: realistic outdoor surveillance, stationary cameras, vehicles/people — close to pipeline/offshore platform monitoring | [viratdata.org](https://viratdata.org/) — request via Kitware |
| **SFU-HW-Objects-v1** | Object detection (MPEG VCM mandatory) | ~5-10 GB | Official MPEG VCM CTC. Outdoor/industrial-style scenes used in standards evaluation | [Mendeley Data](https://data.mendeley.com/datasets/hwm673bv4m) |
| **TVD (Tencent Video Dataset)** | Object tracking (MPEG VCM mandatory) | ~15-25 GB | Official MPEG VCM CTC. High-res sequences for testing JEPA temporal predictor | [multimedia.tencent.com](https://multimedia.tencent.com/en/resources/tvd/) |
| **PEViD-HD** | Privacy evaluation, outdoor/indoor surveillance | ~0.8 GB | Good supplement — realistic human activity scenes | EPFL FTP (`tremplin.epfl.ch`) |

### Supplementary Datasets

| Dataset | Size | Why |
|---------|------|-----|
| **UCF-Crime** | ~50 GB | Real-world anomaly surveillance for rare events (leaks, intrusions) |
| **Custom data** | TBD | Pilot footage from design partners — most valuable long-term data |

### Storage Requirements

| Scope | Raw Video | With Preprocessing |
|-------|-----------|-------------------|
| Minimum (retraining) | 60-180 GB (50-150 hrs) | 120-300 GB |
| Pilot-ready | 180-350 GB (150-300 hrs) | 300-500 GB |
| Full production | 600 GB - 2 TB (500-1500+ hrs) | 1-3 TB |

### Data Volume Rationale

- Model is 14.7M params (ViT-Tiny) — doesn't need massive data like big V-JEPA (1M+ hours)
- Domain-specific focus (remote industrial: static cameras, weather, low motion, anomalies) reduces data requirements
- JEPA self-supervised learning is sample-efficient compared to generative models
- **Quality > Quantity**: 100 hrs of highly relevant remote industrial footage beats 1000 hrs of generic YouTube

---

## Data Acquisition + Preprocessing Checklist

### Phase 1: Acquisition (Week 1)
- [ ] VIRAT Video Dataset 2.0 — request access, download (~80-120 GB)
- [ ] SFU-HW-Objects-v1 — download from Mendeley (~5-10 GB)
- [ ] TVD — download from Tencent Multimedia (~15-25 GB)
- [ ] PEViD-HD — download from EPFL FTP (~0.8 GB)
- [ ] UCF-Crime (optional) — for anomaly robustness (~50 GB)
- [ ] Create folder structure:
  ```
  data/
  ├── raw/
  │   ├── virat/
  │   ├── sfu/
  │   ├── tvd/
  │   └── pevid/
  ├── processed/
  └── annotations/
  ```

### Phase 2: Preprocessing (Week 1-2)
- [ ] Decode videos to frames or use efficient video reader (`decord` preferred)
- [ ] Standardize resolution: 1080p or 720p
- [ ] Filter clips: remove completely static scenes (<5% motion) and extreme low-quality
- [ ] Split: 70% train / 15% val / 15% test (holdout for VCM-style evaluation)
- [ ] Add domain-specific augmentations:
  - Rain, fog, salt spray simulation (offshore environment)
  - Camera shake, low light
  - Compression artifacts (H.265 at various QPs)
- [ ] Generate metadata (motion level, scene type, time of day)
- [ ] Write proper `VideoDataset.__getitem__` — decode actual video frames, not `torch.rand`

**Tool recommendations**: `decord`, `torchvision.io`, or `ffmpeg-python` for video loading.

---

## Training Curriculum — Reverse-Optimized for ~100 Hours on MI300X

### Phase 0: Data Prep & Warm-up (12 hours)
- Finalize `VideoDataset` with VIRAT + SFU
- Train Intra-frame Autoencoder + Entropy Model (lambda=0.05)
- Goal: Replace random-noise checkpoints with real-video baseline

> **Completed on VIRAT**. The current run `85e9d5` is in Phase 1 (joint RD) with λ=60, γ≈3e-5 — loss is balanced (~130-200). Phase 0 on SFU/TVD remains pending.

### Phase 1: Core Temporal Model (25 hours)
- Train full base model: Autoencoder + JEPA Predictor + Entropy
- 3 lambdas in parallel (0.03, 0.05, 0.08) — MI300X 192GB HBM3 can handle concurrent runs
- Dataset: 80-120 hrs (VIRAT + SFU + TVD subset)

### Phase 2: Dual-layer SVC — Highest ROI (35 hours)
- Freeze encoder + JEPA predictor
- Train LatentSplitter + LatentFuser end-to-end
- Loss: reconstruction loss + task loss (YOLO mAP preservation on BL-only decode)
- This is the biggest differentiator — forensic two-stream capability depends on it

### Phase 3: Rate Control & Polish (18 hours)
- Add 2 more lambdas (0.02, 0.12) for 5-point RD curve
- Light fine-tuning on mixed data
- Implement rate control switching logic

### Phase 4: QAT + Export (10 hours)
- Quantization-Aware Training (INT8)
- TensorRT engine export for Jetson Orin NX
- Final validation on holdout set

**Total: ~100 hours**

### Key Optimizations to Fit 100h Budget

- **Parallel training**: MI300X 192GB HBM3 supports 3-4 concurrent lambda runs
- **Prioritize SVC**: 35% of budget on dual-layer SVC — your forensic moat
- **Data efficiency**: 80-120 hrs with aggressive augmentations, not massive datasets
- **Freezing strategy**: Freeze early layers after Phase 1 to speed later phases
- **Mixed precision (bf16)** + gradient checkpointing for throughput

### Expected Results After 100 Hours

- Pilot-viable model with 50%+ bandwidth savings on real surveillance footage
- Working Dual-layer SVC (Base semantic latents continuously streamed, Enhancement residuals stored locally)
- 3-5 rate points for basic rate control
- INT8 TensorRT model ready for Jetson Orin NX 16GB

---

## Training Order Summary

1. **Intra-frame + Entropy** → fix random-noise baseline
2. **JEPA Temporal Predictor** → enable inter-frame compression
3. **Dual-layer SVC** (splitter + fuser) → forensic two-stream capability
4. **Multi-lambda** → rate control flexibility
5. **QAT + TensorRT export** → edge deployment ready

---

## MI300X Training Time Estimates

| Phase | Hours | Notes |
|-------|-------|-------|
| Phase 0: Warm-up | 12 | Single lambda, small model |
| Phase 1: Base Model | 25 | 3 lambdas in parallel |
| Phase 2: Dual-layer SVC | 35 | Most compute-heavy, end-to-end |
| Phase 3: Multi-lambda | 18 | 2 additional lambdas + fine-tuning |
| Phase 4: QAT + Export | 10 | INT8 quantization, TensorRT |
| **Total** | **~100** | ~5-6 weeks wall-clock at full utilization |

MI300X (192GB HBM3) is overkill for 14.7M-param ViT-Tiny but enables:
- Large batch sizes (16-32)
- Parallel multi-lambda training (3-4 concurrent runs)
- Fast experimentation cycles

### Evaluation Protocol (MPEG VCM CTC Standard)

- **Detection**: Faster R-CNN X101-FPN (via Detectron2) on SFU
- **Tracking**: JDE-1088×608 on TVD
- **Metric**: BD-Rate (bitrate savings at same task accuracy) across AI/RA/LD configurations
- **Report**: mAP (detection), MOTA (tracking) vs bitrate

---

## Key Design Decisions

- **Single lambda models** rather than conditional coding — simpler, proven approach for P0
- **Channel-wise split** for BL/EL (first 64ch → BL, remaining 128ch → EL) with optional learned projections
- **YOLO-style probe** for task loss during SVC training — directly optimizes for machine vision, not human viewing
- **INT8 + TensorRT** as target edge format — Jetson Orin NX DLAs support this natively
- **Surprise-gated bit allocation** — a novel feature not present in standards (JPEG AI, MPEG VCM). Uses VOE predictor output to allocate more bits to anomalous/surprising frames. To be implemented post-SVC.
