# LeWM-VC Training Run V1 — sentinel-p1-l0.05-be7de7 → sentinel-p2-l0.05-c8ce00

## Overview

First complete end-to-end training on 319 VIRAT clips (903,919 frames). Two sequential runs:
- **be7de7**: Phase 0 (resumed from Phase 0 checkpoint) → Phase 1 (25k steps, entropy warmup)
- **c8ce00**: Phase 2 (QAT, 15k) → Phase 3 (decoder refine, 15k) → Phase 4 (cooldown, 20k)

## Key Training Parameters

| Parameter | Value |
|-----------|-------|
| λ (RD weight) | 0.05 |
| γ (JEPA weight) | 1.0 (Phase 1-2), 0.5 (Phase 3-4) |
| δ (SIGReg weight) | 0.0005 |
| LR | 1e-4, cosine decay over 125k total steps |
| Precision | fp32 |
| Quantizer | Hard rounding (Phase 2+) |

## Known Bugs During Run

| Bug | Impact | Fixed? |
|-----|--------|--------|
| Entropy model final layer zero-initted → mu=0 permanently | Entropy collapsed (0 BPP) | ✅ (sigma-only zero init + re-randomize on load) |
| Decoder hidden_dim=128 vs checkpoint hidden_dim=512 | Decoder weights silently not loaded | ✅ (restored to 512) |
| Entropy hyper_channels=320 vs checkpoint 256 | Entropy weights silently not loaded | ✅ (restored to 256) |
| ReLU kills gradient to mu channels through 4 layers | Mu gradient died after 1 layer | ✅ (switched to GELU) |
| λ=0.05 while R=60 bits, D=0.07 → λD=0.0035 | Distortion contributed nothing | ✅ (λ=60, γ=3e-5 in current run) |

## Results

| Metric | Value | Interpretation |
|--------|-------|---------------|
| PSNR | 11.71 dB | Constant-gray output |
| BPP | 0.0009 (I), 0.0000 (P) | Entropy collapsed |
| JEPA contribution | 98.5% of total loss | Model optimized for prediction, not compression |

## Root Causes

1. **Entropy model mu never trained** (zero-init bug + ReLU gradient kill)
2. **RD loss unbalanced** (λ 1.2 billion× too small relative to JEPA weight γ)
3. **Architecture mismatches** between evaluation code and checkpoint (decoder, entropy dimensions)
4. **Temporal evaluation mixed clips** (incorrect `rglob` in evaluate.py)
