# LeWM-VC Training Run Summary — sentinel-p2-l0.05-5d285a

> **Note:** This run had PSNR 11.7 dB due to loss imbalance (γJ >> λD). The follow‑up run `sentinel-p1-l60.0-85e9d5` (June 2026) fixes this with λ=60, γ=3e-5 — loss is now ~130-200 and all loss components are within 2× of each other.

## What Ran

All 4 phases completed on 319 VIRAT training clips (903,919 frames), 68 validation clips.

| Phase | Steps | Trainable | Frozen | Lambda | Gamma |
|-------|-------|-----------|--------|--------|-------|
| 0 (warmup) | 50k | encoder, predictor, decoder | entropy, quantizer, rate_control | 1.0 | 2.0 |
| 1 (entropy warmup) | 25k | entropy_model | encoder, predictor, decoder | 0.05 | 1.0 |
| 2 (QAT) | 15k | decoder, entropy_model | encoder, predictor | 0.05 | 1.0 |
| 3 (decoder refine) | 15k | decoder | encoder, predictor, entropy_model | 0.05 | 0.5 |
| 4 (cooldown) | 20k | decoder, entropy_model | encoder, predictor | 0.05 | 0.5 |

## Results

| Metric | Value | Verdict |
|--------|-------|---------|
| PSNR | 11.71 dB | **Terrible** |
| MS-SSIM | 0.1826 | **Terrible** |
| LPIPS | 0.5762 | **Terrible** |
| I-frame BPP | 0.0009 | Near-zero |
| P-frame BPP | 0.0000 | Near-zero |

## Root Cause: Loss Imbalance

Over a 16-frame sequence, the loss contributions were:

| Component | Value | % of total | Should be |
|-----------|-------|-----------|-----------|
| γ × JEPA (γ=1.0) | 4,168,085 | **98.5%** | ~25-50% |
| λ × MSE (λ=0.05) | 5e-5 | 0.0% | ~25-50% |
| Rate (bits) | 61 | 0.0% | ~25-50% |

The JEPA loss dominated because:
1. **γ=1.0** applied to a JEPA loss of ~4M per step
2. **λ=0.05** applied to MSE of ~0.07 gave λD = 0.0035
3. Ratio: γJ / λD ≈ **1.2 billion to 1**

## Current Fix (Running Now)

| Parameter | Old | New |
|-----------|-----|-----|
| λ | 0.05 | 60 |
| γ (Phase 1-2) | 1.0 | 3e-5 |
| γ (Phase 3-4) | 0.5 | 1e-5 |

Expected contributions with fixed values:
- Rate: 61 (24%)
- λ × MSE: 65 (26%)  
- γ × JEPA: 125 (50%)

All within 2× of each other. Estimate: ~5h remaining.
