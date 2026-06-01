# LeWM-VC Training Run — Feature Compression (Phase 1 + Phase 2)

> **Feature compression for MPEG VCM-aligned machine perception.** This replaces the dead-end pixel autoencoder approach. Compresses ResNet18 layer3 features (256ch) to 8-channel latent via learned compressor/decompressor + hyperprior entropy model. Real BPP via torchac arithmetic coding.

## What Ran

Config: `configs/train_feature_compress.yaml`. Training on VIRAT frames (256×256), ResNet18 backbone frozen.

| Phase | Steps | λ | Rate Weight | Trainable | Frozen |
|-------|-------|---|-------------|-----------|--------|
| 0 (warmup) | 2k | — | 0.0 | compressor, decompressor | backbone, quantizer, entropy_model |
| 1 (RD) | 10k | 5.0 → 0.5 | 1.0 | compressor, decompressor, entropy_model | backbone, quantizer |
| 2 (low bitrate) | 5k | 10.0 | 1.0 | compressor, decompressor, entropy_model | backbone, quantizer |

## Results

| Metric | Phase 1 (step 10000) | Phase 2 (step 15000) | Target |
|--------|---------------------|---------------------|--------|
| NLL BPP | 0.203 | 0.199 | 0.08–0.25 |
| Real torchac BPP | 0.174 | 0.173 | 0.08–0.25 |
| Feature PSNR | 17.64 dB | 17.67 dB | >15 dB |
| Cosine similarity | 0.770 | 0.772 | >0.75 |
| Compression ratio | 179× | 181–195× | — |
| Inference speed | ~500 fps (GPU) | ~500 fps (GPU) | — |
| Model params | 331,032 | 331,032 | — |
| Checkpoint size | 1.3 MB | 1.3 MB | — |

## Observations

- **Real BPP is ~15% lower than NLL estimate.** Actual arithmetic coding achieves 0.173 BPP vs NLL estimate of 0.199 BPP. The entropy model overestimates bitrate.
- **Phase 2 (λ=10) barely improved BPP** (0.203 → 0.199 NLL). The 8-channel latent bottleneck appears saturated — further improvement may require:
  - Smaller latent dim (4 or 2 channels)
  - More aggressive entropy model (e.g., autoregressive context model)
  - More training steps at higher λ (needs 50k+ steps)
- **Bottom-right spatial bias** in difference maps is content-dependent, not architectural. Confirmed via spatial uniformity diagnostic — backbone and compressor are perfectly uniform.
- **Near-zero per-frame variance**: BPP varies by ±0.005 across 100 diverse scenes. feat_PSNR varies by ±0.1 dB. Deterministic inference.

## Architecture

```
Input frame (256×256 RGB)
  → ResNet18 layer3 → [B, 256, 16, 16] features
    → Compressor (Conv2d 256→64, GELU, Conv2d 64→8) → [B, 8, 16, 16]
      → Quantizer (256 levels, step=0.007812)
        → Entropy model (hyperprior, 2 Gaussians)
          → torchac arithmetic coding → bitstream at 0.173 BPP
    → Decompressor (Conv2d 8→64, GELU, Conv2d 64→256) → [B, 256, 16, 16]
```

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| ResNet18 backbone | 256ch at 16×16 → compressor matches perfectly |
| 8-channel latent | Sweet spot: 0.173 BPP while preserving features |
| 3×3 conv kernel | Spatial context capture; padding=1 avoids edge shrinkage |
| GELU activation | Smooth gradients vs ReLU, empirically better for low-bitrate |
| Hyperprior entropy | Scale hyperprior captures spatial dependencies at minimal cost |
| torchac arithmetic coding | Real bitstream output, not NLL estimate |
