# JEPA Temporal Prediction for Feature Compressor

## Why Now

The feature compressor produces an 8x16x16 latent (2,048 elements per frame). This is 100x smaller than the old pixel autoencoder latent (192x16x16 = 49,152) and structurally aligned with the detection task. Temporal prediction on this space is tractable.

Current all-intra BPP: 0.173
Target P-frame BPP with JEPA residual: ~0.05
Target GOP-16 average: (0.173 + 15 * 0.05) / 16 = 0.058

## What Exists in Repo

| Asset | Location | Status |
|-------|----------|--------|
| LeWMPredictor | `src/lewm_vc/predictor.py` | 8 layer transformer, 256 hidden, 4 heads. Expects 192 latent dim. Needs adaptation for 8 dim. |
| HyperpriorEntropy | `src/lewm_vc/entropy.py` | 5 layer CNN, predicts mu/sigma. Can code residuals conditioned on prediction. |
| Quantizer | `src/lewm_vc/quant.py` | Already used in feature compressor. No changes needed. |
| NAL types | `src/lewm_vc/bitstream/writer.py` | Already has `P_RESIDUAL` (4), `BL_I` (7), `BL_P` (8). Ready. |
| Temporal dataset | `src/lewm_vc/data/dataset.py` | FrameDataset supports `sequence_length > 1`. Ready. |
| Feature compressor | `src/lewm_vc/feature_compress.py` | Compressor + decompressor. Frozen. |
| Training script | `scripts/train_feature_compress.py` | Current 3 phase all-intra training. Needs temporal phase. |
| VIRAT data | `datasets/virat/frames/` | Sequential frames organized by scene. Each scene has frame_0001.png through frame_NNNN.png. |

## Plan

### Phase 1: Adapt Predictor (Week 1)

Take the existing LeWMPredictor and adapt it for 8 channel latent:

- Create `src/lewm_vc/predictor_feature.py`
- Input projection: `Conv2d(8, 128, 1)` (reduced from 256 since latent is smaller)
- Transformer: 4 layers, 4 heads, 128 hidden (scaled down, task is easier)
- Output heads: mean and log_std per channel, per spatial position
- Keep the same structure: temporal pooling -> transformer -> spatial conv -> output heads

The current predictor.py already does spatial conv fusion of temporal features with the last frame projection. This pattern is correct for our use case.

### Phase 2: Residual Entropy Coding (Week 1)

The residual r_t = z_t - predict(z_{t-1}) has lower entropy than z_t because the predictor removes temporal redundancy.

- Modify training to compute residual between predicted latent and actual latent
- Feed residual to existing HyperpriorEntropy for rate estimation
- At inference: transmit residual via torchac. Decoder adds residual to prediction.
- The entropy model already handles this. No architectural changes needed.

### Phase 3: Temporal Training Loop (Week 2)

Add a Phase 3 to `train_feature_compress.py`:

```
Phase 3 (JEPA temporal):
  Steps: 5000
  LR: 1e-4
  Trainable: predictor
  Frozen: compressor, decompressor, entropy_model, backbone, quantizer
  Loss: nll(predictor) + lambda * rate(residual)
```

Loss function for each training step:

1. Load 2 consecutive frames (t, t+1)
2. Extract features: f_t, f_{t+1} = backbone(t), backbone(t+1)
3. Compress: z_t, z_{t+1} = compressor(f_t), compressor(f_{t+1})
4. Quantize: q_t = quantizer(z_t), q_{t+1} = quantizer(z_{t+1})
5. Predict: pred = predictor([q_t])
6. Compute residual: r = q_{t+1} - pred
7. Rate: estimate bits for r via entropy_model
8. Distortion: recon = decompressor(pred + r), mse vs f_{t+1}
9. Loss: predictor_nll(q_t, q_{t+1}) + lambda * rate(r) + beta * mse

The dataset already supports this: `FrameDataset(sequence_length=2)` returns contiguous pairs. The existing VIRAT scene directories have sequential frame numbering.

### Phase 4: I-frame Refresh Strategy (Week 2)

Determine optimal GOP size:

- GOP=8: (0.173 + 7 * 0.05) / 8 = 0.065 BPP avg
- GOP=16: (0.173 + 15 * 0.05) / 16 = 0.058 BPP avg
- GOP=32: (0.173 + 31 * 0.05) / 32 = 0.054 BPP avg

Tradeoff: longer GOP = lower avg BPP but prediction drift accumulates. Test on 100 VIRAT scenes to find the knee.

Drift mitigation:
- Periodically insert I-frame (full intra) to reset context
- Track prediction error; force I-frame if residual exceeds threshold
- The existing NAL types already support this: `BL_I` and `BL_P`

### Phase 5: Bitstream Format Update (Week 2)

Update `src/lewm_vc/bitstream/writer.py` for temporal:

Frame header:
```
I-frame: [NAL_BL_I] [latent_data] [mu, sigma]
P-frame: [NAL_BL_P] [frame_idx] [ref_frame_idx] [residual_data] [mu, sigma]
```

GOP structure:
```
[I] [P] [P] [P] [P] [P] [P] [P]  (GOP=8)
 0   1   2   3   4   5   6   7
     ↓   ↓   ↓   ↓   ↓   ↓   ↓
   pred  pred pred pred pred pred pred
   from  from from from from from from
    0     1    2    3    4    5    6
```

Each P-frame references the previous reconstructed frame (not the original). This prevents drift during decoding.

### Phase 6: Evaluation (Week 3)

| Metric | Target | Method |
|--------|--------|--------|
| P-frame BPP | ~0.05 | Real torchac on residuals |
| GOP-16 avg BPP | ~0.058 | Measured across 100 VIRAT clips |
| Feature PSNR drift | < 0.5 dB over GOP | Compare last P-frame to I-frame baseline |
| Cosine drift | < 0.02 over GOP | Compare last P-frame to I-frame baseline |
| BD-rate vs all-intra | -60% | Same detection accuracy at lower bitrate |

## Architecture Diagram

```
I-frame:
  frame_t → backbone → compressor → quant → [transmit] → dequant → decomp → recon_t
                                                                          ↓
                                                                    context for predictor

P-frame:
  frame_{t+1} → backbone → compressor → quant → z_{t+1}
                                                     ↓
  context[recon_t] → predictor → pred_{t+1} → [-]
                                              ↓
                                          residual → entropy → [transmit] → + pred → decomp → recon_{t+1}

```

## Files to Modify

| File | Change |
|------|--------|
| `src/lewm_vc/predictor.py` | No changes. Create new file instead. |
| `src/lewm_vc/predictor_feature.py` | New. Adapted predictor for 8ch latent. 4 layer transformer, 128 hidden. |
| `scripts/train_feature_compress.py` | Add Phase 3 (JEPA temporal). Add temporal dataloader. Add residual loss. |
| `configs/train_feature_compress.yaml` | Add Phase 3 config block. |
| `src/lewm_vc/bitstream/writer.py` | Already has NAL types. No changes needed. |
| `scripts/encode_feature.py` | Add temporal encoding loop (GOP structure). |
| `scripts/decode_feature.py` | Add temporal decoding loop (predict + add residual). |
| `scripts/eval_feature_compress.py` | Add temporal metric tracking (drift, GOP avg BPP). |

## Training Data

VIRAT dataset already has sequential frames organized by scene:

```
datasets/virat/frames/VIRAT_S_000000/
  frame_0001.png
  frame_0002.png
  ...
  frame_0200.png

datasets/virat/frames/VIRAT_S_000001/
  frame_0001.png
  frame_0002.png
  ...
```

The existing `FrameDataset` with `sequence_length=2` loads contiguous pairs from these directories. Each scene is a clip. The dataset already handles this.

Risks:
- Some scenes may have low motion (static surveillance). Temporal prediction will be near perfect here, giving artificially low BPP estimates. Need to prioritize scenes with motion for evaluation.
- The predictor may overfit to VIRAT's specific camera angles. Cross dataset eval (PEViD HD if available) would validate generalization.
