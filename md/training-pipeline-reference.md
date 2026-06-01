# LeWM-VC Training Pipeline Reference

## Architecture

```
Input Frame (YUV420 → RGB 256×256)
       ↓
┌──────────────┐
│  LeWMEncoder  │  ViT-Tiny, 6 layers, 3 heads, 192-dim latents
│  (15M params) │  Output: 192ch × 16×16 grid + surprise score
└──────┬───────┘
       ↓ latent
┌──────────────┐
│  Quantizer    │  STE uniform quantization, 256 levels
└──────┬───────┘
       ↓ quantized latent
┌──────────────────┐     ┌──────────────────┐
│  LeWMPredictor    │◄────│  Decoded context  │
│  8-layer ViT      │     │  (4 past frames)  │
│  Output: pred_mean│     └──────────────────┘
└──────┬───────────┘
       ↓ residual = latent - pred_mean
┌──────────────┐
│  Quantizer    │  quantize residual (P-frames)
└──────┬───────┘
       ↓
┌──────────────────┐
│ HyperpriorEntropy │  GMM (2 components) + hyperprior CNN
│  Output: bits/frame│  Hyperprior: 256ch, 4 conv layers + GELU
└──────────────────┘
       ↓
┌──────────────┐
│ LeWMDecoder   │  hidden_dim=512, upsampling conv stack
│  Output: RGB  │  3× up-projection blocks
└──────────────┘
```

## Phase Schedule

| Phase | Name | Steps | λ | γ | rate_weight | δ | Frozen | Trains |
|-------|------|-------|---|---|-------------|---|--------|--------|
| 0 | JEPA Warmup | 30k | 0.0 | 2.0 | 0.0 | 0.001 | entropy_model, rate_controller, quantizer | encoder, predictor, decoder |
| 1 | Entropy Warmup | 10k | 60 | 3e-5 | 1.0 | 0.0005 | encoder | predictor, decoder, entropy_model |
| 2 | QAT | 10k | 60 | 3e-5 | 1.0 | 0.0005 | encoder, predictor | decoder, entropy_model |
| 3 | Decoder Refine | 5k | 60 | 1e-5 | 1.0 | 0.0001 | encoder, predictor, entropy_model | decoder |
| 4 | Cooldown | 5k | 60 | 1e-5 | 1.0 | 0.0001 | encoder, predictor | decoder, entropy_model |

**Loss function:**
```
total_loss = rate_weight × R + λ × D + γ × J + δ × SIGReg + surprise_loss
  R  = -log₂(p(latent | hyperprior))   [bitrate estimate]
  D  = mse_weight × MSE + lpips_weight × LPIPS
  J  = ||zₜ - predictor(zₜ₋₁, ..., zₜ₋ₙ)||²   [JEMA future prediction]
```

## Key Files

| File | Purpose |
|------|---------|
| `configs/train_config.yaml` | Phase schedule, model arch, data, hyperparams |
| `src/scripts/train.py` | Training loop, loss computation, checkpointing |
| `scripts/train_production.py` | Production wrapper — run IDs, W&B, best-ckpt tracking |
| `scripts/eval_monitor.py` | Background eval watcher — polls checkpoints, logs PSNR/BPP |
| `scripts/evaluate.py` | Full evaluation with HTML report |
| `scripts/inference.py` | Encode/decode a sequence |
| `src/lewm_vc/entropy.py` | Hyperprior GMM entropy model |
| `src/lewm_vc/decoder.py` | Decoder network (hidden_dim=512) |

## Fixes Applied

### 1. λ restore bug (`src/scripts/train.py:404`)
- **Symptom:** CLI `--lambda 60` was silently overwritten by checkpoint's `lambda_val=0.05`
- **Fix:** Removed `self.lambda_val = ckpt.get("lambda_val", self.lambda_val)`
- **Verification:** CLI lambda is now the single source of truth

### 2. rate_weight bug (`src/scripts/train.py:246`, `configs/train_config.yaml:94`)
- **Symptom:** Phase 0 with `lambda: 0.0` only killed λ×D term. R (rate_loss) had no multiplier and always contributed, dominating the encoder gradient with noise from the frozen random entropy model (908k vs JEPA 396)
- **Fix:** Added `rate_weight` multiplier with default 1.0. Phase 0 sets `rate_weight: 0.0`
- **Verification:** Phase 0 loss dropped from 909,119 → ~16 (pure JEPA + SIGReg)

### 3. Entropy model mu collapse (`src/lewm_vc/entropy.py`)
- **Symptom:** Mu stayed at 0 because zero-init of final conv layer zeroed both mu and sigma channels → d(KL)/d(mu) = mu/log(2) = 0 → no gradient
- **Fix:** Sigma-only zero init (keep mu channel kaiming-uniform). Replace ReLU with GELU. Re-randomize mu channels on checkpoint load
- **Verification:** Mu values diverge from 0 during training

### 4. Architecture mismatches
- **Symptom:** Code had `decoder.hidden_dim=128` but checkpoints trained with 512; entropy `hyper_channels=320` but checkpoint was 256
- **Fix:** Set defaults to `hidden_dim=512, hyper_channels=256, num_components=2`

### 5. evaluate.py temporal context bug
- **Symptom:** `rglob("*.png")` across all clips interleaved frames, breaking predictor context
- **Fix:** Use only the first clip's frames for temporal evaluation

## Commands

### Launch training from scratch
```bash
# Phase 0 (will auto-transition through all phases)
docker exec -w /workspace/le-maia rocm \
  python3 scripts/train_production.py \
    --config configs/train_config.yaml \
    --phase 0 --lambda 60
```

### Resume from checkpoint with phase override
```bash
# Skip completed phases, resume from best Phase 0 checkpoint
docker exec -w /workspace/le-maia rocm \
  python3 scripts/train_production.py \
    --config configs/train_config.yaml \
    --phase 1 --lambda 60 \
    --resume checkpoints/sentinel-p0-l60.0-XXXX/lambda_60.0/best.pt
```

### Launch eval monitor (separate tmux session)
```bash
docker exec -w /workspace/le-maia rocm \
  python3 scripts/eval_monitor.py \
    --run-dir checkpoints/sentinel-p1-l60.0-XXXX \
    --data datasets/virat/frames \
    --interval 180 --num-frames 32
```

### Full evaluation
```bash
python3 scripts/evaluate.py \
  --checkpoint checkpoints/sentinel-p1-l60.0-XXXX/lambda_60.0/step_5000.pt \
  --data datasets/virat/frames/VIRAT_S_000000 \
  --temporal --report --output eval_results/run_name
```

### Sync code to droplet
```bash
bash scripts/sync_to_droplet.sh <droplet-ip>
# Or with auto-commit + push:
bash scripts/sync_to_droplet.sh --push "commit message" <droplet-ip>
```

## tmux Sessions

| Session | Purpose | Attach Command |
|---------|---------|----------------|
| `train` | Training loop | `ssh root@IP -t "tmux attach -t train"` |
| `monitor` | Eval watcher | `ssh root@IP -t "tmux attach -t monitor"` |

Detach: `Ctrl+B, D`

## Monitoring

### GPU utilization
```bash
ssh root@<droplet-ip> "docker exec rocm rocm-smi"
```

### TensorBoard (local)
```bash
tensorboard --logdir runs/
```

### Eval log
Checkpoints evaluated: `checkpoints/<run-id>/eval_log.json`

### Training TB scalars
`runs/lambda_60.0/events.out.tfevents.*`
- `train/total_loss` — combined loss
- `train/rate_loss` — bitrate estimate
- `train/distortion_loss` — MSE + LPIPS weighted sum
- `train/mse_loss` — raw MSE
- `train/lpips_loss` — raw LPIPS
- `train/jepa_loss` — next-latent prediction error
- `train/sigreg_loss` — Cramér-Wold regularization
- `val/*` — same metrics on validation set

## Expected Timeline (Phase 0→4)

| Phase | Steps | Time (est.) | Eval PSNR |
|-------|-------|-------------|-----------|
| 0 (JEPA warmup) | 30k | ~6h | ~5.6 dB (decoder untrained) |
| 1 (Entropy warmup) | 10k | ~2h | ~5.6 dB (decoder still frozen) |
| 2 (QAT) | 10k | ~2h | 5→20+ dB (decoder learns distortion) |
| 3 (Decoder refine) | 5k | ~1h | 20→24+ dB (polish) |
| 4 (Cooldown) | 5k | ~1h | Fine-tune |

**Target:** PSNR ≥ 20 dB (surveillance machine perception, not human viewing), BPP 0.08–0.25, P/I ratio < 0.6

## Pilot Model Criteria

- PSNR ≥ 20 dB (sufficient for object detection/tracking)
- LPIPS ≤ 0.38
- BPP 0.08–0.25
- P-frame BPP < 60% of I-frame BPP
- Task loss (ResNet-18 feature distance) lower than x265 baseline

## Droplet Setup (AMD MI300X)

- **Host:** Ubuntu 24.04, DigitalOcean PyTorch 1-Click Droplet
- **Container:** `rocm`, working dir `/workspace/le-maia/`
- **GPU:** 1× MI300X (192GB HBM3), Docker exposed via ROCm
- **Host code:** `/root/le-maia/` — synced to container via `docker cp`
- **SSH:** Intermittent connectivity — retry with `sleep 15 && ssh ...`

## Current Run

| Run ID | Phase | λ | Status |
|--------|-------|---|--------|
| `sentinel-p1-l60.0-809b54` | 1 | 60 | Active (entropy warmup) |

Started from best Phase 0 checkpoint `sentinel-p0-l60.0-13aacf/lambda_60.0/best.pt`.
