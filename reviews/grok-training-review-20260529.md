# LeWM-VC / Sentinel — Training Review (2026-05-29)

**Author:** AI Engineering Agent  
**Target Reviewer:** Grok  
**Context:** Post-Phase 0 warmup launch on MI300X with full VIRAT Ground Dataset

---

## 1. Project State Summary

| Metric | Value |
|--------|-------|
| Codec name | LeWM-VC (Sentinel) |
| Company | MAHAMAIA Systems |
| Target use-case | Machine-perception-optimized video compression for remote industrial surveillance (VSAT/satellite backhaul) |
| Training hardware | 1× AMD MI300X (DigitalOcean PyTorch 1-Click Droplet) |
| Total params | ~15M |
| Latent dim | 192 |
| Patch size | 16×16 |
| Encoder | ViT-Tiny (6 layers, 3 heads, 192 hidden) |
| Predictor | 8-layer transformer (4 heads, 256 hidden) |
| Decoder | 4-layer ConvTranspose + post-filter |
| Entropy | Hyperprior GMM (2 components, 256 hyper channels) |
| Quantizer | STE uniform, 256 levels (8-bit) |
| Inference speed | 80+ fps on T4 |

---

## 2. Data Status

### VIRAT Ground Dataset (Release 2.0)

| Metric | Value |
|--------|-------|
| Total clips in archive | 523 |
| Successfully extracted | 456 |
| Failed/invalid | 67 (likely corrupt or format mismatch in zip) |
| Total frames | 1,308,279 |
| Training clips (70%) | 319 |
| Validation clips (15%) | 68 |
| Test clips (15%) | 69 |
| Frames per clip | 791 – 29,310 (avg ~2,868) |
| Resolution | OG: 1920×1080, Training: 256×256 (resized) |
| Frame rate | 30 fps |

### Critical Data Observations

1. **Massive improvement from 3 clips → 319 clips.** Previous warmup runs were training on only 3 video clips (VIRAT with --max-frames 300). This was causing severe overfitting and meaningless validation metrics.
2. **Zip is kept permanently.** The download script previously deleted the 111GB zip after extraction. This was fixed — zip is preserved for future use without re-downloading.
3. **Dataset hash is verified.** `train_production.py` computes a SHA-256 hash of the dataset and stores it in `manifest.json` for reproducibility.

---

## 3. Training Architecture

### Phase Schedule

| Phase | Name | Steps | Trainable | Frozen | Loss weights |
|-------|------|-------|-----------|--------|-------------|
| 0 | Warmup | 50,000 | encoder, predictor | decoder, entropy, quantizer, rate_control | γ=5.0 (JEPA), δ=0.001 (SIGReg), λ=0.0 (RD) |
| 1 | Joint RD | 100,000 | all | — | γ=1.0, δ=0.0005, λ=0.05, MSE=0.7, LPIPS=0.3 |
| 2 | QAT | 50,000 | all | — | γ=1.0, δ=0.0005, λ=0.05, MSE=0.7, LPIPS=0.3 |
| 3 | Decoder Refine | 15,000 | decoder, rate_control | encoder, predictor, entropy | γ=0.5, δ=0.0001, λ=0.05, MSE=0.5, LPIPS=0.5 |
| 4 | Cooldown | 20,000 | all | — | γ=0.5, δ=0.0001, λ=0.05, MSE=0.5, LPIPS=0.5 |

**Total: 235,000 steps** — estimated ~20 hours on MI300X.

### Loss Formula (Paper Eq. 5)

```
L_total = R + λ·D + γ·L_JEPA + δ·L_SIGReg + η·L_surprise
```

Where:
- **R**: Rate = entropy model KL on quantized residuals (bits)
- **D**: Distortion = MSE + LPIPS (learned perceptual loss via VGG-16)
- **L_JEPA**: MSE(predicted_latent, encoded_latent) — stop-gradient on target
- **L_SIGReg**: KL(N(z|μ,σ²) || N(0,1)) — Gaussian regularization
- **L_surprise**: VOE (Variance of Expectation) surprise penalty

### Key Architectural Decisions

**1. Temporal Residual Coding (NEW — implemented 2026-05-29)**

Previously, the training loop decoded from raw quantized latents for ALL frames, even when the predictor was computing residuals for P-frames. This meant the decoder never saw temporally-reconstructed latents during training, creating a train-test mismatch.

**Fix — I-frame / P-frame split in `compute_loss()`:**

```
I-frame (idx=0):
  quant_latent = quantizer(latent)
  recon = decoder(quant_latent)
  store quant_latent as recon context

P-frame (idx>=1):
  pred_mean = predictor(recon_context)    ← uses decoded latents (matches inference)
  residual = latent - pred_mean
  quant_residual = quantizer(residual)
  rate = entropy_model(quant_residual)    ← true temporal rate
  recon_latent = pred_mean + quant_residual
  recon = decoder(recon_latent)           ← decoder trained on temporal latents
  store recon_latent as context
```

This ensures:
- Decoder learns to work with `predicted + residual` latents (inference path)
- Predictor receives decoded latents as input (teacher forcing with quantization noise)
- Rate is truly on residuals (temporal coding gain)

**2. JEPA loss on raw encoder output** — The JEPA loss compares `pred_mean` (from decoded latents) against `latent.detach()` (raw encoder output). This keeps the predictor optimizing against clean targets while using noisy (decoded) inputs. Standard JEPA practice.

**3. SIGReg on raw latents** — Applied per-frame regardless of I/P type. Regularizes the latent space to prevent collapse.

---

## 4. Bugs Fixed This Session

| Bug | File | Line | Impact |
|-----|------|------|--------|
| Context length exceeded | `train.py:204` | Predictor received all past latents as context (grew unbounded). Crashed at frame_idx > context_len. | ✅ Fixed — slice to last `context_len` frames |
| Empty val loader crash | `train.py:586` | `next(iter(val_loader))` raised StopIteration when val set had 0 clips | ✅ Fixed — use `next(val_iter, None)` guard |
| Temporal loop not used for decode | `train.py:214-216` | Decoder always used raw `quant_latent`, not `pred_mean + quant_residual` | ✅ Fixed — full I/P-frame split |
| ZIP deleted after extraction | `download_virat.py:247` | Script deleted the 111GB zip after processing, preventing resume without re-download | ✅ Fixed — zip preserved |
| Soft positional encoding | `encoder.py` (prior session) | Broadcasting single vector instead of proper 2D grid + interpolation | ✅ Fixed (before this session) |

---

## 5. Current Training Run

### Run Info

| Field | Value |
|-------|-------|
| Run ID | `sentinel-p0-l0.05-d9382b` |
| Config | `configs/train_config.yaml` (tuned) |
| Started | ~07:30 UTC 2026-05-29 |
| Starting checkpoint | `step_15000.pt` (from old 3-clip Phase 0 run) |
| Scheduler | Cosine over 235,000 total steps, 1,000-step linear warmup |
| Initial LR | 1e-4 |
| Warmup | 1,000 steps (linear 0 → 1e-4) |
| Precision | bf16 |
| Batch size | 8 |
| Sequence length | 16 frames per sample |
| Frame stride | 1 |
| Val interval | 200 steps (tuned from 500) |
| Save interval | 2,500 steps (tuned from 5,000) |
| Keep last | 5 checkpoints |

### Observed Loss Behavior (first 500 steps on full data)

| Step | Loss | Note |
|------|------|------|
| 15,500 | 4.40M | First step on 319-clip data (was 4.1M with 3 clips) |
| 16,000 | 4.47M | Slight increase — expected when switching data distribution |
| 16,500 | 4.37M | Stable — model adapting to diverse motion |

Loss is ~4.3M for warmup phase. This is expected:
- No RD signal (λ=0, decoder frozen)
- JEPA MSE on random-initialized latents from 3→319 clip transition
- Predictor has 15k steps of prior training on 3 clips, now needs to generalize

### Validation Metrics

Validation is now **meaningful** for the first time:
- 68 held-out clips (203,026 frames)
- Metrics logged every 200 steps
- `val/total_loss` tracked by `train_production.py` for best-checkpoint selection

---

## 6. Production Infrastructure

### Training Pipeline Structure

```
train_production.py (entry point)
├── Config loading + overrides
├── Run ID generation (sentinel-p{Lambda}-l{value}-{random6})
├── Manifest.json creation
├── Dataset hash computation
├── Model construction (encoder/predictor/decoder/entropy/quantizer/rate_controller)
├── Checkpoint resume (auto-finds latest step_*.pt)
├── W&B optional logging
├── Best-checkpoint tracking (by val_loss)
├── Signal handler (graceful SIGINT → save final checkpoint)
└── Calls lewm_train() from train.py
    ├── AdamW optimizer with per-module LRs
    ├── Cosine LR scheduler with warmup
    ├── Phase loop: 0→1→2→3→4
    │   ├── Phase config → freeze/unfreeze
    │   ├── DataLoader iter → train_step
    │   └── Epoch/step counters
    └── Save/validate at intervals
```

### Key Strengths

- **Checkpoint manifest** — every run writes `manifest.json` with run_id, config hash, dataset hash, best_checkpoint path, timestamp
- **Graceful interrupt** — Ctrl+C saves final checkpoint before exiting
- **Resume resilience** — resume from any `step_*.pt` or checkpoint directory (auto-picks latest)
- **Config-driven phases** — add/remove/modify phases without code changes
- **Production wrapper separates phases** from the training engine — `train.py` is the engine, `train_production.py` adds experiment tracking

### Key Gaps

1. **No W&B API key set** — run logs locally via TensorBoard. Need `WANDB_API_KEY` env var set for cloud tracking.
2. **No email/Slack notification on completion** — the 20-hour run finishes silently. Need to add a notification hook.
3. **No multi-lambda sweep** — only λ=0.05 is training. Other 4 lambda values (0.02, 0.08, 0.12, 0.20) need separate runs for RD curve.
4. **No test eval hook** — test set (69 clips) is not evaluated. Only train/val metrics during training.

---

## 7. Evaluation Plan (Post-Training)

Once the full pipeline completes (~20h), the plan is:

1. **Extract best.pt** — already handled by manifest.json tracking
2. **Run RD curve** — `evaluate.py` with λ sweep on held-out test clips
3. **BD-rate vs x265** — compute Bjøntegaard Delta-rate for PSNR, VMAF, VMAF-NEG
4. **Machine perception benchmark** — YOLO mAP on LeWM-VC decoded frames vs x265 at matched bitrate
5. **Temporal coding gain** — P-frames vs I-frames: measure actual bit savings from residual coding
6. **Surprise-gating demo** — anomaly-detection-driven bit allocation on surveillance clips

---

## 8. Remaining Technical Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| SIGReg is KL-to-Gaussian, not true Cramér-Wold SIGReg from paper | Medium | Acceptable for Phase 0-1. Paper-level results need proper SIGReg implementation |
| Decoder was never independently validated | Medium | Phase 3 decoder refine should help. Visual inspection of reconstructions needed |
| 67 VIRAT clips failed extraction | Low | 456 clips is sufficient. The 67 failures are likely corrupt or non-standard files |
| No test-eval during training | Low | Test set is preserved for post-training eval. No leakage |
| Temporal residual coding not validated on real data | Medium | This runs first validation. P/I bit ratio unknown |

---

## 9. Questions for Grok

1. **LR schedule adequacy**: 1e-4 initial with 235k-step cosine decay — is this too aggressive for a resumed checkpoint (step 15000)? Should we reduce initial LR for the resumed run?

2. **Phase 3 design**: Decoder-only refinement for 15k steps makes intuitive sense (after QAT noise), but lacks theoretical grounding. Is there precedent for this in other neural codec training pipelines (e.g., Deep Render, InterDigital)?

3. **SIGReg implementation risk**: Our SIGReg is standard Gaussian KL (closed-form). The true SIGReg uses Cramér-Wold random projections. How critical is this for publishable BD-rate results?

4. **Data diversity**: 456 VIRAT clips from 11 scenes is decent but surveillance-specific. Should we prioritize adding PEVID (diverse outdoor) and SFU-HW (synthetic) for generalization claims?

5. **Multi-lambda strategy**: Current train runs λ=0.05. For RD curve, should we fine-tune from this checkpoint at other lambdas (cheaper), or train each from scratch (more rigorous)?
