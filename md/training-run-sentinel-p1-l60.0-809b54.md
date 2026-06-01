# Training Run: `sentinel-p1-l60.0-809b54`

**Date:** May 31 - Jun 1, 2026
**Hardware:** AMD MI300X (192GB HBM3) — DigitalOcean PyTorch Droplet
**Dataset:** VIRAT surveillance clips (319 train / 68 val)
**Config:** `configs/train_config.yaml` — 4-phase, λ=60

---

## Schedule

| Phase | Name | Steps | Active | Frozen | Start Step |
|-------|------|-------|--------|--------|------------|
| 0 | Warmup (JEPA) | 5000 | encoder, predictor, decoder | entropy_model, rate_controller, quantizer | 0 (from P0 checkpoint) |
| 1 | Joint RD | 10000 | predictor, decoder, entropy_model, quantizer, rate_controller | encoder | 5000 |
| 2 | QAT | 10000 | decoder, entropy_model, quantizer, rate_controller | encoder, predictor | 15000 |
| 3 | Distillation | 5000 | decoder, quantizer, rate_controller | encoder, predictor, entropy_model | 25000 |
| 4 | Cooldown | 5000 | decoder, entropy_model, quantizer, rate_controller | encoder, predictor | 30000 |

**Total planned: 35,000 steps.** Started from Phase 0 checkpoint `sentinel-p0-l60.0-13aacf` (resumed at step 5000).

---

## Checkpoint Progression

| Step | Phase | Loss | Best? | Notes |
|------|-------|------|-------|-------|
| 5000 | 0→1 transition | — | — | Phase transition: warmup → joint_rd |
| 10000 | 1 | 65.80 | ✅ (64.02 val) | First Phase 1 best |
| 15000 | 1→2 transition | 41.81 | ✅ (37.69 val) | Phase transition: joint_rd → qat |
| 20000 | 2 | 9.47 | ✅ (8.95 val) | **Best overall checkpoint** — lowest val_loss |
| 25000 | 2→3 transition | 9.80 | — | Phase transition: qat → distillation |
| 30000 | 3→4 transition | 15.65 | — | Phase transition: distillation → cooldown |

**Best overall: step 20000** (val_loss=8.95). Loss in Phase 3+4 was higher (decoder-only polish on frozen features).

---

## Model Architecture

- **Encoder:** ViT-Tiny (latent_dim=192, patch_size=16, 6 layers, 3 heads)
- **Predictor:** 8-layer transformer (hidden_dim=256, 4 heads, context_len=4)
- **Decoder:** CNN (hidden_dim=512)
- **Entropy:** Hyperprior CNN (hyper_channels=256, 2 components)
- **Quantizer:** 256 levels, step_size=0.007812

---

## Evaluation Results

### Metrics at Best Checkpoint (step 20000)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| PSNR | 21.02 dB | ≥ 20 dB | ✅ |
| LPIPS | 0.3436 | ≤ 0.38 | ✅ |
| Task Loss (ResNet18) | 0.002735 | low | ✅ |

### Real BPP (torchac arithmetic coding)

| Measurement | Value | Target | Status |
|-------------|-------|--------|--------|
| KL BPP (reported during training) | ~0.0 | — | ⚠️ Collapsed |
| Real BPP (torchac, pre-fix) | 6.0 | 0.08–0.25 | ❌ |
| Real BPP (torchac, post-NLL fix) | 4.10 | 0.08–0.25 | ❌ |
| zlib BPP | 5.45 | — | ❌ |

**Key finding: Entropy model collapsed.** KL loss minimized at μ=0, σ=1 for all inputs, giving ~0 BPP estimate. Real BPP with arithmetic coding at those parameters is ~6.0. The encoder was never constrained by a real rate gradient.

---

## NLL Entropy Fix

**Script:** `scripts/train_entropy_fast.py` — pre-computed latents, CPU training

- Pre-encoded 2 VIRAT clips (~1000 frames) → cached latents on CPU
- Trained entropy model only (encoder/decoder frozen) with NLL loss
- 500 steps, val_nll: 6.23
- **Result:** Real BPP dropped 6.0 → **4.10** (32% improvement)
- KL BPP now reads **2.30** (meaningful, not collapsed)

### Why BPP is still too high

The KL-based loss used during training gave the encoder zero incentive to produce compressible latents. The encoder learned to fill all 192 channels with high-entropy information. The NLL fix only re-trains the entropy model to *predict* better — it doesn't change the encoder.

**To reach 0.08–0.25 BPP, the encoder must be retrained with NLL rate loss.**

---

## Demo Results

### Intra-only (4 frames, VIRAT_S_000000)

```
PSNR: 21.23 dB
I BPP: 4.90 (KL-based, not real)
```

### Temporal IPPP (16 frames, VIRAT_S_000000)

```
PSNR: 21.04 dB
I BPP: 4.94 → P BPP: 2.22 (P/I ratio: 0.449)
```

Motion compensation working: P-frames use 55% fewer bits than I-frames.

### Temporal IPPP — Phase 4 (step 35000)

```
PSNR: 21.75 dB  (+0.71 vs step 20000)
I BPP: 0.00 (KL collapsed, cosmetic)
```

Decoder polish improved quality. Side-by-side comparison (triptych) shows smoother reconstructions with reduced block artifacts compared to step 20000, though fine details (pedestrians) remain lost.

---

## Files / Checkpoints

```
checkpoints/sentinel-p1-l60.0-809b54/lambda_60.0/
├── best.pt              (step 20000, 287MB, best val_loss)
├── sentinel_nll_fixed.pt (step 20000 + NLL entropy fix, 287MB)
├── step_10000.pt
├── step_15000.pt
├── step_20000.pt
├── step_25000.pt
└── step_30000.pt
```

---

## Lessons Learned

1. **KL-based rate loss collapses** — `KL(N(μ,σ²)||N(0,1))` is minimized at μ=0, σ=1 with vanishing gradients. Using this as rate estimate gives 0 BPP fiction.
2. **Real BPP must be measured** — KL BPP and real arithmetic coding BPP can differ by 6+ orders of magnitude.
3. **Rate weight matters** — Phase 0 had `rate_weight: 0.0` fix needed to prevent rate from dominating JEPA gradient even with λ=0.
4. **λ restore bug** — `train.py` was overwriting CLI `--lambda` with checkpoint value. Fixed.
5. **NLL fine-tuning works** — 500 steps of NLL on CPU dropped real BPP 6.0→4.1, but encoder unfreezing is needed for the remaining gap.

---

## NLL RD Fine-tuning Attempt (post-training)

**Script:** `scripts/train_rd_nll.py` — full data pipeline, encoder + entropy unfrozen, decoder/predictor frozen

### Hyperparameters

- **λ (rd_lambda):** 60
- **Steps:** 3000 (killed at ~1000 — plateaued)
- **LR:** 1e-5 (encoder), 1e-5 (entropy)
- **Batch:** 4 frames x 4 sequences
- **Loss:** `BPP + λ * (0.7*MSE + 0.3*LPIPS)`

### Results

| Step | BPP | PSNR | λ*Distortion | Notes |
|------|-----|------|-------------|-------|
| 50 | 10.98 | 19.04 | 6.29 | Initial spike (encoder adapting) |
| 100 | 6.37 | 19.00 | 6.27 | Rapid drop |
| 500 | 6.03 | 19.31 | 6.11 | First val checkpoint (BPP=5.83, PSNR=19.35) |
| 1000 | 5.88 | 18.34 | 6.93 | Plateau — stable ~5.8 BPP |

### Analysis

**BPP reduced from 6.0 → 5.8** — minimal improvement. The encoder was trained for 35k steps without rate constraint and fine-tuning at λ=60 (balanced) gives no incentive to compress further.

The loss components show `BPP ≈ 5.8` and `λ*D ≈ 6.0` are well-balanced — the optimizer is at a local minimum where BPP and quality are traded equally. To push BPP lower, λ must be reduced so rate dominates.

### Next Attempt

Subsequent run uses **λ=1** (rate dominates ~58:1) to force aggressive compression: `train_rd_nll.py --checkpoint step_35000_nll.pt --output checkpoints/rd_nll_l1 --steps 2000 --lambda 1`

Result: BPP stuck at ~5.4 with frozen decoder. Frozen decoder was the bottleneck.

---

## Perception Model (32 channels)

**Root cause of high BPP:** 192-channel latent is too large for 0.08-0.25 BPP. At 192×16×16 = 49,152 elements, even 1 bit/element = 0.75 BPP.

**Fix:** latent_dim=32, keep patch_size=16 (existing decoder works). Gives 32×16×16 = 8,192 elements. At ~1.3 bits/element → ~0.16 BPP (on target).

### Config: `configs/train_perception.yaml`

```yaml
model:
  latent_dim: 32
  patch_size: 16
  decoder:
    hidden_dim: 256
  entropy:
    hyper_channels: 64
```

### Schedule

| Phase | Name | Steps | Active | MSE/LPIPS | Rate | Task |
|-------|------|-------|--------|-----------|------|------|
| 0 | Autoencoder warmup | 3000 | encoder, decoder | 0.7/0.3 | off | off |
| 1 | RD NLL | 15000 | encoder, decoder, entropy | 0.7/0.3 | λ=0.05 | off |
| 2 | Task optimize | 5000 | encoder, decoder, entropy | 0.3/0.3 | λ=0.01 | w=1.0 |

### Progress (Phase 1, λ=1.0, step ~500 of 15000)

| Metric | Value | Target |
|--------|-------|--------|
| PSNR | ~19.5 dB | ≥ 20 dB |
| BPP (NLL) | 0.64 | 0.08-0.25 |
| Task loss | N/A (Phase 2) | low |

With λ=1.0, rate contributes ~64% of gradient (vs ~22% with λ=0.05). BPP trajectory is similar to λ=0.05 run (1.0 → 0.64 in 500 steps) due to the decoder+encoder both needing time to co-adapt to the rate constraint. Long-term floor still unknown — need 5000+ steps to judge.

**Note:** λ=0.05 run (Round 1) plateaued at ~0.6 BPP because rate was only 22% of gradient. Round 2 with λ=1.0 gives rate 64% of gradient — expected to converge lower, possibly at 0.3-0.5 BPP. Further λ increases (e.g. λ=5) may be needed to reach 0.08-0.25 if this run plateaus.

### 32-channel vs target BPP math

| latent_dim | Elements | Target BPP | bits/element | Feasible? |
|-----------|----------|------------|-------------|-----------|
| 192 | 49,152 | 0.2 | 0.27 | ❌ too few bits/element |
| 64 | 16,384 | 0.2 | 0.80 | maybe |
| 32 | 8,192 | 0.2 | **1.60** | ✅ most headroom |
| 16 | 4,096 | 0.2 | **3.20** | ✅ but encoder capacity too low |

Lower latent_dim gives more bits per element at same BPP target — 32 channels is the sweet spot for the 0.08-0.25 range.

### Key insight from both runs

With frozen decoder (first attempt at λ=1), BPP was stuck at ~5.4. With unfrozen decoder (current run), BPP dropped to 0.64. The decoder + encoder must co-adapt — you can't freeze one when changing the other's compression behavior.

---

## Shutdown Checklist (when training completes)

### Download from remote before shutdown:

**Essential:**
```
checkpoints/perception/best.pt                         → ~300 MB
checkpoints/sentinel-p1-l60.0-809b54/lambda_60.0/       → reference (optional)
├── sentinel_nll_fixed.pt
├── best.pt
└── step_35000.pt
```

**BPP & results:**
```
checkpoints/rd_nll_l*/                                   → NLL RD attempts
bpp_results_nll/                                         → BPP measurements
perception_bpp_results/                                  → perception model BPP
```

**Demo outputs:**
```
demo_output/                                             → intra-only demo
demo_temporal/                                           → temporal IPPP demo
demo_final/                                              → Phase 4 demo
demo_perception/                                         → perception model demo
```

**Logs:**
```
runs_perception/                                         → TensorBoard logs
```

### Already on local machine (~/Downloads/):
- `sentinel_nll_fixed.pt`
- `demo_output.tar.gz`
- `demo_temporal.tar.gz`
- `demo_final.tar.gz`
- `comparison.png`
- `comparison_triptych.png`

### When ready to shutdown:
```bash
# Package everything
docker exec -w /workspace/le-maia rocm tar czf /tmp/lemaia_final.tar.gz \
  checkpoints/perception/best.pt \
  checkpoints/sentinel-p1-l60.0-809b54/lambda_60.0/sentinel_nll_fixed.pt \
  checkpoints/sentinel-p1-l60.0-809b54/lambda_60.0/best.pt \
  checkpoints/rd_nll_l* \
  bpp_results_nll \
  perception_bpp_results \
  demo_perception \
  runs_perception

docker cp rocm:/tmp/lemaia_final.tar.gz /root/
# Then SCP from local: scp root@129.212.177.242:/root/lemaia_final.tar.gz ~/Downloads/
```
