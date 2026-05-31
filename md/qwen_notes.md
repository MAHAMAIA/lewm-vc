Here is a comprehensive, production-ready specification for your AI coder. It translates the assessment into exact technical requirements, code modifications, and training protocols needed to achieve competitive RD performance and functional semantic surprise detection.

---
## 🎯 Objective
Transform the current pipeline into a **benchmark-competitive learned video codec** that:
1. Matches or beats x265 CRF 28 (~0.07 bpp @ 25.2 dB PSNR on Y channel at 256×256)
2. Produces a **strictly monotonic RD curve** across λ sweeps
3. Demonstrates **correct VoE (Video Outlier/Anomaly) detection** (anomaly surprise > normal surprise)
4. Passes standard evaluation protocols (UVG/HEVC test sets, Y-PSNR, proper BD-rate computation)

---
## 🔧 Critical Code Modifications

### 1. `train_rd_from_scratch.py` (High Priority)
| Issue | Fix | Implementation Notes |
|-------|-----|----------------------|
| **128×128 resolution** | Change to `(256,256)` | Update `VideoDataset(frame_size=(256,256))`, increase batch size to `4` (fits in 24GB VRAM), adjust `target_size` in decoder |
| **50 epochs per λ** | Increase to `150` | Add `EPOCHS_PER_LAMBDA = 150` |
| **Non-monotonic RD** | Fix loss scaling & LR schedule | Use `rate_per_pixel = nll / np.log(2)` (no batch-size division), add LR warmup: `LinearLR(optimizer, start_factor=0.1, total_iters=5)` → `CosineAnnealingLR` |
| **Gradient instability** | Proper mixed precision + unscale | `scaler.unscale_(optimizer)` **before** `clip_grad_norm_`, then `scaler.step()` |
| **Entropy model capacity** | Increase `hyper_channels` to `640` | `ContextualEntropyModel(latent_dim=192, hyper_channels=640, context_hidden=256)` |
| **Checkpointing** | Save best val model, not just final | Track `best_val_loss`, save `ae_lambda_{lam}_best.pt` + `entropy_lambda_{lam}_best.pt` |

**Key Code Patch (Training Loop):**
```python
# Replace scaler/clip sequence with:
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(
    list(autoencoder.parameters()) + list(entropy_model.parameters()), 
    max_norm=1.0
)
scaler.step(optimizer)
scaler.update()
```

### 2. `eval_rd_complete.py` (High Priority)
| Issue | Fix | Implementation Notes |
|-------|-----|----------------------|
| **RGB PSNR** | Switch to **Y-channel PSNR** (ITU-R BT.601) | Add `rgb_to_yuv` conversion, compute MSE only on Y channel |
| **Inconsistent frame sampling** | Use fixed 150-frame window for all methods | `frames = frames[:150]` after loading |
| **BPP miscalculation** | Correct formula: `total_bits / (H × W × num_frames)` | Remove division by batch/pixel in loop, compute once at end |
| **Missing BD-rate** | Add cubic spline BD-rate computation | Use `scipy.interpolate.CubicSpline` on `log2(bpp)` vs `PSNR` |
| **x265 benchmark mismatch** | Encode same 150 frames, decode, align lengths | Ensure `min_len` alignment before PSNR calculation |

**Y-PSNR Conversion Snippet:**
```python
def rgb_to_yuv_torch(rgb):
    # rgb: [B,3,H,W] in [0,1]
    r, g, b = rgb[:,0], rgb[:,1], rgb[:,2]
    y = 0.299*r + 0.587*g + 0.114*b
    return y

# In eval loop:
y_recon = rgb_to_yuv_torch(recon)
y_orig = rgb_to_yuv_torch(frame_t)
mse_y = torch.mean((y_recon - y_orig)**2)
psnr_y = 10 * torch.log10(1.0 / mse_y).item()
```

### 3. `create_all_demos.py` (Medium Priority)
- Update `target_size=(256,256)`
- Load `ae_lambda_{lam}_best.pt` instead of `_final.pt`
- Fix heatmap normalization: `bits_per_patch / bits_per_patch.max()` → apply `cv2.COLORMAP_JET` for clearer contrast
- Add FPS metadata to output videos

### 4. VoE Predictor (New/Modified Script)
The predictor must be **jointly trained** with the AE to learn temporal dynamics:
```python
# Add to AE forward():
def forward(self, x, prev_latent=None):
    # x: [B,T,C,H,W]
    latents = []
    for t in range(x.shape[1]):
        z = self.encoder(x[:,t])
        if prev_latent is not None:
            pred = self.predictor(prev_latent)  # simple Conv2D or GRU
            surprise = torch.mean((z - pred)**2, dim=[2,3])  # [B]
        else:
            surprise = torch.zeros(x.shape[0], device=x.device)
        latents.append(z)
        prev_latent = z.detach()
    return torch.stack(latents), surprise
```
**Training:** Add `surprise_loss = λ_s * torch.mean(surprise_anomaly - surprise_normal)` to force higher error on anomalous frames. Evaluate on held-out anomaly clips.

---
## 📊 Training & Evaluation Protocol

### Phase 1: Pretraining (Optional but Recommended)
- Train AE on reconstruction only (`λ=0.0`) for 30 epochs at 256×256
- Stabilizes latent distribution before RD pressure

### Phase 2: Rate-Distortion Fine-Tuning
- **λ sweep:** `[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]` (covers 0.05–1.5 bpp range)
- **Curriculum:** Start with `λ_target * 2.0`, decay linearly to `λ_target` over 50 epochs
- **LR:** `1e-4` (AE), `5e-5` (entropy), cosine decay + 5-epoch warmup
- **Validation:** Every 5 epochs, save if `val_loss < best_val_loss - 1e-4`
- **Early stop:** Patience = 15 epochs

### Dataset Strategy
- **Primary:** PEViD-HD (expand to all `.mpg` files, 80/10/10 split)
- **Augmentation:** Random crop/flip (256×256 → 224×224 during training, resize to 256×256 for eval)
- **Future:** Add UVG/MCL-JCV for generalization (post-funding)

---
## 📐 BD-Rate Computation Standard
Use this exact pipeline in `eval_rd_complete.py`:
```python
import numpy as np
from scipy.interpolate import CubicSpline

def compute_bd_rate(rate1, psnr1, rate2, psnr2):
    # Interpolate log2(bpp) vs PSNR for both curves
    cs1 = CubicSpline(psnr1, np.log2(rate1))
    cs2 = CubicSpline(psnr2, np.log2(rate2))
    # Common PSNR range
    psnr_min = max(min(psnr1), min(psnr2))
    psnr_max = min(max(psnr1), max(psnr2))
    psnr_range = np.linspace(psnr_min, psnr_max, 1000)
    # Area under curves
    avg_rate1 = np.mean(2**cs1(psnr_range))
    avg_rate2 = np.mean(2**cs2(psnr_range))
    return 100 * (avg_rate1 / avg_rate2 - 1)  # % BD-rate
```
Call with: `bd_rate = compute_bd_rate(my_rates, my_psnrs, x265_rates, x265_psnrs)`

---
## ✅ Validation Checklist (Before Claiming Results)
Run these checks after training:
1. **Monotonicity:** PSNR must strictly increase as BPP increases across λ
2. **Y-PSNR ≥ 24.5 dB at ≤ 0.08 bpp** (competitive with x265 CRF 28)
3. **BD-rate ≤ 0%** vs x265 CRF 28 on same 150-frame test clip
4. **VoE ratio ≥ 1.5x:** `surprise_anomaly / surprise_normal > 1.5`
5. **No NaN/Inf in gradients or loss** across all epochs
6. **Deterministic eval:** Same frames, same seeds, same `mode='inference'` quantizer

---
## 📦 Deliverables for AI Coder
1. Update `train_rd_from_scratch.py` with 256×256, 150 epochs, fixed scaling, mixed precision, best-checkpoint saving
2. Update `eval_rd_complete.py` with Y-PSNR, correct BPP, BD-rate function, UVG/HEVC loader
3. Add `voe_predictor.py` with joint surprise training & correct evaluation metric
4. Provide a `run_all.sh` script that trains λ sweep → evaluates → computes BD-rate → generates demos
5. Include a `config.yaml` for hyperparameters (resolution, λ list, LR, epochs, dataset paths)

---
## ⚠️ Critical Warnings
- **Do not claim competitive BD-rate until checklist passes.** Investors will verify.
- **Do not use RGB PSNR.** Video coding benchmarks require Y-channel.
- **Do not skip validation saves.** Final checkpoints often overfit to training distribution.
- **Ensure `quantizer(mode='inference')` is used in eval.** Training STE ≠ inference rounding.

Implement these changes, run the full pipeline, and share the `rd_curve.csv` + `voe_predictor_results.txt`. If the checklist passes, you'll have a defensible, benchmark-ready codec for your pitch and next funding round. 🚀
