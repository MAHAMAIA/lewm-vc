# rate_weight Fix — 2026-05-31

## The Bug

Phase 0 was supposed to train encoder + predictor with pure JEPA signal (no rate, no distortion), but `rate_loss` was **always** added to `total_loss` without any multiplier — neither λ nor any other weight controlled it.

At `src/scripts/train.py:246`:
```python
total_loss = (
    rate_loss          # ← NO multiplier. Always present.
    + lambda_val * distortion_loss   # λ=0 → 0 during Phase 0
    + gamma * jepa_loss              # γ=2.0 → 396
)
```

Setting `lambda: 0.0` in the Phase 0 config killed the distortion term, but `rate_loss` (~908k at step 800) was still 2300× larger than `γ * jepa_loss` (~396). The encoder received a massive gradient from the **frozen random entropy model** — essentially noise — drowning out the JEPA signal.

## The Fix

### 1. Added `rate_weight` multiplier (`src/scripts/train.py`)

```python
# Line 180
rate_weight = phase_cfg.get("rate_weight", 1.0)

# Line 248
total_loss = (
    rate_weight * rate_loss    # ← now controllable per-phase
    + lambda_val * distortion_loss
    + gamma * jepa_loss
    + delta * sigreg_loss
    + surprise_loss
)
```

Default is `1.0` (full rate contribution) for all phases unless overridden.

### 2. Config change (`configs/train_config.yaml`)

Added `rate_weight: 0.0` to Phase 0:

```yaml
phase0:
    steps: 30000
    lambda: 0.0
    gamma: 2.0
    delta: 0.001
    rate_weight: 0.0    # ← NEW: zero out frozen entropy rate
    freeze: ["entropy_model", "rate_controller", "quantizer"]
    mse_weight: 1.0
    lpips_weight: 0.0
```

### 3. Restarted training from scratch

- Killed the old `sentinel-p0-l60.0-f238be` run (step ~800, rate-dominated)
- Launched new run in tmux session `train` on the droplet
- Training command: `python3 scripts/train_production.py --config configs/train_config.yaml --phase 0 --lambda 60`

## Expected Phase 0 Behavior

With `rate_weight=0` and `lambda=0`, Phase 0 loss is:

```
total_loss = 0*R + 0*D + γ*J + δ*SIGReg
           = 2.0 × jepa_loss + 0.001 × sigreg_loss
```

The encoder/predictor receive gradient **only** from the JEPA target network (predicting future frame latents) plus a small SIGReg regularization. No random noise from the frozen entropy model.

Phase 1+ phases use the default `rate_weight: 1.0` (not specified in config), so they get the full rate-distortion-JEPA loss as intended.

## Attaching to Monitor

```bash
ssh root@129.212.177.242 -t "tmux attach -t train"
```
