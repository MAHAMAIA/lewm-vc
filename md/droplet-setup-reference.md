# LeWM-VC Droplet Setup & Training Reference

> **Purpose:** Stand up a new MI300X droplet for LeWM-VC training without rediscovering all the bugs.

## 1. Initial Setup

### Create Droplet
- **Provider:** DigitalOcean
- **Image:** PyTorch 1-Click (Ubuntu 24.04, ROCm pre-installed)
- **Size:** Premium AMD with MI300X (192GB HBM3)
- **Storage:** Add a 5TB volume for datasets (optional)

### System Prep
```bash
# SSH in
ssh root@<droplet-ip>

# Docker container (comes pre-installed as 'rocm')
docker ps  # should show 'rocm' container running

# Install tmux (critical — SSH drops regularly)
apt install -y tmux

# Verify GPU
docker exec rocm rocm-smi | tail -3
# Expected: MI300X, 0% utilization, 160W idle
```

## 2. Code Deployment

### Sync from local
```bash
# On local machine:
./scripts/sync_to_droplet.sh
# This rsyncs: src/ scripts/ configs/ tests/ + root files
# Then copies into Docker container via docker cp
```

### Manual code copy (if sync script broken)
```bash
# Copy files one at a time to avoid path bugs:
scp <local_file> root@<droplet-ip>:/root/le-maia/<path>
ssh root@<droplet-ip> "docker cp /root/le-maia/<path> rocm:/workspace/le-maia/<path>"

# Single scp + docker cp for a file:
scp src/scripts/train.py root@129.212.177.242:/root/le-maia/src/scripts/train.py && \
ssh root@129.212.177.242 'docker cp /root/le-maia/src/scripts/train.py rocm:/workspace/le-maia/src/scripts/train.py'

# Rebuild package after code changes:
docker exec rocm pip install -e /workspace/le-maia -q
```

### ⚠️ SCP Multi-File Bug
```bash
# DON'T do this — it flattens paths:
scp src/scripts/train.py configs/train_config.yaml root@ip:/root/le-maia/
# Files land at /root/le-maia/train.py and /root/le-maia/train_config.yaml (wrong!)

# DO use separate scp commands:
scp src/scripts/train.py root@ip:/root/le-maia/src/scripts/train.py
scp configs/train_config.yaml root@ip:/root/le-maia/configs/train_config.yaml
```

## 3. Dataset Setup

### VIRAT Download
```bash
# In container:
docker exec -w /workspace/le-maia rocm python3 scripts/download_virat.py

# The script:
#   - Downloads ~111GB zip from Kitware
#   - Extracts 456+ clips
#   - Converts to 256×256 PNG frames via ffmpeg
#   - Keeps the zip (no longer deleted)
#   - Output: datasets/virat/frames/<clip_name>/frame_XXXX.png

# Check progress:
docker exec rocm tmux capture-pane -t virat_dl -p -S -3

# After download, copy frames into container:
docker cp /root/le-maia/datasets/virat/frames/. rocm:/workspace/le-maia/datasets/virat/frames/
```

### Data Statistics
```bash
# Count clips and frames
docker exec -w /workspace/le-maia rocm python3 -c "
from lewm_vc.data import find_clips
clips = find_clips(['datasets/virat/frames'])
print(f'Clips: {len(clips)}, Frames: {sum(c.num_frames for c in clips)}')
"

# Typical: 456 clips, 1,308,279 frames, 169 GB
```

## 4. Training Launch

### Standard Launch (Phase 0-4 auto-progression)
```bash
tmux kill-session -t lewm_single 2>/dev/null
tmux new-session -d -s lewm_single 'docker exec -e PYTHONUNBUFFERED=1 -w /workspace/le-maia rocm python3 scripts/train_production.py --config configs/train_config.yaml --phase 0 --lambda 60 --resume <checkpoint_path>'
```

### Resume from Checkpoint
```bash
# Find latest checkpoint:
docker exec rocm ls -t /workspace/le-maia/checkpoints/*/lambda_60.0/best.pt | head -1

# Resume with specific phase:
docker exec -e PYTHONUNBUFFERED=1 -w /workspace/le-maia rocm python3 scripts/train_production.py \
  --config configs/train_config.yaml \
  --phase <0-4> \
  --lambda 60 \
  --resume /workspace/le-maia/checkpoints/<run_id>/lambda_60.0/best.pt
```

### Running in tmux
```bash
# Create session:
tmux new-session -d -s lewm_single 'docker exec ... long command'

# Check output:
tmux capture-pane -t lewm_single -p -S -5

# Attach:
tmux attach -t lewm_single

# Detach: Ctrl+B, then D

# Kill:
tmux kill-session -t lewm_single

# Note: tmux sessions die if Docker container is restarted
```

## 5. Monitoring

### Quick Status
```bash
# Currently training?
docker exec rocm ps aux | grep train_production | grep -v grep

# GPU utilization:
docker exec rocm rocm-smi | tail -3

# Current step from tmux buffer:
tmux capture-pane -t lewm_single -p -S - | grep step= | tail -1

# Latest checkpoint:
docker exec rocm ls -lt /workspace/le-maia/checkpoints/*/lambda_60.0/ | head -5

# Checkpoints on host (NOT inside container — check container path):
docker exec rocm ls /workspace/le-maia/checkpoints/
```

### Training Output
```
Phase X (<name>) — <steps> steps
  step=<global_step> loss=<value> lr=<lr>
    -> NEW BEST (val_loss=...) saved to best.pt
  Saved: .../step_<global_step>.pt
  Phase transition: <prev> -> <next>
```

### Phase Schedule (Current Config)
| Phase | Name | Steps | What Trains | λ | γ | Freeze |
|-------|------|-------|-------------|---|---|--------|
| 0 | Warmup | 50k | encoder, predictor, decoder | 1.0 | 2.0 | entropy, rate_ctrl, quantizer |
| 1 | Entropy Warmup | 25k | entropy_model | CLI | 3e-5 | encoder + pred+dec (entropy warmup) |
| 2 | QAT | 15k | decoder, entropy_model | CLI | 3e-5 | encoder, predictor |
| 3 | Decoder Refine | 15k | decoder | CLI | 1e-5 | encoder, predictor, entropy |
| 4 | Cooldown | 20k | decoder, entropy_model | CLI | 1e-5 | encoder, predictor |

## 6. Known Bugs & Fixes (Check Before Training)

### ✅ Fixed

| Bug | File | Fix | Date |
|-----|------|-----|------|
| `load_checkpoint` overwrites CLI `--lambda` with checkpoint value | `train.py:404` | Removed `self.lambda_val = ckpt.get(...)` | May 31 |
| Entropy model mu permanently zero (full layer zero-init) | `entropy.py` | Changed to sigma-only zero-init: `weight[latent_dim:]` | May 31 |
| ReLU kills gradient to mu channels | `entropy.py` | Replaced ReLU with GELU | May 31 |
| Decoder hidden_dim mismatch (128 vs 512) | `decoder.py` | Default hidden_dim restored to 512 | May 31 |
| Entropy hyper_channels mismatch (320 vs 256) | `entropy.py` | Default restored to 256 | May 31 |
| Decoder weights not loaded (evaluate.py) | `evaluate.py` | Verified strict=True for decoder load | May 31 |
| Temporal evaluation mixed clips (rglob bug) | `evaluate.py` | Changed to single-clip discovery | May 31 |
| nan_to_num on path garbled | `train.py` | Fixed to `nan_to_num_()` | May 30 |
| Rate clamp at 1e6 (too aggressive) | `train.py` | Changed to 1e12 safety net | May 30 |
| VIRAT zip deleted after extraction | `download_virat.py` | No longer delete zip | May 29 |
| Context length exceeded | `train.py` | Fix: slice to last context_len frames | May 29 |
| Empty val loader crash | `train.py` | Fix: `next(val_iter, None)` guard | May 29 |

## 7. Config Reference

### Key Config Parameters (`configs/train_config.yaml`)
```yaml
training:
  batch_size: 8
  num_workers: 4
  precision: fp32            # bf16 caused NaN — use fp32
  max_grad_norm: 1.0
  weight_decay: 0.01
  lr_encoder: 1.0e-4
  lr_predictor: 2.0e-5        # Reduced from 1e-4 (prevents destabilization)
  lr_decoder: 2.0e-5          # Reduced from 1e-4
  lr_entropy: 1.0e-4
  lr_warmup_steps: 1000
```

## 8. Evaluation

### Temporal IPPP Evaluation
```bash
docker exec -w /workspace/le-maia rocm python3 scripts/evaluate.py \
  --checkpoint /workspace/le-maia/checkpoints/<run_id>/lambda_60.0/best.pt \
  --config configs/train_config.yaml \
  --data datasets/virat/frames \
  --output eval_results \
  --temporal --num-frames 100 \
  --report
```

### Expected Metrics
| Metric | Target | Notes |
|--------|--------|-------|
| PSNR | > 28 dB | Requires proper λ (60) and decoder convergence |
| BPP | 0.1-0.5 | Non-zero, meaningful bitrate |
| I/P ratio | < 1.0 | P-frames cheaper than I-frames |
| LPIPS | < 0.3 | Perceptual quality |
| MS-SSIM | > 0.95 | Structural similarity |

## 9. SVC Training (Post-Base-Training)

```bash
docker exec -w /workspace/le-maia rocm python3 scripts/train_svc.py \
  --checkpoint checkpoints/<run_id>/lambda_60.0/best.pt \
  --data datasets/virat/frames \
  --steps 15000
```

## 10. Common Commands Summary

```bash
# Container interactive
docker exec -it rocm bash

# Run script
docker exec -w /workspace/le-maia rocm python3 scripts/<script>.py [args]

# Check GPU
rocm-smi | tail -3

# Kill training inside container
docker exec rocm pkill -9 -f train_production

# Rebuild package
docker exec rocm pip install -e /workspace/le-maia -q

# Reset hung container
docker restart rocm

# Clear Python cache
docker exec rocm sh -c "find /workspace/le-maia -name __pycache__ -exec rm -rf {} + 2>/dev/null"

# Verify entropy model activations
docker exec rocm python3 -c "from lewm_vc.entropy import HyperpriorEntropy; e=HyperpriorEntropy(192,256); print([type(l).__name__ for l in e.hyperprior_cnn])"
# Should show: Conv2d, GELU, Conv2d, GELU, Conv2d, GELU, Conv2d, GELU, Conv2d
```
