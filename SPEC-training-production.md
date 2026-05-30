# Sentinel Production Training — Spec

## Goal

Turn `src/scripts/train.py` into a **repeatable, observable, client-ready** training pipeline.
Every run is fully captured — config, code version, data snapshot, metrics, checkpoints —
so we can answer "what model is this and can I ship it?" in 10 seconds.

---

## Architecture

```
scripts/train_production.py    ← new entry point (thin wrapper)
         │
         ▼
src/scripts/train.py           ← existing training logic (unchanged)
         │
         ├── wandb.init()      ← experiment tracking
         ├── checkpoint        ← best-by-val-loss, with metadata
         ├── manifest.json     ← run fingerprint
         └── src/scripts/evaluate.py  ← post-training qualification
```

No changes to `src/scripts/train.py`. All production logic lives in the wrapper.

---

## 1. Run Manifest (`manifest.json`)

Written to `checkpoints/<run_id>/manifest.json` at start:

```json
{
  "run_id": "sentinel-20260528-xyz123",
  "timestamp": "2026-05-28T19:00:00Z",
  "phase": 0,
  "lambda_rd": 0.05,
  "config_file": "configs/train_config.yaml",
  "config_hash": "sha256:abc...",
  "git_commit": "a050cef",
  "git_dirty": false,
  "data": {
    "roots": ["datasets/virat/frames"],
    "num_videos": 5,
    "num_frames": 1500,
    "videos": ["video_01", "video_02", ...],
    "dataset_hash": "sha256:def..."         ← sha256 of sorted video dir listings
  },
  "wandb_run": "https://wandb.ai/...",
  "best_checkpoint": "checkpoints/phase0_best.pt"
}
```

Updated at end of training with `best_checkpoint` path.

---

## 2. W&B Integration

- **Project**: `sentinel`
- **Config logged**: full merged config dict (yaml + CLI overrides)
- **Metrics per step**: `train/loss`, `train/loss_jepa`, `train/loss_rd`, `train/lr`, `val/psnr`, `val/lpips`, `val/bpp`
- **Artifacts**: best checkpoint uploaded as W&B Artifact
- **CLI**: `--wandb` flag to enable. Falls back to no-op if no API key or offline.

---

## 3. Checkpoint Manager

Replace simple `torch.save` with:

```
checkpoints/
  <run_id>/
    latest.pt              ← every `save_interval` steps
    best.pt                ← best val loss (updated atomically)
    step_05000.pt          ← periodic named snapshots
    manifest.json
    wandb_run_id.txt
```

`best.pt` saves with metadata block:
```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_loss": best_val_loss,
    "config_hash": config_hash,
    "git_commit": GIT_COMMIT,
    "timestamp": TIMESTAMP,
}, path)
```

---

## 4. Graceful Shutdown & Resume

Signal handlers for SIGTERM/SIGHUP:
- Save checkpoint on signal
- W&B marks run as `crashed`
- Resume picks up `latest.pt` and continues from last step

Resume via `--resume <run_id>` or `--resume latest`.

---

## 5. Client-Ready Qualification Gate

New script `scripts/qualify_model.py` that loads a checkpoint and runs:

| Metric     | Threshold    | Source                    |
|------------|-------------|---------------------------|
| PSNR       | ≥ 26 dB     | evaluate.py               |
| LPIPS      | ≤ 0.35      | evaluate.py               |
| BPP        | ≤ 0.15      | entropy model avg         |
| Task mAP   | drop ≤ 2%   | task_probe.py             |
| SVC recon  | pass/fail   | train_svc.py --validate   |

Outputs a `qualification.json`. Returns exit 0 if all pass, non-zero with details if not.

---

## Implementation Order

### Phase A — Foundation (this session)
1. Create `scripts/train_production.py` wrapper
2. Run manifest generation
3. Add `--wandb` flag (no-op if not available)
4. Best-checkpoint-by-val-loss
5. Sync & test on droplet with Phase 0

### Phase B — Resilience
6. Signal handlers for graceful shutdown
7. Resume from `latest.pt`
8. W&B Artifact upload

### Phase C — Qualification
9. `scripts/qualify_model.py`
10. Integration into deploy pipeline (post-training auto-qualify)

---

## Files to Create / Modify

| File | Action | Why |
|------|--------|-----|
| `scripts/train_production.py` | **Create** | Entry point wrapper |
| `scripts/qualify_model.py` | **Create** | Client-ready gate |
| `pyproject.toml` | Edit | Add `wandb` optional dep |
| `configs/train_config.yaml` | Edit | Add `wandb:` section |
| `scripts/deploy_and_train.sh` | Edit | Call `train_production.py` instead of `train.py` |

---

## Config Additions

```yaml
wandb:
  project: sentinel
  entity: null           # personal account default
  enabled: true
  log_interval: 50       # less frequent than console logging
```

---

## CLI Interface

```bash
python scripts/train_production.py \
    --config configs/train_config.yaml \
    --phase 0 \
    --lambda 0.05 \
    --wandb \
    --resume latest

# qualification
python scripts/qualify_model.py \
    --checkpoint checkpoints/run_id/best.pt \
    --config configs/train_config.yaml
```
