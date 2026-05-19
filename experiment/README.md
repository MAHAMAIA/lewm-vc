# LeWM-VC Reproduction Suite

## Directory Layout

```
experiment/
├── README.md
├── env.sh                       # Environment setup (PYTHONPATH, paths)
├── 01_download_data.sh          # Download PEViD-HD + UVG datasets
├── 02_train_intra_ae.sh         # Train autoencoders at 6 λ values
├── 03_train_gmm_entropy.sh      # Train GMM entropy models at 6 λ values
├── 04_evaluate_intra_rd.py      # Reproduce Table 4 (intra-frame RD curve)
├── 05_train_predictor.py        # JEPA predictor pre-training (Phase 1)
├── 06_train_joint_temporal.py   # Joint fine-tuning (Phase 2)
├── 07_evaluate_temporal.py      # Reproduce Table 5 (temporal compression)
├── 08_probe_semantic.py         # Reproduce Tables 6, 7 (probe accuracy)
├── 09_ablation_studies.py       # Reproduce Table 8 (ablations)
├── 10_compute_surprise.py       # Reproduce Sec 5.5 (surprise metric)
├── 11_measure_efficiency.py     # Reproduce Table 9 (fps / params)
├── 12_evaluate_uvg.py           # Reproduce Table 10 (UVG cross-dataset)
└── common.py                    # Shared utilities (dataset loader, metrics)
```

## Prerequisites

- Python ≥ 3.10 with PyTorch ≥ 2.0
- `pip install -e ".[dev]"` from repo root (installs `lewm_vc` package)
- FFmpeg with libx265 for x265 baseline encoding
- YOLOv5 (auto-downloaded by Ultralytics on first use)
- ~150 GB free disk for datasets + checkpoints
- NVIDIA GPU with ≥ 16 GB VRAM (or AMD MI300X for training)

## Running Order

Experiments are numbered and can be run sequentially:

```bash
# Step 1: Environment
source experiment/env.sh

# Step 2: Download datasets (PEViD-HD + UVG)
bash experiment/01_download_data.sh

# Step 3-6: Training (skip if using provided checkpoints)
bash experiment/02_train_intra_ae.sh      # ~8 hours on MI300X
bash experiment/03_train_gmm_entropy.sh   # ~4 hours
python experiment/05_train_predictor.py   # ~2 hours
python experiment/06_train_joint_temporal.py  # ~6 hours

# Step 7-12: Evaluation (use existing checkpoints)
python experiment/04_evaluate_intra_rd.py
python experiment/07_evaluate_temporal.py
python experiment/08_probe_semantic.py
python experiment/09_ablation_studies.py
python experiment/10_compute_surprise.py
python experiment/11_measure_efficiency.py
python experiment/12_evaluate_uvg.py
```

## Checkpoints

Pre-trained checkpoints used by the paper:

| File | Source | Used By |
|------|--------|---------|
| `checkpoints_milestone1/ae_lambda_*.pt` | 02_train_intra_ae | 04, 09 |
| `checkpoints_milestone1/entropy_lambda_*.pt` | 03_train_gmm_entropy | 04, 09 |
| `checkpoints_milestone2/temporal_epoch*.pt` | 05+06 | 07 |
| `checkpoints_milestone4b/` | (evaluation only) | 08 |

## Results Mapping

| Experiment Script | Paper Section | Table / Figure |
|-------------------|--------------|----------------|
| 04 | Sec 4.2 | Table 4 |
| 07 | Sec 4.3 | Table 5 |
| 08 | Sec 4.4 | Tables 6, 7 |
| 09 | Sec 4.6 | Table 8 |
| 10 | Sec 4.5 | Figure 2 |
| 11 | Sec 4.7 | Table 9 |
| 12 | Sec 5.1 | Table 10 |
