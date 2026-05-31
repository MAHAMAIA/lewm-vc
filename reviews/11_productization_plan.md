# LeWM-VC Productization Plan — 6 Weeks to Design Partner Ready

## What "Productize" Means

The codec exists as a research prototype: Python scripts that load PyTorch checkpoints, process YUV files, and print metrics. A "product" means a design partner can run it on their own footage with minimal friction and get back measurable results they trust.

This plan covers three parallel tracks: **training data**, **deployment package**, and **evaluation pipeline**.

---

## Track 1: Training Data (Weeks 1–4)

The two-clip checkpoint will fail on a design partner's real footage. Fix this first.

### Week 1: Acquire and Preprocess VIRAT Ground

VIRAT Ground is a large-scale outdoor surveillance dataset with multi-camera pedestrian and vehicle events, 110+ hours of footage. It is the single highest-value addition to the training corpus.

**Download:**
```bash
# VIRAT Ground (from Kitware)
# Each video is provided as MPG4 with embedded object annotations
wget "https://data.kitware.com/api/v1/collection/56f56db28d777f753209ba9f/download" \
    -O virat_ground.zip
unzip virat_ground.zip -d datasets/virat_ground
```

**Preprocessing pipeline:**

```bash
# For each VIRAT clip:
# 1. Resize to 256×256
# 2. Convert to YUV420
# 3. Extract 16-frame GOP segments at 30 fps
# 4. Skip leading/trailing frames without events

for f in datasets/virat_ground/VIRAT_S_*.mp4; do
    basename=$(basename "$f" .mp4)
    ffmpeg -i "$f" \
        -vf scale=256:256,fps=30 \
        -frames:v 1000 \
        "datasets/virat_preprocessed/${basename}_%04d.png"
done
```

**Expected yield:** ~50,000 training frames from VIRAT (compared to ~200 currently). Remaining 19 PEViD-HD clips add another ~3,000 frames.

**Storage:** ~50 GB for preprocessed frames. Compress to tar archives for transfer.

### Week 2: Train Expanded Checkpoint

**Training protocol** (same two-phase approach as the paper):

**Phase 1 — Predictor pre-training (20 epochs):**
```bash
python scripts/training/track1_train.py \
    --data-dir datasets/virat_preprocessed/ \
    --phase pretrain \
    --epochs 20 \
    --batch-size 8 \
    --lr 1e-3 \
    --checkpoint checkpoints/v0.1_encoder.pt
```
- Freeze encoder, decoder, entropy model
- Train only the JEPA predictor on latent prediction MSE
- Loads existing v0.1 checkpoint's encoder/decoder weights

**Phase 2 — Joint fine-tuning (80 epochs):**
```bash
python scripts/training/track1_train.py \
    --data-dir datasets/virat_preprocessed/ \
    --phase joint \
    --epochs 80 \
    --batch-size 8 \
    --lr 5e-5 \
    --lambda-rd 0.05 \
    --checkpoint checkpoints/track1_pretrained.pt
```
- Unfreeze all components
- Train with full rate-distortion loss L = λR + D

**Expected wall time:** ~48 hours on a single MI300X or 4× T4 GPUs with data-parallel training. The 14.7M parameter model is small enough to train efficiently.

**Config sweep at 4 λ values:**
```bash
for lam in 0.001 0.005 0.01 0.05; do
    python scripts/training/track1_train.py \
        --data-dir datasets/virat_preprocessed/ \
        --phase joint \
        --epochs 40 \
        --batch-size 8 \
        --lr 5e-5 \
        --lambda-rd $lam \
        --checkpoint checkpoints/track1_pretrained.pt \
        --output checkpoints/track1_lambda_${lam}.pt
done
```

### Week 3: Validate on Held-Out Sequences

```bash
# Run LeWM-Eval on 5 VIRAT sequences not seen during training
for seq in virat_test_sequence_*.mp4; do
    python evaluation/semantic_probe.py \
        --frames "$seq" \
        --teacher yolov5su.pt \
        --output "results/${seq}_track1.json"
done

# Compare against the v0.1 two-clip checkpoint on the same sequences
# Script produces a side-by-side table
python experiment/04_evaluate_intra_rd.py \
    --checkpoint-new checkpoints/track1_lambda_0.05.pt \
    --checkpoint-old checkpoints/v0.1_lambda_0.05.pt \
    --output results/track1_vs_v01_comparison.json
```

**Gate criteria:** BD-accuracy on held-out sequences must be within 5 pp of the reported paper numbers on the original test set. If BD-accuracy degrades by more than 5 pp, the training data mix needs adjustment (add PEViD-HD frames back, reduce VIRAT weight, or increase epochs).

### Week 4: Deliverable

- `track1_lambda_0.05.pt` checkpoint released on GitHub Releases
- Validation report published: BD-accuracy comparison between v0.1 (2 clips) and v0.2 (VIRAT + full PEViD-HD) across 10+ held-out sequences
- Known limitations documented:
  - Still 256×256 resolution
  - Surveillance domain only (trained on outdoor pedestrian/vehicle scenes)
  - No VTM anchor comparison yet
  - Video analytics accuracy may vary on indoor scenes, nighttime, or extreme weather

---

## Track 2: Deployment Package (Weeks 1–4)

The design partner needs something they can run, not a Python script they debug.

### Week 1: Docker image
- Single Docker image: `docker pull mahamaia/lewm-vc:latest`
- Entry point: `docker run --gpus all -v /footage:/data mahamaia/lewm-vc encode /data/input.mp4 /data/output.mkv`
- Bundles: PyTorch runtime, checkpoint, ffmpeg, x265 (for baseline comparison)
- Output: compressed bitstream + decoded frames + metrics JSON

### Week 2: CLI tool
- `lewm-vc encode input.mp4 output.lewm` — compress a video
- `lewm-vc decode output.lewm frames/` — decompress to PNGs
- `lewm-vc compare input.mp4` — encode with x265 + LeWM-VC, compare BPP and accuracy
- Single binary (or pip-installable Python package with bundled model)

### Week 3: Metrics reporting
- `lewm-vc eval input.mp4` runs the full pipeline:
  1. Encode with x265 at 5 CRF values + LeWM-VC at 4 λ values
  2. Probe each with YOLOv5su
  3. Print table: codec, BPP, class accuracy, BD-accuracy
  4. Output `results.json` and `rate_accuracy.pdf`

### Week 4: Deliverable
- Docker image published on GitHub Container Registry
- CLI tool on PyPI: `pip install lewm-vc`
- One-page documentation: "How to evaluate LeWM-VC on your footage in 10 minutes"

---

## Track 3: Evaluation Pipeline (Weeks 1–2)

The evaluation script already exists (`evaluation/semantic_probe.py`). Package it.

### Week 1: Standalone Docker for LeWM-Eval
- `docker pull mahamaia/lewm-eval:latest`
- Entry point: `docker run -v /frames:/data mahamaia/lewm-eval probe /data/frames/ --teacher yolov5su`
- No codec dependency — works with any decoded frames

### Week 2: Baseline comparison script
- `lewm-vc compare input.mp4` wraps the full comparison
- Internal flow: extract frames → encode with x265 at 5 CRF → probe each → encode with LeWM-VC at 4 λ → probe each → compute BD-accuracy → plot curves → print table
- Output: single page PDF with results table + rate-accuracy plot

---

## Design Partner Onboarding (Week 5–6)

### Week 5: Onboarding kit
- One-page welcome doc: what they get, what we need from them, timeline
- Docker + CLI install guide
- Data sharing agreement template (anonymized footage, no retention)
- Success criteria template: "We expect to see X% bitrate reduction at Y% accuracy retention"

### Week 6: First evaluation run
- Partner shares 2–3 clips (1 minute each, typical scenes)
- Run full comparison pipeline
- Schedule 30-minute results review
- Discuss next steps: LOI, production pilot, paid contract terms

---

## Success Metrics at Week 6

| Metric | Target | How to measure |
|--------|--------|----------------|
| v0.2 checkpoint trained | Done | Checkpoint file exists, validation results published |
| Docker image published | Done | `docker pull` works |
| CLI on PyPI | Done | `pip install lewm-vc` works |
| Design partner evaluation | At least 1 completed | Partner has results JSON in hand |
| BD-accuracy on VIRAT | Measured | Published alongside v0.2 checkpoint |
| Full comparison script | Done | `lewm-vc compare input.mp4` produces results + plot |

---

## What This Does NOT Cover

- **Jetson/Ryzen AI deployment** — packaging for edge devices comes after the first paid customer
- **SaaS dashboard** — metering, multi-tenant, billing come after 3+ design partners
- **VTM anchor** — VVC comparison is valuable for VCM credibility but doesn't block design partner eval
- **Certification product** — LeWM-Eval certification as a paid service is years 3–5

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| VIRAT download or preprocessing issues | Low | Dataset is public, well-documented, direct download available |
| v0.2 doesn't generalize much better | Moderate | If so, need more diverse data (BDD100K, CRxK) — adds 1-2 weeks |
| Design partner slow to share footage | High | Start asking before the package is ready; pipeline building and data collection run in parallel |
| Docker build issues (CUDA, drivers) | Low | Test on a fresh cloud instance before publishing |
