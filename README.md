# LeWM-Eval: Reproducible Benchmark for Machine-Oriented Video Compression

**Measure what matters: task accuracy at matched bitrate.**

LeWM-Eval is a codec-agnostic evaluation framework for machine-oriented video compression. It measures how well a compressed video representation preserves task-relevant semantic information — object location, class, motion trajectory — rather than pixel fidelity (PSNR, SSIM).

This repository provides:
- **LeWM-Eval** — the evaluation framework (codec-agnostic semantic probing pipeline)
- **LeWM-VC** — a reference codec implementation (JEPA-based latent prediction, included to validate and demonstrate the methodology)

---

## For VCM Researchers

LeWM-Eval is designed to align with the MPEG VCM (Video Coding for Machines) Common Test Conditions structure:

- **Bitrate matching:** Sweeps codec quality parameters (CRF, λ) to ±5% of target BPP
- **Semantic probing:** Trains identical lightweight CNN probes on any codec's output vs. frozen teacher pseudo-labels
- **Cross-teacher validation:** Supports multiple detectors (YOLOv5s, YOLOv5su) — teacher-agnostic comparisons
- **Metrics reported:** BPP, objectness accuracy, class accuracy, rate-accuracy curves
- **Planned:** Tracking accuracy (ByteTrack+HOTA), privacy leakage (ReID probing), RDA curves (BD-rate analog)

The framework is codec-agnostic: any codec — H.265, H.266/VVC, AV1, or any learned codec — can be evaluated by storing decoded frames to disk and running the probe pipeline against them. See [`evaluation/`](./evaluation/) for the implementation.

---

## Repository Structure

```
├── evaluation/
│   ├── semantic_probe.py          # Standalone semantic probing entry point
│   ├── README.md                  # Instructions for evaluating any codec
│   └── benchmark_utils.py         # Shared utilities (coming in PyPI release)
├── lewm_vc/                       # Reference codec implementation
│   ├── encoder.py                 # ViT-Tiny encoder
│   ├── decoder.py                 # ConvTranspose decoder
│   ├── predictor.py               # JEPA transformer predictor
│   ├── entropy.py                 # GMM entropy model
│   └── quant.py                   # Scalar quantization
├── checkpoints_milestone*/        # Public pretrained weights
├── experiment/                    # Numbered reproduction scripts (01–12)
└── benchmark_milestone*/          # Per-milestone benchmark outputs
```

## Quick Start: Evaluating Your Codec

```bash
pip install lewm-eval  # standalone package (coming soon)

# Or use the source directly:
git clone https://github.com/thepreetam/le-maia.git
cd le-maia
pip install -e ".[dev]"
```

### Evaluate an external codec

```bash
python evaluation/semantic_probe.py \
    --codec x265 \
    --input /path/to/your/video.yuv \
    --frames 100 \
    --teacher yolov5s \
    --output results.json
```

Supports any codec that can produce decoded frames. See [`evaluation/README.md`](./evaluation/README.md) for adding custom codec wrappers.

### Reproduce LeWM-VC results

```bash
python experiment/04_evaluate_intra_rd.py    # Intra-frame RD curve (Table 4)
python experiment/07_evaluate_temporal.py     # Temporal compression (Table 5)
python experiment/08_probe_semantic.py        # Semantic probe accuracy (Tables 6, 7)
```

---

## Key Results (LeWM-VC Reference Codec)

| Metric | Value |
|--------|-------|
| Class accuracy advantage vs. H.265 @ 1.95 BPP | **+7.2 pp** (86.5% vs 79.3%) |
| Class accuracy advantage vs. H.265 @ 0.11 BPP | **+1.7 pp** (94.4% vs 92.7%) |
| P-frame bitrate reduction vs. all-intra | 62% (GOP=8) |
| Inference throughput (NVIDIA T4) | 80+ fps |
| Total parameters | 14.7M |

All results reproducible from public checkpoints. Exact command lines, random seeds, and checkpoint hashes documented in [`experiment/`](./experiment/).

---

## Status (May 2026)

| Component | Status |
|-----------|--------|
| Semantic probing pipeline (codec-agnostic) | Released |
| LeWM-VC reference codec | Released, public checkpoints |
| x265/H.264 wrappers | Included |
| MPEG VCM engagement | Initial outreach underway |
| Standalone PyPI package (`lewm-eval`) | In development |
| VVC/H.266 wrapper | Planned |
| Public leaderboard | In development |

---

## Citation

```bibtex
@misc{lewmeval2026,
  author = {Preetam Mukherjee},
  title = {LeWM-Eval: A Reproducible Benchmark for Machine-Oriented Video Compression},
  year = {2026},
  url = {https://github.com/thepreetam/le-maia}
}
```

## License

MIT
