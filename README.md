# LeWM-VC: JEPA-Based Learned Video Codec

**Reference implementation for machine-oriented video compression.**

LeWM-VC is a learned video codec that compresses frames into a compact latent space using Joint Embedding Predictive Architecture (JEPA), eliminating hand-crafted motion vectors. It achieves 62% temporal bitrate savings over all-intra coding while preserving 7.2 pp more classification accuracy than H.265 at matched bitrate.

The evaluation methodology used to measure these results is **LeWM-Eval** — a codec-agnostic evaluation framework for machine-oriented video compression. LeWM-Eval lives in its own repository: [github.com/MAHAMAIA/lewm-eval](https://github.com/MAHAMAIA/lewm-eval).

This repository contains:
- **LeWM-VC** — the reference codec implementation (encoder, decoder, JEPA predictor, GMM entropy model)
- **Reproduction scripts** — numbered experiments (01–12) that reproduce all paper results
- **Benchmark outputs** — per-milestone evaluation logs and CSVs

---

## For VCM Researchers

The evaluation methodology used in this project (semantic probing, bitrate matching, cross-teacher validation) is packaged as **LeWM-Eval** — a standalone, codec-agnostic framework at [github.com/MAHAMAIA/lewm-eval](https://github.com/MAHAMAIA/lewm-eval).

LeWM-Eval is designed to align with the MPEG VCM Common Test Conditions structure. Any codec can be evaluated by decoding frames to disk and running the probe pipeline. The LeWM-VC codec in this repo is a reference implementation that validates the methodology.

---

## Repository Structure

```
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

### Evaluate a codec with LeWM-Eval

The standalone evaluation framework is at [github.com/MAHAMAIA/lewm-eval](https://github.com/MAHAMAIA/lewm-eval):

```bash
git clone https://github.com/MAHAMAIA/lewm-eval.git
cd lewm-eval
pip install ultralytics torch torchvision opencv-python pillow tqdm
python evaluation/semantic_probe.py --frames /path/to/decoded/frames/
```

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
| LeWM-VC reference codec | Released, public checkpoints |
| Reproduction scripts | Released (experiment/01–12) |
| LeWM-Eval evaluation framework | Separate repo: [github.com/MAHAMAIA/lewm-eval](https://github.com/MAHAMAIA/lewm-eval) |
| MPEG VCM engagement | Initial outreach underway |
| Standalone PyPI package | In development |
| Public leaderboard | Planned |

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
