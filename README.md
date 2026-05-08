# LeWM-VC: JEPA-Based Video Codec

Learned video compression using Joint Embedding Predictive Architecture (JEPA). Compresses video frames into a compact latent space, predicts future latents via a transformer predictor, and codes only the residual — achieving temporal compression without explicit motion vectors.

## Status (May 2026)

LeWM-VC is a research codec under active development. All results below are reproducible with the scripts in this repository.

### What Works

| Component | Status | Key Result |
|-----------|--------|------------|
| Intra-frame compression (ViT encoder + GMM entropy model) | Working | Monotonic RD curve, 0.11–0.25 BPP at 22–29 dB PSNR on PEViD-HD at 256x256 |
| JEPA temporal prediction | Working | 62% bitrate savings over all-intra coding, P/I ratio 0.37x per frame |
| Latent-space semantic preservation | Working | 86.5% class accuracy vs 79.3% for x265 at matched bitrate |
| Surprise-gated quantization (VOE) | Mechanism functional | Produces calibrated surprise metric; thresholds require dataset-specific calibration |
| FFmpeg plugin | In development | C wrapper exists, not yet integrated with trained models |
| SIGReg regularization | In development | Current code uses Gaussian KL prior; sketched Cramér-Wold implementation in progress |

### What's Not Yet Done

- BD-rate comparison against VVC/H.266
- Multi-resolution support (currently 256x256 only)
- Real-time encoding on edge hardware
- Surprise gating on labeled anomaly datasets

## Architecture

```
I-Frame Path (Intra-frame Coding):
  Input(256x256) ──► [ViT Encoder(6L)] ──► Latent[192x16x16] ──► [GMM Entropy] ──► Bitstream
                                                                                      │
  Bitstream ──► [GMM Entropy(decode)] ──► Latent[192x16x16] ──► [Decoder(4L)] ──► Output(256x256)


P-Frame Path (Inter-frame with JEPA Temporal Prediction):
                                          ┌──────────────────────────┐
                                          │  JEPA Predictor          │
                                          │  8L transformer, ctx=4   │
                                          └───────────┬──────────────┘
                                                      │ Predicted Latent
                                                      ▼
  Input(256x256) ──► [ViT Encoder(6L)] ──► Latent ───┼──────────► [⊖] ──► Residual[192x16x16] ──► [GMM Entropy] ──► Bitstream
                                        │              │                   │                                   │
                                        │       (pred) │                   │                                 Decode
                                        │              └───────────────────┘                                   │
                                        │                                                                       │
                                        └────────────────────────────► [⊕] ◄── Residual[192x16x16] ◄── [GMM Entropy] ◄── Bitstream
                                                                    │
                                                             Latent[192x16x16]
                                                                    │
                                                          [Decoder(4L)] ──► Output(256x256)
```

**Components:**
- **Encoder:** ViT-Tiny, 6 layers, 192-dim latent grid (16x16 spatial)
- **Predictor:** 8-layer transformer, 256-dim hidden, 4 heads, context length 4
- **Entropy Model:** 2-component Gaussian Mixture Model with hyperprior CNN
- **Decoder:** 4-layer ConvTranspose upsampling with residual blocks

## Quick Start

### Installation

```bash
git clone https://github.com/thepreetam/le-maia.git
cd le-maia
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install torch torchvision opencv-python-headless numpy tqdm
```

### Reproducing Results

Train intra-frame codec and compute RD curve:

```bash
python3 milestone1_rd_curve.py
```

Train temporal predictor and measure P/I ratio:

```bash
python3 milestone2_temporal.py
```

Evaluate surprise gating:

```bash
python3 milestone3_surprise_gating.py
```

Run latent probe benchmark (semantic preservation):

```bash
python3 milestone4b_latent_probe.py
```

All scripts require PEViD-HD dataset in datasets/pevid-hd/. Download from EPFL FTP: tremplin.epfl.ch, username datasets@mmspgdata.epfl.ch, password ohsh9jah4T (see Korshunov & Ebrahimi, SPIE 2013 for details).

## Results Summary

### Temporal Compression (Milestone 2)

| Mode | BPP | PSNR |
|------|-----|------|
| All-intra | 0.228 | 25.13 dB |
| Temporal (IPPP) | 0.087 | 25.40 dB |
| Savings | 61.79% | |

### Semantic Preservation (Milestone 4b)

| Method | Objectness Acc | Class Acc | BPP |
|--------|----------------|-----------|-----|
| LeWM-VC latent probe | 97.5% | 86.5% | 1.95 |
| x265 pixel probe | 97.7% | 79.3% | 1.95 |

## Citation

```bibtex
@misc{lewmvc2026,
  author = {Preetam Mukherjee},
  title = {LeWM-VC: JEPA-Based Video Codec with Temporal Latent Prediction},
  year = {2026},
  url = {https://github.com/thepreetam/le-maia}
}
```

## License

MIT https://github.com/thepreetam/le-maia
