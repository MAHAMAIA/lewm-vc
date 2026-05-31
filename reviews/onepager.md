# LeWM-VC: Machine-Native Video Compression

**A codec built for the actual consumer of surveillance video — machine perception systems, not human eyes.**

---

## The Problem

Global video surveillance generates more footage daily than humans could ever watch. The consumer is now exclusively machine perception: object detectors, trackers, behavior analytics. But every frame is compressed with codecs designed for human vision — preserving eyelash texture while discarding the low-frequency semantic structure that detectors need.

**The cost:** At 1.95 bits per pixel, a YOLOv5 probe trained on H.265-decoded frames achieves 79.3% accuracy. The same probe trained on LeWM-VC compressed latents at matched bitrate achieves 86.5%. Seven percentage points lost to pixels no human will ever see.

---

## What We Built

**LeWM-VC** is a video codec that skips pixel reconstruction entirely. Each frame becomes a compact semantic latent (192×16×16) through a Vision Transformer encoder. Temporal redundancy is exploited by a learned transformer predictor that forecasts the next latent directly — no block-based motion vectors, no optical flow. Only the prediction residual is transmitted.

**Key results on PEViD-HD surveillance video:**

| Metric | Value |
|--------|-------|
| P-frame bitrate reduction vs. all-intra | 62% |
| Accuracy advantage over H.265 at 1.95 BPP | +7.2 pp |
| Accuracy advantage over H.265 at 0.11 BPP | +1.7 pp |
| Inference throughput (NVIDIA T4) | 80+ fps |
| Memory per frame | 12 KB (vs. 1.5 MB raw) |
| Parameters | 14.7M |
| GPU memory (inference) | 1.2 GB |

**LeWM-Eval** is the evaluation framework that ships with the codec. It is codec-agnostic — customers can independently verify our numbers and benchmark any competitor against us using the same methodology.

All results reproducible from public checkpoints. Code, weights, and Docker container at [github.com/MAHAMAIA/lewm-vc](https://github.com/MAHAMAIA/lewm-vc).

---

## Why Now

Three things converged in 2024–2026:

1. **JEPA architectures** made stable latent prediction possible without pixel reconstruction.
2. **Edge GPU cost crossover** (Jetson Orin, AMD Ryzen AI) makes a 14.7M-parameter transformer codec run real-time at commodity price points.
3. **MPEG Video Coding for Machines (VCM)** is formalizing a machine-oriented compression standard right now, and the **EU AI Act** mandates task-accuracy documentation for high-risk AI systems including surveillance.

The window to define the standard is open. Standards follow adoption. We are building the codec that customers need today.

---

## Stage

Pre-revenue. Proof-of-concept validated on two PEViD-HD clips. Dataset expansion to 20+ sequences and VIRAT Ground 2.0 underway. Raising $3–5M seed to fund team, data, and benchmark infrastructure.

---

## Contact

**Preetam Mukherjee** — preetam@mahamaia.com  
**Soumyajit Mandal (Ph.D., MIT)** — Co-Founder, CTO  
San Francisco, CA
