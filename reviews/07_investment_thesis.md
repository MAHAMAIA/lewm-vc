# The MAHAMAIA Thesis
## Why Machine-Native Video Compression Is the Infrastructure Bet of the Next Decade — and Why This Team Has the Window

**Prepared by:** MAHAMAIA Systems  
**Date:** May 2026  
**Classification:** Confidential — For Prospective Investors

---

> *"The companies that deploy working codecs into production define the standard, because standards follow adoption, not the reverse."*

---

## The Thesis in One Sentence

MAHAMAIA Systems builds LeWM-VC, a machine-native video codec for surveillance infrastructure. It saves 62% storage at better detection accuracy than H.265. LeWM-Eval ships with the codec and is codec-agnostic — customers verify our numbers independently and benchmark competitors against us.

---

## The Structural Mismatch

The codec stack was built for human eyes. H.265 — the dominant compression standard in surveillance infrastructure — was designed to produce visually pleasing pixel reconstructions. It minimizes pixel-level distortion. It preserves high-frequency texture and fine spatial detail that human viewers appreciate. It allocates bits to information that machine perception systems cannot use and fails to preserve the semantic structure that those systems require.

The consequence is measurable. At 1.95 bits per pixel, a detection model operating on H.265-decoded frames achieves 79.3% object classification accuracy. The same model operating on LeWM-VC compressed latents at identical bitrate achieves 86.5%. Seven and a half percentage points of detection accuracy sacrificed on the altar of a pixel fidelity metric that no machine in the pipeline ever consults.

This is not a gap that will be closed by incremental improvement to H.265. It is a structural mismatch between the optimization target of the compression infrastructure and the requirements of its actual consumers. The infrastructure was built for a consumer that is increasingly absent. That is a large, durable, addressable market failure.

---

## The Codec Is Real

LeWM-VC is not a proposal. It is a working codec with public checkpoints, a Docker inference container, and exact command lines documented in the repository. Results are reproducible from public artifacts. The architecture:

- **ViT-Tiny encoder** with 6 layers, 192-dim hidden, 3 heads. Processes 256×256 frames at 16×16 patch resolution. Output: latent grid of shape 192×16×16.
- **JEPA temporal predictor** — 8-layer transformer, 256-dim hidden, 4 heads, context length 4. Forecasts the next latent directly. No hand-crafted motion vectors, no optical flow. Only the prediction residual is transmitted.
- **GMM entropy model** — 2-component Gaussian Mixture Model with hyperprior CNN. The latent distribution is bimodal (dense background, sparse foreground). The GMM captures this structure. Replacing it with a Laplace prior increases bitrate 5.6×.
- **Decoder** — 4 ConvTranspose2d layers with residual blocks. Recovers 256×256 RGB. 2.3M parameters.
- **Total:** 14.7M parameters. 80+ fps on T4. 1.2 GB peak GPU memory. 12 KB/frame memory bandwidth.

The ablation studies support the architecture's necessity: without the two-phase predictor pre-training, temporal savings collapse from 62% to 4.4%. Without affine normalization, PSNR drops ~1.5 dB at matched bitrate. The performance stems from principled design, not incidental tuning.

---

## Why Now

Three enabling conditions converged in 2024–2026 that make LeWM-VC's architecture viable in a way it was not previously:

**JEPA maturation.** Joint Embedding Predictive Architectures (LeCun 2022; LeWM 2025) demonstrated that predicting future states in compact latent space — without reconstructing pixels — produces stable, semantically rich representations without representational collapse. The theoretical foundation is new. Attempting the same architecture in 2021 would have produced a system that collapsed during training.

**Edge compute cost crossover.** The NVIDIA Jetson Orin and AMD Ryzen AI NPU have made real-time transformer inference economically viable at the camera level. A 14.7M parameter model at 80+ fps is deployable on current edge hardware at commodity prices. This was not true when DVC was published in 2019 or when DCVC was published in 2022.

**MPEG VCM formalization and AI Act enforcement.** The standards process creates a compliance market that did not exist before 2023. The EU AI Act creates a regulatory requirement for documented task accuracy that cannot be met with existing tooling. The window for deploying the first production-grade machine-native codec is open.

---

## How the Moat Works (Product-First)

The moat has three layers, ordered by when they become effective:

**Layer 1 — Deployment data and customer relationships.** A codec trained on real surveillance data learns surveillance-specific dynamics that a general-purpose codec cannot replicate without equivalent data. This creates a data flywheel: more deployment → better model → better compression → more deployment. This moat starts with the first design partner.

**Layer 2 — Deployment data.** A codec trained on real surveillance feeds learns domain-specific dynamics that no competitor can replicate without equivalent deployment scale. This is the near-term moat.

**Layer 3 — IP.** A provisional patent on the JEPA temporal prediction architecture applied to video compression is being filed. This creates priority date and a negotiating position with hardware partners.

---

## The Business Model

**Stream 1 — SaaS / Edge SDK (12–18 months).** Per-camera-per-month for surveillance operators. At $8/camera/month: a 500-camera deployment is $48K ARR. This is the near-term revenue engine.

**Stream 2 — Codec SDK Licensing (18–36 months).** Per-device royalty to camera OEMs. Activated by VCM standardization. Sequential: SaaS proves demand, then OEMs license.

**Stream 3 — Benchmark Certification (24–48 months).** Paid LeWM-Eval certification for codec vendors. High margin, downstream of adoption.

---

## Risks

**The dataset problem is real.** The empirical foundation rests on two training clips. Dataset expansion to the full PEViD-HD corpus and VIRAT Ground 2.0 is underway, with results expected Q3 2026.

**The enterprise BD gap exists.** Founding team covers the full technical stack. VP BD hire is first non-founder role, budgeted in seed.

**The standards timeline is uncertain.** MPEG processes are unpredictable. Mitigation: SaaS revenue is independent of VCM timeline.

---

## Why This Team

**Preetam Mukherjee — Co-Founder, CEO.** Designed and implemented LeWM-VC and LeWM-Eval from first principles. Reproducibility posture reflects engineering discipline that transfers to production.

**Soumyajit Mandal — Co-Founder, CTO.** Ph.D. MIT, 26 patents, 175+ publications, ~7,600 citations. Current: Brookhaven National Laboratory. Prior: Schlumberger-Doll Research, faculty at Case Western and University of Florida. Custom ASIC design, edge deployment, SBIR pathways.

The combination of learned codec architecture (Preetam) and custom silicon / hardware engineering (Soumyajit) directly addresses the company's two hardest problems: building the right compression algorithm and deploying it at the edge.

---

## Closing

MAHAMAIA Systems is building the codec that machine perception infrastructure runs on. The product is real, the results are measurable, and the timing is right because the three enabling conditions — JEPA, edge compute, VCM — are all aligned for the first time. The companies that deploy working codecs into production in 2026–27 will define the standard, because standards follow adoption, not the reverse.

We are building the thing customers need. That is the investment.

---

*MAHAMAIA Systems — May 2026*  
*All technical results reproducible at github.com/MAHAMAIA/lewm-vc*  
*This document is confidential and intended solely for the named recipient.*
