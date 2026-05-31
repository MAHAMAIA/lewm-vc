# Research Scaling Roadmap: LeWM-VC Beyond the Proof-of-Concept

> **Date:** 2026-05-16  
> **Context:** Senior-collaborator advice on scaling LeWM-VC from a proof-of-concept arXiv preprint to a top-venue submission. Organized as a prioritization framework, not a review checklist.

---

## 1. The One Thing That Unlocks Everything Else

The entire paper currently rests on **two training clips**. Before any architectural improvement, the single highest-leverage investment is a proper training corpus.

| Timeline | Action | Impact |
|----------|--------|--------|
| **Short term** | Train on all available PEViD-HD sequences, not just two | Rerun every existing table with error bars — transforms the paper's evidentiary status overnight |
| **Medium term** | Incorporate VIRAT Ground, MEVA, or CityFlow (large-scale, multi-camera, multi-scenario surveillance datasets) | Legitimately claim the codec generalizes within the surveillance domain — the actual deployment claim |

**Why this matters architecturally:** The GMM hyperprior's catastrophic failure on UVG (BPP saturating at 1.95) is almost certainly because it was fit to the latent distribution of two clips. A properly diverse training set would make the entropy model more robust and likely close a meaningful fraction of the UVG gap without any architectural changes.

---

## 2. The Surprise Metric Needs a Second Paper or Needs to Go

The surprise mechanism is the weakest part of the work — presented as a contribution, never demonstrated working. Two clean paths:

### Path A — Make it a real contribution

Requires:
1. Evaluating on UCF-Crime or ShanghaiTech Anomaly where genuine sudden events exist
2. Showing the gate actually activates and that resulting bitrate allocation measurably improves downstream detection on those frames
3. Comparing against learned adaptive bitrate methods from the VCM literature

This is essentially a separate paper — *"Surprise-Gated Adaptive Quantization for Surveillance Codecs"* — and a strong one if the evidence holds.

### Path B — Demote it honestly

Move the surprise metric to a diagnostic tool section. Remove it from the abstract's claim list. Describe it as "a training-free monitoring signal whose adaptive gating utility is left to future work."

**Either path is fine. Straddling them — presenting it as a contribution but noting it never activates — is the one thing that genuinely damages credibility with expert reviewers.**

---

## 3. The Latent Design Has Untapped Research Surface

### Spatial cross-attention in the predictor
The current design spatially averages each context frame before temporal processing, collapsing spatial structure. Replacing this with full spatial cross-attention (attend directly to 16×16 patch tokens across frames) is a one-ablation change that would likely improve the P/I ratio and produce inspectable spatial attention maps — making the "we learn motion implicitly" claim concrete rather than theoretical.

### Multi-scale latent grids
The paper identifies dense occlusion (>5 objects in a 16×16 patch) as a failure mode — a direct consequence of fixed patch resolution. A two-scale design (one 16×16 grid for global dynamics, one 8×8 grid for fine detail) addresses the stated failure mode and connects to the feature pyramid literature in detection.

### Implement SIGReg properly
The paper used standard Gaussian KL instead of the full Cramér-Wold SIGReg from LeWM. This is an easy follow-up since the LeWM codebase exists. If SIGReg prevents collapse more effectively it could measurably improve latent quality, particularly at aggressive compression rates where the current encoder shows non-monotonic RD behavior.

---

## 4. The Semantic Probing Framework Is the Most Transferable Contribution

LeWM-Eval is arguably more community-valuable than LeWM-VC itself — a reproducible, codec-agnostic semantic evaluation protocol that the field currently lacks.

| Action | Why |
|--------|-----|
| **Release LeWM-Eval as a standalone benchmark tool** | Positions the work as infrastructure for VCM community, not just one codec paper |
| **Extend probe to tracking accuracy** | Surveillance cares about trajectory consistency as much as per-frame class accuracy. SORT/ByteTrack evaluation strongly favors temporally-coherent latent representations — exactly what JEPA produces |
| **Test probe depth sensitivity** | Systematic ablation (1, 3, 5, 10 layers) characterizes information content in both representations and pre-empts a likely reviewer objection |

---

## 5. Positioning Strategy for Venue Submission

### Near-term: Workshop or specialized track
ICIP or VCM workshop. The current version is a strong submission at this tier.

### Top-venue gap analysis

| What's needed | Effort | Score impact |
|---|---|---|
| 10+ clip training set with error bars | Medium (compute, not ideas) | +1.5 pts |
| Learned codec baseline with probe eval | Medium (FVC or DCVC-DC) | +1.0 pt |
| Surprise gate demonstrated working | High (needs right dataset) | +0.5 pt |
| Statistical significance on probe results | Low (bootstrap, one afternoon) | +0.5 pt |
| Actual arithmetic coder measurement | Low (one operating point) | +0.3 pt |

- **First two items addressed:** ~7/10 → Accept at ICIP, Borderline-Accept at ICCV
- **All five items addressed:** Legitimate CVPR submission, especially if spatial cross-attention ablation adds something interesting

---

## 6. One Strategic Observation

The paper's thesis — that compression quality should shift from PSNR to task accuracy as video consumers shift from humans to machines — is correct, timely, and supported by the MPEG VCM process. The architectural demonstration is real.

**The risk** is that the paper gets filed away as "interesting but small-scale" and never gets the follow-up it deserves.

**The clearest path to avoiding that outcome** is to own the evaluation framework, not just the codec. If LeWM-Eval becomes the standard way to measure machine-oriented video codecs — the way BD-rate became standard for pixel codecs — then every future paper in this space cites this work regardless of whether LeWM-VC itself is the eventual winning architecture. That's a much stronger long-term position than winning a PSNR comparison.
