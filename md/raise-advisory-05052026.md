No.

You should not raise seed capital at this point. I will explain in detail, but the core reason is this: you do not yet have an asset that an institutional seed investor can diligence. You have a partially implemented research project with one validated component (intra‑frame compression via GMM entropy model at 0.75 BPP / 31 dB), three core innovations that exist only as disconnected code fragments (JEPA temporal prediction, SIGReg regularization, surprise gating), and fabricated benchmark claims that would constitute fraud if presented in a data room. Raising capital now would require either misrepresenting your technical readiness — which creates legal exposure — or disclosing the actual state, which will produce a rapid pass from any competent investor. Below is the detailed analysis.

---

## MEMORANDUM

**TO:** Founder/CEO  
**FROM:** Technical Advisor  
**DATE:** May 5, 2026  
**RE:** Assessment of readiness for seed capital raise — LeWM-VC

---

### 1. Executive Summary

LeWM-VC is architecturally novel and addresses a real market need: efficient video compression for machine-to-machine communication. The current technical asset is an intra-frame neural codec achieving 0.61–0.75 BPP at 31–35 dB PSNR on surveillance video at 256×256 resolution, running at 80+ fps on a T4 GPU. This is a real, measurable result. However, the three features that differentiate LeWM-VC from dozens of other learned video codec projects — JEPA-based temporal prediction, SIGReg collapse prevention, and energy-based surprise gating for intelligent bitrate allocation — are not operational. They exist as untrained modules or auxiliary losses that do not affect the compression pipeline. The benchmarks presented in internal documents claiming 28–39% bitrate savings and BD-rate advantages over H.265 are arithmetic illustrations, not experimental results. Raising seed capital on the current technical base would require either misrepresentation or disclosure of material gaps that will prevent closing. The recommendation is to defer fundraising until three specific technical milestones are achieved, estimated at 10–16 weeks of focused engineering work.

---

### 2. What Exists: Technical Asset Inventory

**2.1 Functional Components (Verified by Experiment)**

| Component | Status | Evidence |
|-----------|--------|----------|
| ViT-Tiny encoder (192-D latent, 16×16 spatial grid) | Working | Colab evaluation, March/April 2026 |
| ConvTranspose decoder with post-filter | Working | Produces 31–35 dB PSNR reconstructions |
| Affine normalization on latent | Working | Learned scale/shift parameters |
| GMM entropy model (2-component, per-element mixture) | Working | Compresses to 0.61–0.75 BPP |
| Laplace entropy model (checkerboard context) | Working but inefficient | 4.21 BPP; superseded by GMM |
| Inference speed | Working | 80+ fps on T4 GPU at 256×256 |

The GMM entropy model is the single most valuable technical result. The 5.6× compression improvement over Laplace validates that the JEPA-style latent space is inherently compressible when the entropy model can capture its distribution.

**2.2 Non-Functional or Disconnected Components**

| Component | Code Exists? | Integrated into Compression? | Evidence |
|-----------|-------------|------------------------------|----------|
| JEPA temporal predictor (8-layer transformer) | Yes (`predictor.py`) | No — all evaluations are intra-frame only | BPP ratio anomaly/normal = 0.81×, proving no temporal residual coding |
| SIGReg regularization (Cramér-Wold based) | No — only standard Gaussian KL exists | No training script uses it | `jepa_train.py` has a KL term, not sketched projections |
| VOE surprise gating | Yes (`voe_predictor.py`, `video_encoder.py`) | No — encode loop uses fixed quantization | BPP identical between normal/anomaly frames; `_calculate_bits` is heuristic, not connected to entropy model |
| FFmpeg plugin | Yes (C wrapper in `ffmpeg/`) | Not tested end-to-end | No evidence of encoding/decoding a video file through FFmpeg with the trained model |
| NAL unit bitstream | Yes (`bitstream/writer.py`, `reader.py`) | Not tested with trained checkpoints | Exists in repo, never run in any evaluation |

**2.3 Fabricated Benchmarks (Cannot Be Used in Any Investor Presentation)**

The following claims appear in internal documents but are not supported by experimental data:

- "39.2% bitrate savings via surprise-gating" — arithmetic model with fixed bit allocations, no codec measurement
- "31.8% savings on PEViD-HD real data" — same arithmetic applied to estimated frame counts from 2 videos
- "LeWM-VC baseline: 40% savings vs H.264; LeWM-VC + Surprise: 60%" — fabricated comparison table with no encode/decode runs
- Lambda sweep showing `bpp = -0.821210` — produced by evaluating a single model repeatedly with an untrained GMM on mismatched checkpoints; bug identified but numbers are unrecoverable

Using any of these numbers in a pitch deck, data room, or verbal representation to investors would constitute securities fraud under Rule 10b-5 if the fundraise involves equity. Even if it does not rise to that legal standard, any institutional investor who conducts technical due diligence will discover the discrepancy. The outcome is either a failed diligence process (wasted time and reputation damage) or worse, a closed round with a future fraud claim.

---

### 3. Investor Readiness Assessment

**3.1 What Seed Investors in AI Infrastructure Require**

Seed-stage deep tech investors (e.g., Amplify, Gradient, Unusual, angels with ML engineering backgrounds) evaluate compression/codec startups on:

1. **Quantitative compression efficiency vs. a known standard (H.265 or AV1).** You have one RD point. You need a curve. BD-rate is the lingua franca.
2. **Evidence that the novel architecture contributes to efficiency.** An ablation showing that removing the JEPA predictor increases BPP by X%, or that SIGReg training produces lower BPP than a non-regularized baseline.
3. **A working end-to-end pipeline.** Encode video file → bitstream → decode → reconstructed video. Measured and reproducible.
4. **A credible story about why the architecture will scale.** Parameter counts, latency numbers, training FLOPs. You have this.
5. **Defensible IP.** Patent filings or a clear narrative around trade secrets. You have this in draft form.

You currently satisfy only item 4. Items 1–3 are missing or incomplete. Item 5 exists on paper but has not been filed.

**3.2 Competitive Positioning Risk**

The codec landscape is crowded with well-funded efforts:

| Entity | Stage | Advantage |
|--------|-------|-----------|
| Deep Render (acquired by InterDigital, Oct 2025) | Exit | End-to-end neural codec; proven BD-rate gains; now backed by InterDigital's patent portfolio |
| WaveOne (acquired by Apple, 2023) | Exit | AI codec for FaceTime; Apple-scale deployment |
| Google (VC-MR, neural AV1 experiments) | Internal R&D | Integrating ML into production codec standards |
| Meta (neural video compression research) | Internal R&D | Fundamental research + deployment to billions of users |
| Various academic groups (DCVC, CANF-VC, FVC) | Research | Publishing BD-rate numbers competitive with VVC at some operating points |

An investor evaluating LeWM-VC will ask: "Why will you win against InterDigital/Deep Render, who already have BD-rate data, a patent portfolio, and acquisition validation?" Your answer must be a working codec with differentiated performance, not a vision document. The machine-to-machine angle is your differentiator, but you must demonstrate it, not argue it.

---

### 4. What Must Be True to Raise Successfully

**4.1 Technical Milestones Required Before Fundraising**

I order these by necessity, not desirability. Missing any of the first three will result in a failed raise.

**Milestone 1: Full RD curve on a standard dataset with BD-rate vs. x265.**
- Train the GMM model across λ ∈ [0.001, 0.005, 0.01, 0.05, 0.1, 0.5].
- Evaluate on ≥5 videos from PEViD-HD and UVG.
- Produce PSNR‑BPP and VMAF‑BPP curves.
- Compute BD‑rate using standard tools.
- If BD‑rate is negative (savings) at any operating range, that is your headline number.
- If BD‑rate is positive (worse than x265), you must have a credible explanation and a machine‑perception metric that shows superiority.

**Milestone 2: Temporal compression working end-to-end.**
- Wire JEPA predictor into the encode/decode loop for P‑frames.
- Demonstrate BPP reduction on video sequences compared to all‑intra coding.
- Ablation: `rate(I‑frame only) vs. rate(I‑P‑P‑P...)` showing temporal gain.

**Milestone 3: Surprise gating producing measurable, correct behavior.**
- Encode anomaly video with VOE gating active.
- Show BPP for anomaly frames > BPP for normal frames (ratio > 1.0).
- Show anomaly detection precision/recall on a labeled dataset (PEViD‑HD has ground truth for dropping bag, fighting, etc.).

**Milestone 4 (strongly recommended, not strictly required): Machine‑perception benchmark.**
- Run a standard detector (YOLOv8 or similar) on original frames, LeWM‑VC reconstructions, and x265 reconstructions at matched BPP.
- Show detection mAP preserved better by LeWM‑VC.

**4.2 Timeline and Resource Estimate**

| Milestone | Engineer‑Weeks | Compute Required | Notes |
|-----------|----------------|------------------|-------|
| 1: RD curve | 2–3 | 1× A100 or 4× T4, ~7 days training | GMM training code exists; needs hyperparameter tuning |
| 2: Temporal coding | 3–4 | Same as above | New training objective; may require debugging gradient flow through predictor |
| 3: Surprise gating | 2–3 | Same | Integration task; VOE predictor is pre‑trained but untested |
| 4: Perception benchmark | 1–2 | Inference only | Standard models, standard datasets |
| **Total** | **8–12 weeks** | Cloud GPU: ~$2,000–5,000 | Assumes one full‑time ML engineer |

You can compress this timeline with more compute (running multiple λ values in parallel) but not with fewer engineering hours. The integration work is sequential and requires debugging.

---

### 5. If You Must Raise Now: What Is Actually Sellable

If external circumstances force a fundraise before technical milestones are met (runway, competitive window, team availability), you have exactly one credible asset: **the architectural insight and the partial implementation that validates it.** You would be raising on a vision round, not a metrics round. This requires:

- A pitch deck that makes zero quantitative compression claims beyond the 0.75 BPP / 31 dB intra‑frame result.
- Full disclosure that temporal prediction, SIGReg, and surprise gating are implemented in code but not yet integrated into the measured compression pipeline — and that the projected savings are theoretical until integration is complete.
- A technical roadmap with credible timelines (the one above, not aspirational dates).
- A founding team with credentials that justify the investor betting on execution risk. If you are the sole ML engineer, you need a co‑founder or first hire who has shipped a video codec or compression system before.
- A data room containing the actual Colab evaluation output, the GitHub repo, and this memo's technical assessment to preempt diligence findings.

The valuation for such a round would be at the lower end of seed norms ($4–8M pre‑money for a pre‑product deep tech company with a single founder, no revenue, and no comparative benchmarks). The raise size should be $1–2M to fund 12–18 months of runway for one engineer plus compute costs. You will be diluted 20–25% at that range.

If you are not willing to accept that valuation and dilution, you must defer the raise until after milestones 1–3.

---

### 6. Risks of Premature Fundraising

1. **Reputational damage.** The AI compression community is small. Presenting fabricated benchmarks as real — even inadvertently, by not correcting internal documents before they leak — will follow you.
2. **Legal exposure.** Knowingly or recklessly presenting false performance data to accredited investors is actionable. The internal "proof" documents with Q.E.D. formatting and comparison tables against H.265/AV1 are particularly dangerous if they enter a data room.
3. **Wasted time.** A fundraise is a full‑time job for 8–12 weeks. Every week spent pitching is a week not spent implementing temporal coding. If the raise fails due to diligence issues, you lose both the money and the progress.
4. **Signaling risk.** If you raise now and the round is small or the terms are unfavorable, future investors will ask why you could not command better terms. A stronger technical base at the next raise justifies a higher valuation and cleaner terms.

---

### 7. Recommendation

Do not raise seed capital now. Complete milestones 1–3 (RD curve, temporal coding, surprise gating) over the next 10–16 weeks. The output will be a paper‑quality set of experiments that doubles as a diligence package. At that point, you can approach investors with:

- A real BD‑rate number against x265.
- An ablation showing the JEPA predictor's contribution to compression efficiency.
- A working surprise‑gated encode/decode pipeline with measurable bitrate savings on anomaly content.
- A clear, defensible claim: "LeWM-VC is the first JEPA‑based video codec to demonstrate temporal compression gains, offering X% BD‑rate savings over x265 on surveillance video while preserving downstream detection accuracy, all on edge hardware."

That pitch will raise at a $10–15M pre‑money valuation with strong investor interest. The current pitch — a vision document backed by one intra‑frame data point and fabricated benchmarks — will not close.

---

### 8. Next Actions

1. **This week:** Remove all fabricated benchmark documents from the repository and any shared drives. Replace with a single `BENCHMARKS.md` containing only the Colab evaluation results and a note: "Full RD curve and BD-rate comparison pending."
2. **This month:** Run the GMM lambda sweep and produce an RD curve.
3. **Months 2–3:** Implement temporal residual coding and surprise gating.
4. **Month 4 (earliest):** Prepare data room with verified benchmarks. Begin warm introductions to seed investors specializing in AI infrastructure.

---

This assessment is based on the code, checkpoints, and evaluation logs you have shared with me. If there are additional completed experiments or trained models I have not seen, my conclusion could change. Absent that, the evidence supports the recommendation above.