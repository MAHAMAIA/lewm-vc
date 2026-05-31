# MAHAMAIA Systems — Seed Round Pitch Deck Structure

## Slide 1 — The Cover

"MAHAMAIA builds LeWM-VC, a machine-native video codec for surveillance infrastructure."

---

## Slide 2 — The Problem

H.265 optimizes for what looks good to humans. GPU object detectors are now the primary consumer of surveillance video. A 100-camera deployment generates more footage than a human team can review. The entire codec stack was built for the wrong consumer.

---

## Slide 3 — Market Timing

Three forcing functions align in 2026: MPEG VCM formalization, EU AI Act enforcement, edge compute cost crossover. The companies that deploy working codecs into production in 2026–27 will define the standard, because standards follow adoption, not the reverse.

---

## Slide 4 — The Solution

LeWM-VC compresses surveillance video to what machine perception systems need — 62% less storage at better accuracy than H.265. Architecture: Camera → LeWM-VC → Perception Pipeline. LeWM-Eval ships alongside the codec as the verification tool that proves performance and enables fair comparison against alternatives.

---

## Slide 5 — Results

- **62%** bitrate reduction vs. all-intra coding
- **+7.2 pp** class accuracy advantage over H.265 at matched bitrate
- **80+ fps** on NVIDIA T4 GPU

Customer framing: "A 500-camera deployment saves ~$120K/year in storage costs at better detection accuracy."

---

## Slide 6 — How It Works

ViT encoder → latent grid (192×16×16) → JEPA temporal predictor → residual coding → GMM entropy model. No hand-crafted motion vectors. 14.7M parameters. 12 KB per frame at inference.

---

## Slide 7 — Traction

Working codec with public checkpoints, reproducible results, AMD MI300X compute relationship. Gaps honestly reported: no revenue, no LOI, no VP BD hire yet. The most important milestone before closing is a signed design partner LOI — a named organization running LeWM-VC on deployment data with a contract path.

---

## Slide 8 — The Moat

LeWM-Eval ships with the codec and is codec-agnostic. Customers verify our numbers independently and compare us against alternatives. Deployments create benchmark data. Data improves the codec. The codec improves faster than any competitor's because no one else has equivalent deployment data. That data position is the moat.

---

## Slide 9 — Market Size

Bottom-up TAM. Beachhead: 2 million addressable cameras in GDPR-regulated markets at $50/camera/year = $100M ARR. Adjacent markets (autonomous vehicles, industrial inspection, logistics) represent a 10x multiplier.

---

## Slide 10 — Business Model

- **Stream 1 — SaaS / Edge SDK (12–18 months):** Per-camera-per-month subscription for surveillance operators. Primary near-term revenue engine.
- **Stream 2 — Codec SDK Licensing (18–36 months):** Per-device royalty to camera OEMs. Activated by VCM standardization.
- **Stream 3 — Benchmark Certification (24–48 months):** Paid LeWM-Eval certification. High margin, downstream of adoption.

---

## Slide 11 — Go-to-Market

Design partner motion. Target: surveillance operators with 50+ cameras in GDPR markets. Co-development at reduced cost in exchange for deployment data and a referenceable case study.

---

## Slide 12 — Team

Two founders: Preetam Mukherjee (codec architecture, evaluation methodology) and Soumyajit Mandal, Ph.D. MIT (custom ASIC design, 26 patents, Brookhaven National Laboratory). Covers full technical stack. VP BD hire is the first non-founder hire, budgeted in seed.

---

## Slide 13 — The Ask

$3–5M seed round. 18-month milestones: first design partner LOI, LeWM-VC v2 with multi-dataset training, MPEG VCM contribution acknowledged. Use of funds: 40% team, 35% data and compute, 25% operations.

---

## Slide 14 — Closing

"Surveillance operators are paying to store pixels no one watches and running analytics on degraded video. LeWM-VC fixes both problems. We are building the codec that machine perception infrastructure runs on. That is the investment."

---

## Appendix

A1 — Technical architecture  
A2 — Full results tables  
A3 — LeWM-Eval methodology  
A4 — Competitive landscape  
A5 — Regulatory landscape  
A6 — Financial model  
A7 — IP strategy
