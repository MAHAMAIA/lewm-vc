# Investor Perspective: Commercializing MAHAMAIA

> **Date:** 2026-05-16 (revised)  
> **Strategic Lens:** Codec-first, adoption drives standardization.

---

## The Core Thesis

MAHAMAIA builds LeWM-VC, a machine-native video codec for surveillance infrastructure. It saves 62% storage at better detection accuracy than H.265. Customers deploy it to reduce storage costs, improve detection accuracy, and document compliance.

LeWM-Eval is the evaluation framework that ships with the codec. It is codec-agnostic — customers verify our numbers independently and compare us against any alternative.

---

## What You Actually Have Right Now

### Reads as genuinely strong through the product lens:

- **A working codec (LeWM-VC) with public checkpoints and reproducible results.** Not a paper — a deployable system. 80+ fps on T4, 14.7M parameters, 1.2 GB GPU memory.
- **A concrete customer value proposition.** "A 500-camera deployment saves $120K/year at better detection accuracy." This maps to an existing budget line (storage costs) for a known buyer (surveillance operators).
- **Timing that is real.** MPEG VCM formalization, EU AI Act enforcement, edge compute cost crossover — all converging in 2026.
- **Open-source posture that builds trust.** Public code, checkpoint hashes, Docker container. Engineering discipline that transfers to production.
- **AMD MI300X compute relationship** — institutional validation.

### What will kill a raise if unaddressed:

- **No design partner / LOI.** Every serious investor will ask "who has bought this?" The answer today is "no one." Fix before close.
- **Two training clips.** Signals the gap between "research result" and "product." Fixable with compute runway.
- **No VP BD hire.** Technical founding team is strong. Commercial hire is the first non-founder role.
- ****

---

## The Narrative

The pitch is not "we own the measurement standard." It is:

> *"Every surveillance camera today compresses with H.265, which was built for human eyes. The consumer is now a GPU object detector. LeWM-VC is the first codec built for that GPU — and it saves 62% storage while improving detection accuracy. The evaluation framework that proves it works ships with the codec. Our customers use it to verify our numbers and compare us against alternatives. But the codec is what we sell today."*

---

## What Needs to Be True Before You Approach Investors

### Next 30 days:
1. **Get one design partner.** A surveillance operator running LeWM-VC on deployment data with a signed LOI. This is the single most important milestone.
2. **Put LeWM-Eval on PyPI** as the standalone evaluation tool that ships with the codec.
3. **Publish a technical blog post** — "LeWM-VC: A Machine-Native Codec for Surveillance" — framing the product, not the benchmark.

### Next 60 days:
4. **VP BD hired** or final-stage with named candidate.
5. **Full PEViD-HD training complete.** Results with error bars across 10+ sequences.
6. **Provisional patent filed** on JEPA temporal prediction for video compression.

### Next 90 days:

8. **LeWM-Eval methodology submitted as MPEG VCM working document.**

---

## Which Investors to Target

**Tier 1 — Deep tech infrastructure:** a16z (infrastructure team), Felicis, Lightspeed, Lux Capital, Radical Ventures.
**Tier 2 — Corporate strategics:** Qualcomm Ventures, NVIDIA NVentures, Bosch Ventures, Hanwha, AMD Ventures.
**Tier 3 — Security / surveillance specialists:** Outsider VC, security-focused family offices.

### How to approach:
- Publish your way in. The arXiv paper + a technical blog post gets read by technical partners at funds.
- Conference presence at CVPR, ICCV, MPEG VCM workshop.
- Use the AMD relationship for intros to AMD Ventures.
- The design partner is also the intro — ask your LOI partner for VC introductions.

---

## The Business Model

Three revenue streams, sequenced:

| Stream | Description | Timeline |
|--------|-------------|----------|
| **SaaS / Edge SDK** | Per-camera/month for surveillance operators. $5–15/camera/month. 500-camera deployment = $48K ARR. | 12–18 months |
| **Codec SDK Licensing** | Per-device royalty to camera OEMs. Activated by VCM standardization. | 18–36 months |
| **Benchmark Certification** | Paid LeWM-Eval certification for codec vendors. High margin, downstream of adoption. | 24–48 months |

---

## The One Thing

The most important milestone before closing the round is a signed design partner LOI. Everything else (MPEG engagement, academic citations) is secondary to a named organization running LeWM-VC on deployment data with a contract path.

The pitch that raises a massive seed:

> *"The surveillance industry is being forced to rebuild its entire compute stack for machine perception. We have the only codec built for that stack — 62% storage savings at better detection accuracy. One design partner is deploying it. Here is the 18-month plan to get to revenue."*
