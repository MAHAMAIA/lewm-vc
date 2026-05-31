# MAHAMAIA Systems — Master Strategy Document
## LeWM-VC · LeWM-Eval · Seed Round · Codec-First Adoption Model

**Document version:** 1.2 (customer-first reframe)  
**Date:** 2026-05-16  
**Scope:** Product strategy · Benchmark infrastructure · Seed fundraising

---

> **How to use this document.**
> Single source of truth for three tracks: the codec (LeWM-VC, the product), the evaluation framework (LeWM-Eval, ships with the codec), and the company (MAHAMAIA Systems). The strategic priority: **LeWM-VC is the product. LeWM-Eval is the tool that proves it works.**

---

## Part 1: Honest State of Play

### 1.1 What Is Real

- **A working codec (LeWM-VC)** with public checkpoints, Docker container, and exact reproduction scripts. 62% bitrate reduction, 7.2 pp accuracy advantage over H.265. Real numbers on real hardware at 80+ fps on a T4. 14.7M parameters, 1.2 GB GPU memory.
- **LeWM-Eval** ships with the codec and is codec-agnostic. Any competitor can be benchmarked against LeWM-VC using the same methodology.
- **Correct market timing.** MPEG VCM, EU AI Act, edge GPU cost crossover — all converging in 2026.
- **Strong open-source posture.** Public code, checkpoint hashes, Docker container, exact CLIs.
- **AMD MI300X compute partnership** — institutional validation.

### 1.2 What Is Not Yet Real

| Gap | Severity | Close Strategy |
|-----|----------|----------------|
| **No design partner / LOI** | Critical | Close first customer before fundraise close |
| **Two-clip training corpus** | Major | Expand to full PEViD-HD + VIRAT, Q3 2026 |
| **No VP BD hire** | Major | First non-founder hire, budgeted in seed |
| **No product — just a codec library** | Major | The codec IS the product. Package for SaaS deployment. |
| **No learned codec baselines** | Moderate | Add as community grows |
| **Surprise mechanism never demonstrated** | Low | Demote to diagnostic tool; not core to product |

---

## Part 2: Strategic Priority

### From (Evaluation-Centric):
- LeWM-Eval is the primary product
- LeWM-VC is the reference implementation
- Monetize through evaluation-as-a-service

### To (Codec-First):
- **LeWM-VC is the product.** Customers buy a codec that saves storage and improves detection accuracy.
- **LeWM-Eval is the verification tool that ships with it.** It provides reproducible verification that customers trust.
- **Adoption builds the moat.** As LeWM-VC is deployed, the codec improves through real-world data. Competitors cannot match this without equivalent deployment.

### Customer Value Equation
"A 500-camera deployment using LeWM-VC saves approximately $120K/year in storage costs at $8/camera/month while improving detection accuracy by 7.2 pp over H.265."

---

## Part 3: LeWM-VC Product Roadmap

### Phase 1: Proof-of-Concept (Existing)
- Working codec, public checkpoints, reproducible results
- Two-clip training corpus
- arXiv paper

### Phase 2: Production Validation (Months 1–6)
- Expand to full PEViD-HD (20+ sequences)
- First design partner running on deployment data
- Package LeWM-VC for edge deployment (Jetson, Ryzen AI)
- **Gate:** First design partner LOI signed

### Phase 3: Scale (Months 6–12)
- Multi-dataset training (VIRAT Ground, MEVA)

- Benchmark paper showing rank-inversion finding
- **Gate:** LeWM-Eval methodology cited in VCM discussion

---

## Part 4: Seed Fundraising Strategy

### The Narrative
"We build the codec that machine perception infrastructure runs on. LeWM-VC saves 62% storage at better detection accuracy than H.265. LeWM-Eval is the reproducible verification tool that ships with it."

### Traction Metrics to Hit Before First Meeting
| Metric | Current | Target |
|--------|---------|--------|
| Design partner LOI | 0 | 1+ |
| Codecs benchmarked against LeWM-VC | 0 | 5+ |
| External benchmark runs | 0 | 3+ |
| MPEG VCM engagement | None | Observer status + working document |

### Investor Target List
- **Tier 1:** a16z (infrastructure), Felicis, Lightspeed, Lux Capital
- **Tier 2:** Qualcomm Ventures, NVIDIA NVentures, Bosch Ventures, AMD Ventures
- **Tier 3:** Radical Ventures, Threshold Ventures

---

## Part 5: Key Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| No design partner | High | Seed capital funds BD hire; founders doing outreach until then |
| MPEG VCM doesn't adopt LeWM-Eval | Moderate | De facto standard via customer adoption doesn't require committee approval |
| Single founder risk | Moderate | Two-founder team (Preetam + Soumyajit); VP BD hire budgeted |
| Codec performance doesn't scale | Low | Domain specificity is a feature for beachhead |
| Surprise mechanism never works | Low | Demote to diagnostic; not core to product |

---

## Part 6: 18-Month Timeline

**Months 1–3 (Customer Validation)**
- First design partner LOI signed
- Full PEViD-HD training completed

- VP BD hired

**Months 4–6 (Scale)**
- LeWM-VC v2 with multi-dataset training
- 3+ independent external benchmark runs
- MPEG VCM working document submitted

**Months 7–12 (Standardization)**
- LeWM-Eval methodology cited in VCM core experiments
- First paying SaaS customer
- Benchmark paper at NeurIPS D&B

**Months 13–18 (Expansion)**
- Second design partner signed
- LeWM-VC v3 with edge deployment optimizations
- Series A preparation begins

---

## Part 7: Exit Scenarios

| Scenario | Timeline | Probability |
|----------|----------|-------------|
| **Infrastructure capture:** Codec becomes standard; EaaS revenue scales | 5–7 years | 20% |
| **Acquisition by Qualcomm/NVIDIA:** Codec IP acquired for hardware integration | 3–5 years | 30% |
| **Acquisition by VMS platform (Milestone/Genetec):** Technology acquired for compliance product | 4–6 years | 25% |
| **Modest independent company:** Sustainable revenue, not category-defining | Indefinite | 25% |
