# MAHAMAIA Systems — Due Diligence Preparation

## What Investors Will Ask, What the Answers Are, and What Documents to Have Ready

**Document version:** 1.1 (customer-first reframe)  
**Date:** 2026-05-16  
**Use:** Prepare before any investor meeting. Internalize this. Do not read it in the meeting.

---

## Preamble: How Deep Tech Due Diligence Works

Consumer and SaaS due diligence focuses on metrics: MRR, CAC, churn, NPS. Deep tech infrastructure due diligence focuses on three questions in sequence:

1. **Is the technical insight real?** Can a technical partner verify that the claimed results are reproducible and the architecture is sound?
2. **Is the market large and accessible?** Is there a credible path from "this works in a lab" to "this generates revenue at scale"?
3. **Is this team the one that will execute it?** Do they understand the path? Are they honest about what they don't know?

A specific note: do not attempt to conceal the two-clip training dataset limitation. Every serious technical investor will find it. You naming it first, explaining what it means, and describing the plan to address it creates confidence.

**The strategic framing:** We build LeWM-VC, a codec that saves 62% storage at better detection accuracy than H.265. LeWM-Eval is the reproducible verification tool that ships with it. Standards follow adoption. Investors familiar with the Arena trajectory will recognize the playbook, but the product narrative comes first.

---

## Section 1: Technical Due Diligence Questions

**Q1: "How is this different from H.265, AV1, or VVC?"**

**The answer:** Those codecs optimize for pixel fidelity — what looks good to humans. LeWM-VC optimizes for what machine perception systems need: object location, class, motion trajectory. At matched bitrate, LeWM-VC preserves 7.2 pp more classification accuracy than H.265. The difference is architectural, not incremental.

**Q2: "How is this different from DCVC, CANF-VC, or other learned codecs?"**

**The answer:** Those are better H.265s — they learn the transforms and motion compensation but retain pixel reconstruction as the training objective. LeWM-VC replaces motion vectors entirely with a JEPA temporal predictor operating in latent space. The compressed representation is a semantic feature stream, not a pixel approximation. The evaluation framework (LeWM-Eval) ships with the codec and is codec-agnostic — any competitor can be benchmarked against us using the same methodology.

**Q3: "Is this a codec company or a benchmark company?"**

**The answer:** Codec company. LeWM-VC is the product. LeWM-Eval is the tool that proves it works and lets customers compare us against alternatives. But the codec is what we sell.

**Q4: "Isn't 256×256 too small for real surveillance?"**

**The answer:** Yes, for the current evaluation resolution. The architecture scales to any resolution that is a multiple of 16. The decoder's upsampling layers are hardcoded for 256×256 in the current proof of concept but are architecturally straightforward to extend. Higher-resolution evaluation on native 1920×1080 is a planned extension and will be demonstrated in the next version.

**Q5: "Two training clips is very small. How do we know this generalizes?"**

**The answer:** We don't yet — that is why dataset expansion is the highest priority technical action and is funded by the seed round. The ablation studies show the architecture is robust when training data is available. The domain specificity is a feature for the beachhead market (surveillance), not a limitation for production deployment. Dataset expansion to 20+ sequences and VIRAT Ground is expected Q3 2026.

**Q6: "The surprise metric doesn't work. Is that a problem?"**

**The answer:** It is a presentation issue in the current paper, not an architectural problem. The surprise metric is a diagnostic signal. It is not core to the codec's compression performance or the evaluation framework's validity. It will be corrected in the next paper version.

---

## Section 2: Business and Market Questions

**Q7: "Who is the customer, and why would they buy this?"**

**The answer:** Surveillance operators with 50+ cameras running machine analytics pipelines. They buy LeWM-VC because it saves 62% on storage costs while improving detection accuracy. A 500-camera deployment saves ~$120K/year at typical storage pricing. The ROI is measured in months — the payback period against storage and compute savings is shorter than the annual contract term. Primary buyer: CTO or VP Engineering at a smart city platform, airport security operator, or retail loss prevention network.

**Q8: "Why would an OEM license this instead of building their own?"**

**The answer:** Building a learned codec requires expertise in JEPA architectures, learned entropy coding, ViT encoder design, and video codec pipeline engineering — a rare combination. The OEMs (Hanwha, Axis, Bosch) are hardware manufacturers, not ML codec companies. By the time VCM standardization creates compliance pressure, LeWM-VC will have deployment data and benchmark results that no internal effort can quickly match.

**Q9: "$8/camera/month for a codec? How did you arrive at that?"**

**The answer:** Comparable to the per-camera cost of cloud video management SaaS (Verkada, Eagle Eye Networks charge $15–50/camera/month). At 62% storage savings and improved detection accuracy, the value proposition is clear against the current storage cost line item. The pricing will be refined with the first design partner.

**Q10: "Isn't the moat the evaluation framework, not the codec?"**

**The answer:** The near-term moat is deployment data and customer relationships — a codec trained on real surveillance feeds that no competitor can replicate without equivalent deployment scale. The evaluation framework becomes a moat over time as it gains adoption through customer usage. Standards follow adoption, not the reverse.

**Q11: "How does this relate to MPEG VCM?"**

**The answer:** VCM is a forcing function — it creates demand for machine-oriented evaluation, which accelerates adoption of the framework that ships with our codec. We are engaging the working group. Our engagement strategy targets academic VCM contributors (easiest warm intros), then frequent WG2 contributors, then working group leadership. See the VCM integration guide for details.

---

## Section 3: Team Questions

**Q12: "Why you? What's special about this team?"**

**The answer (be honest, not modest):**

The technical depth in this space — JEPA architectures, learned entropy coding, video codec design, semantic probing methodology — is rare. The combination of that depth with understanding of the market application (machine perception pipelines in surveillance) and the standards process (MPEG VCM) is rarer.

What I don't have yet is production engineering depth and enterprise sales experience. Those are the gaps the first non-founder hires address — an ML engineer and a VP Business Development — both budgeted in the seed round.

**Do not oversell. The investor knows you are at an early stage. The question is whether you are honest about your gaps and have a plan for them.**

**Q13: "What does success look like in 18 months with this round?"**

**The answer (be specific):**

- One paying design partner running LeWM-VC on deployment data with measurable storage savings and detection accuracy improvement.
- LeWM-VC v2 submitted to CVPR 2027 — expanded dataset, learned codec baselines, statistical significance testing.

- MPEG VCM contribution submitted and acknowledged.
- Team of 4–5 (VP BD, ML engineer, benchmark engineer).

---

## Section 4: Objection Handling

### The "Why won't Google/Qualcomm/NVIDIA build this?" objection

**The answer:**

- **Google** optimizes for human viewers (YouTube). An evaluation framework owned by Google is rejected by Microsoft, Amazon, and Qualcomm. The standard requires institutional neutrality.
- **Qualcomm** could build a machine-oriented codec, but they have no distribution into the surveillance OEM market — their customers would not trust a codec standard owned by a chip vendor their competitors use. LeWM-VC's neutrality is a feature, not a weakness.
- **NVIDIA** (Metropolis) could build one, but they are an integration partner, not a surveillance codec company. Their interest is selling GPUs.

The most dangerous competitive scenario is a large corporate entity contributing an evaluation framework to MPEG VCM before LeWM-Eval is established. Mitigation: the defense is customer relationships and deployment data, which create switching costs. LeWM-Eval emerges as a secondary defense.

### The "Two clips? That's not a dataset." objection

**Proactive address:** Name it on slide 8 (traction) or in the technical conversation before they find it. "Our current evaluation is on PEViD-HD surveillance video. We trained on two clips as a proof of concept — here is the plan to expand to 20+ clips and a second dataset, expected Q3 2026."

### The "Single founder risk" objection (DOES NOT APPLY)

**Proactive address:** "We are a two-founder team — Preetam Mukherjee (codec architecture, evaluation framework) and Soumyajit Mandal, Ph.D. MIT, 26 patents, custom ASIC design. The gap is not a co-founder — it is a first non-founder VP BD hire, budgeted in the seed round."

### The "No customers, no LOIs" objection

**Proactive address:** Name the design partner outreach status explicitly. "We have 10 organizations in the design partner pipeline. [Name] at [Company] is the most advanced — we expect a signed LOI by [date]." If you have no conversations in progress, start them before approaching investors.

---

## Section 5: Closing Statement

When an investor asks "why should I invest now?" — do not hedge.

**The answer:**

"The window for deploying the first production-grade machine-native codec is open. MPEG VCM is being formalized right now. The EU AI Act creates a compliance requirement that operators need to meet. Edge hardware can run transformer inference at commodity prices for the first time.

"LeWM-VC is real — 62% storage savings at better detection accuracy, 80+ fps on a T4, public checkpoints, reproducible results.

"The companies that deploy working codecs into production in 2026–27 will define the standard, because standards follow adoption, not the reverse. We are building the thing customers need. That is the investment."

Then stop talking. Let the investor sit with that.

**If they ask "who have you talked to at VCM?":** The working group is ISO/IEC JTC 1/SC 29/WG 2. Our entry strategy: academic VCM contributors at VCIP/ICIP are the easiest warm intros (search for "VCM core experiment" papers); frequent WG2 contributors include Lingling Wang, Xin Zhao, Wenjie Lu, Tie Liu; the convener is Sean McCarthy (Nokia). The highest-probability path is emailing the corresponding author of a recent VCM-related arXiv paper — they will forward to the right WG2 contact if the work is relevant.
