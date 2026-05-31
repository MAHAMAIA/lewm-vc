# MAHAMAIA Systems — 100-Day Action Plan
## Post-Seed Close Execution Roadmap

**Plan version:** 1.1 (customer-first reframe)  
**Start date:** Day 1 after seed close  
**End date:** Day 100

---

## Executive Summary

The 100-day period is defined by five parallel workstreams:

1. **Customer (Highest Priority):** Design partner LOI signed. This is the single most important milestone — a named organization running LeWM-VC on deployment data with a contract path. Nothing else matters as much.
2. **Research (LeWM-VC v2):** Dataset expansion, learned codec baselines, statistical validation
3. **LeWM-VC Packaging:** Packaging for edge deployment, Docker image, external codec comparisons
4. **Business Development:** VP BD hired, design partner pipeline built
5. **Team & Governance:** ML engineer onboarded, advisory board constituted

### Hard Gates at Day 100

- [ ] **Customer gate:** First design partner LOI signed (a surveillance operator running LeWM-VC on deployment data)
- [ ] **Data gate:** Dataset expansion to 10+ PEViD-HD sequences complete; results with error bars
- [ ] **Standards gate:** LeWM-Eval methodology submitted as MPEG VCM working document

---

## Workstream 1: Customer (Highest Priority)

This workstream is the highest priority because revenue and market validation depend on it. Do not defer customer outreach for research "completeness."

### Days 1–14
- [ ] Identify 15 target organizations: surveillance operators with 50+ cameras in GDPR markets
- [ ] Draft design partner LOI template and technical brief
- [ ] VP BD outreach begins (even without hire, founding team runs initial outreach)

### Days 15–45
- [ ] First 5 outreach emails sent
- [ ] 2–3 technical calls completed
- [ ] LOI in negotiation with 1 lead prospect

### Days 46–100
- [ ] First design partner LOI signed
- [ ] Onboarding call with design partner: provide LeWM-VC evaluation on their sample footage

**Gate to Phase 2:** Signed LOI from a named organization with contract path.

---

## Workstream 2: Research (LeWM-VC v2)

### Days 1–14
- Download and preprocess full PEViD-HD corpus (~20 clips)
- Begin training on expanded dataset at 3 λ values
- Contact VIRAT Ground maintainers for dataset access

### Days 15–50
- Run LeWM-Eval on x265 (5 CRF points), x264 (5 points), one learned codec
- Train probes on each codec result; generate confidence intervals via bootstrap
- Rerun headline tables with mean ± std across 10+ sequences

### Days 51–75
- Statistical validation: which gaps are >95% CI?
- LeWM-VC v2 methods section for updated paper

### Days 76–100
- Provisional patent filing (Soumyajit leads)
- Hardware scoping: ASIC design space for entropy coder and ViT encoder

---

## Workstream 3: LeWM-VC Packaging and Distribution

### Days 1–20
- Package LeWM-VC for edge deployment (Jetson, Ryzen AI)
- Implement reference wrappers: x265, x264, CompressAI for comparison
- Set up Docker image for reproducible deployment

### Days 20–40
- Internal validation across design partner sample footage
- Performance profiling on target edge hardware

### Days 60–80
- Design partner onboarding package: evaluation kit, documentation, support
- First external codec comparison published (other learned codec vs LeWM-VC)

### Days 80–100
- Register project on Papers With Code
- Publish LeWM-VC performance benchmarks across expanded dataset

---

## Workstream 4: Business Development

### Days 1–20
- VP BD onboarding interviews; target hire Day 25–35
- Outreach materials drafted

### Days 20–50
- First batch outreach (5 targets)
- First calls completed

### Days 50–80
- LOI negotiation with lead prospect
- Second batch outreach (5 targets)

### Days 85–100
- Design partner engaged and evaluating LeWM-VC on their data

---

## Workstream 5: Team Building

### Days 1–10
- Job descriptions posted for VP BD and ML engineer
- Screening begins

### Days 10–35
- Offers extended and accepted
- Onboarding

### Days 35–100
- Advisory board recruitment (target: 3 committed by Day 100)

---

## Success Narrative at Day 100

"We have one design partner running LeWM-VC on deployment data. The codec is validated across 10+ sequences with error bars. The evaluation framework is public and running. The team is in place. The window to establish the standard through deployment is open."

---

## Post-Day-100 Preparation

- LeWM-VC v2 submission to CVPR 2027 or ICIP 2026
- Second design partner outreach batch
- Series A narrative refinement
