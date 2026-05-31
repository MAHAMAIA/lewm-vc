# LeWM-Eval Standardization Plan
## Making LeWM-Eval the BD-Rate of Machine-Oriented Video Compression

**Document version:** 1.0  
**Date:** 2026-05-16  
**Author:** Strategic advisory for MAHAMAIA Systems / LeWM-VC project  
**Status:** Working roadmap

---

## Preamble: Why the Evaluation Framework Is the Bigger Prize

BD-rate (Bjøntegaard Delta Rate, 2001) is a single-page ITU document that became one of the most cited tools in video compression research. Its author did not invent H.264. The person who defined how to measure codecs shaped the field's direction for two decades — because every paper that followed had to speak the same language to be taken seriously.

LeWM-VC is a good codec paper. LeWM-Eval is potentially a more important contribution. The machine-oriented video compression field (MPEG VCM, task-driven codecs, edge perception pipelines) currently has no agreed-upon evaluation standard. Every paper measures something different: some use detection mAP, some use classification accuracy, some use feature-level distortion, some use VMAF proxies. None are reproducible across labs. This is the gap LeWM-Eval can fill — if it is developed, positioned, and adopted deliberately.

This document is the exhaustive plan for doing that.

---

## Part 1: Diagnosis — What Does "Becoming a Standard" Actually Require?

Before building anything, it is worth being precise about what standardization means in practice and what has made past evaluation frameworks succeed or fail.

### 1.1 Anatomy of a Successful Benchmark

Examining frameworks that achieved field-wide adoption — ImageNet, COCO, BLEU, BD-rate, FID, MTEB — reveals five common properties:

**P1 — The benchmark measures something the field agrees matters but cannot currently measure well.**  
ImageNet succeeded because classification accuracy on a large held-out set was genuinely unsolved and clearly meaningful. BD-rate succeeded because "average bitrate savings across a rate-distortion curve" was the right question and nobody had a formula for it. LeWM-Eval's opportunity is identical: task accuracy at matched bitrate is clearly the right question for machine-oriented codecs, and no agreed measurement exists.

**P2 — The benchmark is cheaper to use than to ignore.**  
A benchmark that requires significant infrastructure investment gets ignored. One that ships as `pip install lewm-eval && lewm-eval run --codec my_codec --dataset meval-surveillance` gets adopted. Friction is the enemy of standardization.

**P3 — The benchmark has a leaderboard with enough participants to feel competitive.**  
A benchmark with one entry is a paper appendix. One with ten entries is a community resource. One with fifty is a standard. The path from one to fifty requires deliberate seeding, not organic growth alone.

**P4 — The benchmark is maintained and versioned.**  
Dead benchmarks get forked, fragmented, and eventually abandoned. A benchmark that updates its dataset, fixes bugs, and publishes version changelogs signals institutional commitment and discourages divergence.

**P5 — The benchmark has organizational backing beyond a single lab.**  
IEEE, ACM, MPEG, or a consortium of three or more named institutions dramatically increases longevity and perceived neutrality. A benchmark owned by one lab is a tool; one co-governed by several is infrastructure.

### 1.2 Where LeWM-Eval Currently Sits

Assessing the current state honestly:

| Property | Current Status | Gap |
|---|---|---|
| P1 — Measures the right thing | Yes — task accuracy at matched BPP is the right metric | Needs broader recognition |
| P2 — Low friction to use | Partial — code exists but CLI not yet abstracted | Major engineering work needed |
| P3 — Competitive leaderboard | No — one entry (LeWM-VC itself) | Seeding strategy required |
| P4 — Maintained and versioned | No formal versioning | Governance structure needed |
| P5 — Organizational backing | Single lab (MAHAMAIA) | Coalition building required |

The honest summary: LeWM-Eval has the right core idea and a working reference implementation. It needs a complete rebuild as a standalone tool, a coalition of institutional co-authors, and a deliberate community strategy. None of these are research problems — they are execution problems.

---

## Part 2: Technical Development Roadmap

### 2.1 Phase 0 — Extraction and Decoupling (Month 1–2)

The first and most important technical step is making LeWM-Eval completely independent of LeWM-VC. Currently the evaluation code is embedded in the codec repository. This must change.

**Deliverable: `lewm-eval` as a standalone Python package on PyPI.**

The package must:

- Accept any codec as a black-box function with the interface `encode(frames) -> bitstream` and `decode(bitstream) -> latent_or_frames`
- Support both latent-space codecs (like LeWM-VC, FVC) and pixel-reconstruction codecs (like x265, DCVC-DC)
- Handle the BPP normalization internally, including flagging when a codec reports estimated vs. actual bitrate
- Ship the teacher model management (download, cache, version-pin) automatically
- Produce a standardized JSON results file that can be uploaded to the leaderboard directly

**Package structure:**

```
lewm_eval/
├── __init__.py
├── cli.py                  # lewm-eval run / lewm-eval submit / lewm-eval compare
├── core/
│   ├── bitrate.py          # BPP computation, matching, estimation vs. actual flagging
│   ├── probing.py          # Probe training, evaluation, confidence intervals
│   ├── surprise.py         # Surprise metric computation
│   └── rdc.py              # Rate-distortion-accuracy curve generation
├── teachers/
│   ├── yolov5s.py
│   ├── yolov5su.py
│   └── registry.py         # Versioned teacher model registry
├── datasets/
│   ├── pevid_hd.py
│   ├── virat.py            # Planned
│   ├── meva.py             # Planned
│   └── registry.py
├── codecs/
│   ├── x265.py             # Reference wrapper
│   ├── lewm_vc.py          # Reference wrapper
│   └── base.py             # Abstract codec interface
└── results/
    ├── schema.py            # JSON results schema
    └── leaderboard.py       # Submission formatting
```

**Critical design principle:** The probe training must be reproducible from a fixed random seed with a fixed train/test split that is locked at dataset version time, not at evaluation time. This prevents the current problem where different papers use different splits and results are incomparable.

### 2.2 Phase 1 — Dataset Expansion (Month 2–5)

The single greatest technical weakness in the current evaluation is the two-clip training corpus. Fixing this is prerequisite to any credibility claim.

**Tier 1 — Surveillance domain (required for v1.0):**

| Dataset | Sequences | Resolution | Annotation Type | License |
|---|---|---|---|---|
| PEViD-HD (current) | All sequences (~20 clips) | 1080p | None (use teacher) | Academic |
| VIRAT Ground 2.0 | 25 hours | 720p/1080p | Activity, object | Public |
| MEVA | 144 hours | 1080p | Activity, vehicle, person | Public |
| CityFlow | 40+ cameras | 1080p | Vehicle tracking | Academic |
| UCF-Crime | 1,900 clips | Variable | Anomaly labels | Academic |

Each dataset integration requires: download scripts, standardized preprocessing pipeline (resize, normalize, frame extraction), teacher pseudo-label generation, fixed train/val/test splits with reproducible seeds, and documentation of any filtering applied.

**Tier 2 — Adjacent machine-perception domains (v2.0 target):**

- Autonomous driving: nuScenes camera streams, Waymo Open Dataset (perception relevance, different motion statistics)
- Industrial inspection: MVTEC Anomaly Detection video extension (different failure modes)
- Satellite/aerial: DOTA video sequences (different scale, motion, and semantic structure)

The inclusion of Tier 2 datasets transforms LeWM-Eval from a surveillance benchmark into a machine-perception compression benchmark, which is a substantially larger and more impactful framing.

**Versioning policy:**

Datasets must be versioned independently of the evaluation code. The leaderboard must track which dataset version a result was obtained on. A codec evaluated on `meval-surveillance-v1.0` is not comparable to one evaluated on `meval-surveillance-v2.0`. This is how COCO handles its updates and it is the right model.

### 2.3 Phase 2 — Metric Expansion (Month 3–6)

The current LeWM-Eval measures three things: BPP, PSNR, and probe class accuracy. This is insufficient for a field-defining benchmark. The following metrics must be added, in priority order:

**M1 — Tracking accuracy (highest priority).**  
Per-frame classification says nothing about temporal coherence, which is arguably the most important property of a surveillance codec. Adding a SORT or ByteTrack evaluation pipeline — tracking objects across a multi-second sequence using either latents or decoded frames as input — provides a metric that: (a) is directly relevant to deployment, (b) favors temporally coherent representations (which JEPA produces), and (c) is not easily gamed by per-frame metrics. Implementation: run ByteTrack on probe detections across 10-second clips, compute HOTA (Higher Order Tracking Accuracy) and IDF1.

**M2 — Rate-distortion-accuracy curves (high priority).**  
Currently the paper reports point estimates at one or two BPP values. A proper RDA curve — sweeping λ or CRF to produce accuracy as a function of BPP — is the machine-oriented equivalent of the RD curve and should be the primary visualization in every LeWM-Eval result. The area under this curve (analogous to BD-rate) should be the headline scalar metric.

**M3 — Latent invertibility score (medium priority).**  
For latent-space codecs, measuring how much information is lost vs. how much is retained for downstream tasks requires knowing the mutual information between the latent and the original frame. A practical proxy: train an inverter network to reconstruct frames from latents and measure reconstruction PSNR as an upper bound on recoverable information. This is distinct from the codec's own decoder PSNR and characterizes the latent space independently of reconstruction quality.

**M4 — Privacy leakage metric (medium priority).**  
The paper discusses privacy implications (Section 5.3) but does not measure them. A face re-identification probe — training a ReID network on latents vs. decoded frames and measuring rank-1 accuracy — quantifies the compression-as-anonymization effect claimed in the paper. This metric is unique to LeWM-Eval and no existing benchmark offers it. It directly addresses regulatory concerns (GDPR, CCPA) that make this framing commercially relevant.

**M5 — Surprise calibration score (lower priority, contingent on W3 fix).**  
Once the surprise gate is demonstrated working on appropriate data (UCF-Crime), add an Expected Calibration Error (ECE) metric for the surprise mechanism: how well does the surprise score predict the downstream task accuracy degradation? A well-calibrated surprise metric would allow practitioners to make bitrate allocation decisions without task-specific labels.

**M6 — Cross-domain transfer score (lower priority).**  
The UVG failure is currently reported as a qualitative limitation. Formalizing it as a metric — reporting the RDA curve on an out-of-domain dataset for every codec — turns a weakness into a systematic evaluation dimension. A codec that generalizes well scores higher; domain-specific codecs are not penalized if their in-domain performance justifies specialization.

### 2.4 Phase 3 — The Leaderboard (Month 4–8)

The leaderboard is not a website feature — it is the mechanism by which the benchmark becomes competitive and therefore useful.

**Technical implementation:**

The simplest viable leaderboard is a GitHub repository with a structured results directory and a GitHub Actions workflow that validates and renders results. This requires zero infrastructure beyond what already exists, has no hosting cost, and is maximally transparent (every result is a pull request with reviewable methodology).

Structure:
```
lewm-eval-leaderboard/
├── results/
│   ├── lewm-vc_v0.1_meval-surv-v1.0.json
│   ├── x265_crf28_meval-surv-v1.0.json
│   └── [submitted results]
├── validate.py              # CI validation script
├── render.py                # Generates leaderboard table
└── README.md                # Auto-generated leaderboard table
```

Each result JSON must contain:
- Codec name, version, paper citation
- Dataset name and version
- Exact hyperparameters used
- All metric values with confidence intervals
- Whether BPP is estimated (entropy lower bound) or measured (actual arithmetic coding)
- Hardware used, inference time
- Checkpoint hash for reproducibility

**Submission process:**

Authors submit results by opening a pull request. A CI pipeline validates the JSON schema, checks that BPP values are within plausible ranges, and verifies the checkpoint hash if a public checkpoint is provided. Human review is required for acceptance — this prevents gaming but must be fast (48-hour SLA target).

**Seeding the leaderboard (critical):**

A leaderboard with one entry is a personal tracking spreadsheet. The benchmark needs at minimum 5–8 entries at launch to feel like a community resource. Seeding strategy:

1. Run LeWM-Eval on every publicly available learned video codec that can be wrapped in the standard interface: x265 (baseline), x264, DCVC-DC (if weights are public), FVC, CompressAI factorized, CompressAI mean-scale hyperprior. This takes compute time but no collaboration agreements.
2. Reach out to 3–5 groups working on VCM or learned codecs and offer co-authorship on the benchmark paper in exchange for running their codec and contributing a result. This is the standard model for benchmark papers (COCO, ScanNet, etc.).
3. Offer to run LeWM-Eval on others' codecs in exchange for citing the benchmark. This is a service that positions MAHAMAIA as a neutral evaluator.

### 2.5 Phase 4 — Tooling and Ecosystem Integration (Month 6–12)

For a benchmark to survive beyond the paper that introduced it, it must integrate with the tools researchers already use.

**Integration targets:**

- **CompressAI:** The dominant Python library for learned image/video compression research. A `lewm-eval` plugin for CompressAI that evaluates any CompressAI model on the benchmark with one command would immediately reach every researcher using that library.
- **MMCompression / OpenMMLab:** Similar plugin or adapter.
- **HuggingFace Datasets:** Upload all LeWM-Eval datasets and pseudo-labels to HuggingFace Hub with a `datasets.load_dataset('mahamaia/meval-surveillance-v1')` interface. This reduces the friction of dataset access to a single line.
- **Papers With Code:** Register the benchmark on Papers With Code so that results automatically appear on the leaderboard when papers are submitted to arXiv. This is how most active benchmarks in ML maintain their competitive presence.

---

## Part 3: Publication Strategy

### 3.1 The Two-Paper Architecture

The path to standardization runs through two distinct publications with different audiences and purposes.

**Paper A — LeWM-VC v2 (codec paper, target: CVPR/ICCV 2027)**

This is the expanded version of the current preprint, addressing all weaknesses identified in the conference review:
- 10+ PEViD-HD sequences + VIRAT training data
- Learned codec baselines with probe evaluation
- Demonstrated surprise gating on UCF-Crime
- Statistical significance testing
- Spatial cross-attention predictor ablation
- Actual arithmetic coder measurement

This paper argues: JEPA-based latent prediction is a better temporal model for machine-oriented compression than motion vectors.

**Paper B — LeWM-Eval (benchmark paper, target: NeurIPS 2027 Datasets & Benchmarks track)**

This is a separate paper presenting LeWM-Eval as an independent contribution. It does not assume the reader cares about LeWM-VC — it argues that the field needs a standard evaluation framework and here is one. Key sections:
- Survey of existing evaluation practices in machine-oriented compression (showing fragmentation)
- Design principles of LeWM-Eval (why these metrics, why this dataset, why this probe architecture)
- Baseline results across 10+ codecs on 3+ datasets
- Analysis of which metrics correlate with deployment outcomes vs. which are misleading
- Governance and versioning policy

NeurIPS Datasets & Benchmarks is the correct venue because: (a) it explicitly values contribution type "establishing a new benchmark," (b) it has high visibility across the ML community beyond vision specialists, (c) it does not require the benchmark to be the best-performing system — only to be well-motivated and reproducible.

### 3.2 Workshop Strategy (Pre-Publication Community Building)

Before Paper B, the benchmark should appear at workshops to gather feedback, build awareness, and accumulate co-authors/participants. Target workshops:

**2026 (near-term):**
- MPEG VCM Open Workshop — This is the most important single venue. MPEG VCM participants are exactly the people who would adopt or standardize LeWM-Eval. Presenting there, even as a position paper, puts the benchmark in front of the standardization process.
- CVPR 2026 Workshop on Video Understanding and Compression / Workshop on Efficient Deep Learning for Computer Vision — Both attract the codec + vision intersection audience.
- ECCV 2026 Workshop on Video Coding for Machines — Direct target audience.

**2027 (pre-NeurIPS submission):**
- Present expanded results (v1.0 leaderboard with 10+ entries) at CVPR 2027 workshop to demonstrate community traction before the NeurIPS submission.

### 3.3 Positioning the Benchmark Paper

The benchmark paper must make a claim that is independent of LeWM-VC's performance. The right framing:

*"We show that existing evaluation practices for machine-oriented video compression are inconsistent, non-reproducible, and often measure the wrong thing. We introduce LeWM-Eval, a standardized framework that evaluates codecs on task accuracy at matched bitrate across multiple surveillance and machine-perception datasets. We run 12 codecs on LeWM-Eval and show that codec rankings differ substantially from PSNR-based rankings — with implications for which architectural choices actually matter for deployment."*

The finding that "PSNR ranking ≠ task accuracy ranking" is the key result. If this can be demonstrated across 10+ codecs on 3+ datasets, it is a significant empirical contribution that stands entirely independently of whether LeWM-VC is the best-performing codec.

---

## Part 4: Community and Coalition Strategy

### 4.1 The Co-Author Model

The fastest path to institutional legitimacy is co-authorship by researchers from 3–5 independent groups. This is how COCO, ScanNet, Waymo Open Dataset, and every major benchmark achieved credibility beyond the originating lab.

Target coalition structure for LeWM-Eval v1.0:
- **1 traditional codec group** (e.g., a HEVC/VVC contributor lab) — provides credibility that the benchmark is fair to non-neural codecs
- **1 learned codec group** (e.g., DCVC/DCVC-DC team at USTC or a comparable group) — provides credibility with the neural codec community
- **1 VCM/standards group** (someone involved in MPEG VCM) — provides direct path to standardization consideration
- **1 surveillance/application group** (an institution doing actual surveillance analytics) — provides credibility that the metrics matter for deployment

Coalition building approach:
1. Email authors of the top 5 most-cited papers in machine-oriented compression offering to run LeWM-Eval on their method and include their results in the benchmark paper with co-authorship
2. Post a "call for participation" on arXiv as a companion to the LeWM-VC preprint, explicitly inviting codec authors to contribute results
3. Present at MPEG VCM and explicitly request feedback from members on metric design — this positions the framework as community-developed rather than imposed

### 4.2 The Neutral Evaluator Position

One subtle but important strategic point: MAHAMAIA should position LeWM-Eval as a neutral evaluator even though it was created to evaluate LeWM-VC. This means:

- LeWM-VC must not rank first on the leaderboard by default. If it does, the benchmark looks self-serving. If other codecs outperform it on some metrics, that should be prominently reported and celebrated as evidence of the benchmark's fairness.
- The benchmark paper should have at least one finding where LeWM-VC performs worse than a baseline on some metric. The privacy leakage metric (M4) and cross-domain transfer score (M6) are natural candidates — LeWM-VC will likely underperform general-purpose codecs on cross-domain transfer, and that should be stated clearly.
- The governance structure (Part 5) should include a process for resolving disputes about result validity that does not give MAHAMAIA unilateral authority.

### 4.3 Engagement with MPEG VCM

The MPEG Video Coding for Machines standardization process is the single most important external process to engage with. If LeWM-Eval becomes part of the VCM evaluation methodology — even informally as a reference implementation — it achieves standardization through the most authoritative channel available.

Concrete engagement steps:
1. **Submit a contribution document to MPEG VCM** (ISO/IEC JTC1/SC29/WG2 or successor). MPEG accepts contributions from non-members in some cases; check current submission policy. The document should propose LeWM-Eval metrics as a complement to the existing VCM evaluation framework.
2. **Attend MPEG plenary meetings** as an observer. MPEG meetings are partially open to observers; presence signals commitment and allows informal relationship building with the key people shaping the standard.
3. **Align metric definitions with VCM terminology** wherever possible. If VCM uses specific names for "task accuracy" or "feature compression quality," adopt those names in LeWM-Eval to reduce the translation cost for VCM participants.
4. **Reference VCM working documents** in the benchmark paper and vice versa. This creates a citation relationship that VCM contributors will notice.

---

## Part 5: Governance Structure

Benchmarks fail without governance. The following structure balances openness with quality control.

### 5.1 Versioning Policy

LeWM-Eval must version both the evaluation code and the datasets independently.

**Evaluation code versions** (semantic versioning):
- Major version: breaking changes to metric definitions or probe architecture
- Minor version: new metrics, new datasets, new teacher models added
- Patch version: bug fixes, documentation, tooling improvements

**Dataset versions:**
- Named versions: `meval-surveillance-v1.0`, `meval-surveillance-v2.0`
- Results on different dataset versions are not comparable and must be tagged separately on the leaderboard
- Dataset versions are immutable once released — no silent updates

**Policy: once a result is on the leaderboard under a given dataset version, that result is permanent**, even if a bug is later found in the evaluation code. Bug fixes produce a new version; authors are invited (not required) to resubmit.

### 5.2 Result Validation Policy

Results must be:
- Accompanied by the result JSON in the specified schema
- Produced by code that is either (a) publicly available, or (b) verified by a LeWM-Eval maintainer who ran it independently
- BPP measurement method explicitly declared (estimated vs. actual)
- Checkpoint hash provided if a checkpoint is claimed to produce the result

Results that cannot be independently verified are tagged as "unverified" on the leaderboard. This preserves openness while signaling to readers which results have been checked.

### 5.3 Advisory Board

Establish a lightweight advisory board of 5–7 members from diverse institutions and backgrounds:
- At minimum one traditional codec expert (HEVC/VVC background)
- At minimum one machine learning / representation learning researcher
- At minimum one surveillance or edge-vision practitioner
- At minimum one VCM standards participant

The advisory board's role is: advising on metric design changes, arbitrating disputes about result validity, and providing public endorsement of the benchmark's fairness. Board members serve two-year terms. They do not have unilateral authority — decisions are made by maintainer consensus with board input.

---

## Part 6: Commercial and Regulatory Alignment

### 6.1 Why This Matters Beyond Academia

The surveillance codec market is not primarily driven by academic publication — it is driven by procurement decisions at municipalities, airports, retailers, and governments. These buyers respond to:
- Regulatory compliance (GDPR Article 25 "data protection by design," CCPA, AI Act)
- Integration with existing VMS (Video Management Software) vendors
- Total cost of ownership (compute, storage, bandwidth)

LeWM-Eval can speak directly to these buyers if it includes metrics they recognize:

**Privacy leakage** (M4 above) maps directly to GDPR Article 25 compliance arguments. A codec with lower face re-identification accuracy at matched task performance is demonstrably more privacy-preserving. This is a procurement differentiator.

**Bandwidth and storage cost** (BPP at matched task accuracy) maps directly to TCO calculations. A 62% bitrate reduction is $X million in storage costs at scale — this calculation should appear in the benchmark documentation.

**Edge deployment profile** (inference latency, memory footprint, NPU compatibility) maps to integration decisions. LeWM-Eval should include a standardized hardware efficiency section alongside accuracy metrics.

### 6.2 Regulatory Roadmap

Several regulations create mandatory disclosure requirements that LeWM-Eval metrics could satisfy:

- **EU AI Act (2024, enforcement 2026):** High-risk AI systems (surveillance) require technical documentation of capabilities and limitations. A LeWM-Eval result sheet — showing task accuracy, privacy leakage, and failure modes in a standardized format — is a natural compliance artifact.
- **NIST AI RMF:** US government contractors using AI for surveillance increasingly need to demonstrate performance on relevant tasks. A standardized benchmark result from a neutral framework is more defensible than internal testing.

The benchmark paper should include a section connecting LeWM-Eval metrics to these regulatory frameworks. This is unusual for a computer vision paper but directly relevant to the deployment context and will be noticed by the practitioner community.

---

## Part 7: Phased Timeline and Milestones

### Phase 0: Foundation (Months 1–3)
**Goal:** Standalone package exists; initial coalition formed

- [ ] Extract LeWM-Eval from LeWM-VC repo into standalone `lewm-eval` package
- [ ] Implement standard codec interface (`encode`, `decode`, `bitrate`)
- [ ] Add wrappers for x265, x264, CompressAI factorized, CompressAI mean-scale
- [ ] Fix random seed / train-test split to be dataset-version-locked
- [ ] Add bootstrap confidence intervals to all probe accuracy results
- [ ] Expand PEViD-HD to all available sequences (~20 clips)
- [ ] Upload to PyPI: `pip install lewm-eval`
- [ ] Create GitHub leaderboard repository with CI validation
- [ ] Contact 5 potential co-authors for benchmark paper; secure 2–3 commitments
- [ ] Submit position paper to MPEG VCM open workshop

**Success metric:** LeWM-Eval runs on 5+ codecs with a single command; leaderboard has 5+ entries

### Phase 1: Expansion (Months 4–6)
**Goal:** v1.0 release with multi-dataset support; leaderboard competitive

- [ ] Integrate VIRAT Ground 2.0 dataset
- [ ] Integrate UCF-Crime for surprise metric validation
- [ ] Add tracking accuracy metric (ByteTrack + HOTA)
- [ ] Add RDA curve generation and AUC-RDA scalar metric
- [ ] Add privacy leakage metric (ReID probe)
- [ ] Demonstrate surprise gating on UCF-Crime (fixes W3)
- [ ] Run LeWM-Eval on DCVC-DC or FVC (fixes W2)
- [ ] Leaderboard reaches 8+ entries
- [ ] Present at CVPR 2026 VCM workshop
- [ ] HuggingFace dataset upload for `meval-surveillance-v1.0`
- [ ] Papers With Code registration

**Success metric:** Leaderboard has 8+ codecs; three independent groups have run LeWM-Eval on their codec

### Phase 2: Validation (Months 7–10)
**Goal:** Benchmark paper submitted; LeWM-VC v2 submitted

- [ ] Add spatial cross-attention predictor to LeWM-VC (ablation)
- [ ] Train LeWM-VC on VIRAT + full PEViD-HD
- [ ] Add actual arithmetic coder measurement at one operating point
- [ ] Complete statistical significance testing across all tables
- [ ] Submit LeWM-VC v2 to CVPR/ICCV 2027
- [ ] Draft LeWM-Eval benchmark paper with co-authors
- [ ] Advisory board constituted (5 members)
- [ ] CompressAI plugin released
- [ ] Leaderboard reaches 12+ entries
- [ ] Community feedback incorporated into v1.1

**Success metric:** Both papers submitted; benchmark paper has 4+ institutional co-authors

### Phase 3: Standardization (Months 11–18)
**Goal:** Benchmark paper accepted; MPEG VCM engagement active

- [ ] NeurIPS 2027 D&B track submission
- [ ] `meval-surveillance-v2.0` released with MEVA and CityFlow
- [ ] Tier 2 datasets (autonomous driving, industrial) scoped
- [ ] Second MPEG VCM contribution submitted
- [ ] VMS vendor pilot evaluation using LeWM-Eval (commercial bridge)
- [ ] Leaderboard reaches 20+ entries
- [ ] First external lab runs LeWM-Eval without MAHAMAIA involvement

**Success metric:** Paper accepted; leaderboard self-sustaining; at least one MPEG VCM document references LeWM-Eval

---

## Part 8: Resource Requirements

### 8.1 Compute

The benchmark requires running evaluation across multiple codecs and datasets. Rough estimates:

| Task | Estimated Compute |
|---|---|
| Generating teacher pseudo-labels for VIRAT (25 hours at 30fps) | ~40 GPU-hours (YOLOv5su on T4) |
| Running LeWM-Eval on 1 codec × 1 dataset (100 clips) | ~8 GPU-hours |
| Seeding leaderboard with 8 codecs | ~64 GPU-hours |
| Training RDA curves for LeWM-VC v2 | ~80 GPU-hours |
| Total for Phase 0–1 | ~300–400 GPU-hours |

AMD MI300X credits (already used for training) are appropriate for this workload.

### 8.2 Engineering Time

The package extraction and CLI work (Phase 0) is the most time-intensive engineering task — roughly 4–6 weeks of focused engineering effort. The metric additions (Phase 1) are mostly well-defined computations (~2 weeks each). The leaderboard CI/CD is lightweight (~1 week).

### 8.3 Collaboration Overhead

Coalition building and co-author coordination adds meaningful overhead, primarily in communication and result validation. Budget approximately 20% of total project time for this — it is not optional and cannot be delegated to later.

---

## Part 9: Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| A larger lab (Google, Meta, ByteDance) launches a competing benchmark first | Medium | High | Move fast on Phase 0; get arXiv preprint for benchmark paper by Month 6 to establish priority |
| MPEG VCM adopts incompatible metric definitions | Medium | Medium | Engage early (Phase 0); design metrics to be VCM-compatible from the start |
| Leaderboard gaming (overfitting to specific probe or dataset) | Medium | Medium | Version datasets frequently; add held-out test sets not released publicly |
| Coalition co-authors drop out | Low-Medium | Medium | Get written commitments early; distribute result generation so no single co-author is blocking |
| Compute costs make adoption prohibitive | Low | High | Optimize the probe training to run on CPU; provide pre-computed pseudo-labels so adopters need minimal compute |
| LeWM-VC is outperformed on semantic metrics by a simpler baseline | Low | Low | This would be a good outcome — it validates the benchmark's fairness |

---

## Part 10: The Minimum Viable Version

If the full plan above is too ambitious for the current stage, here is the minimum version that still creates a credible foundation:

1. **Extract LeWM-Eval to a standalone repository** with a clean codec interface and `pip install`. This costs two weeks and costs nothing else.
2. **Run it on 5 codecs** (x265, x264, CompressAI factorized, CompressAI mean-scale, LeWM-VC). Report results in a companion arXiv note: "LeWM-Eval: Preliminary Results Across Five Codecs." No venue submission needed.
3. **Register on Papers With Code** immediately after the arXiv note. This puts the benchmark in front of the community passively.
4. **Email 10 codec paper authors** offering to run their codec on LeWM-Eval for free and include their results. Accept whoever responds.

This minimal version costs roughly 4 weeks of engineering and 2 weeks of outreach. It produces a leaderboard with 5–10 entries, an arXiv companion note, and the foundation for the full benchmark paper. It is executable immediately and establishes priority before any competing effort.

---

## Conclusion

The path from "evaluation framework in a codec paper appendix" to "field standard" is a 12–18 month project requiring equal parts engineering, publication, and community building. The technical work is the easiest part. The hardest part is the coalition — getting three independent groups to run your evaluation, endorse your metrics, and co-author your paper. That work starts with emails, not commits.

The core insight remains: whoever defines how the field measures machine-oriented compression will shape which architectural choices look good for the next decade. LeWM-Eval has a 12–18 month window before this space becomes crowded enough that a new entrant cannot establish priority. The plan above is the execution path for using that window.

---

*Document maintained by MAHAMAIA Systems. Version history tracked in the LeWM-Eval repository.*
