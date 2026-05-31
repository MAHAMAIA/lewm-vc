# Sentinel Productization — Consolidated Plan

Synthesizes existing plans with strategic corrections from May 2026 review. Replaces `per-Mbps-saved` pricing with per-Gateway/month, adds dual-layer SVC as P0, focuses exclusively on remote industrial wedge.

---

## Strategic Foundation

### Target Wedge: Remote Industrial (Not Urban Surveillance)

Sentinel is built for one specific problem: **high satellite/VSAT bandwidth costs at remote industrial sites.** Not urban surveillance, not smart cities, not retail.

The value prop only holds where:
- Bandwidth costs $3,000–$10,000 per Mbps per month (satellite/VSAT)
- Camera counts are in the dozens, not thousands
- AI monitoring is already the standard (analytics on every stream)
- Forensic video must be preserved for legal/compliance

**Urban surveillance sites have cheap fiber — Sentinel saves nothing there.**

### Business Model Path

| Phase | Model | Timeline |
|-------|-------|----------|
| Now | **Hardware bundle** — Gateway appliance at $X/mo per unit | Immediate |
| 12-18 mo | **IP licensing** — License the codec to VMS vendors, camera OEMs, chip companies | After patent filings + standards alignment |
| 3-5 yr | **Acquisition** — Deep Render / InterDigital-style exit to a larger codec or chip company | Patent portfolio + VCM standards body position |

**Do not pursue:** SaaS/cloud metering, per-Mbps-saved billing, multi-tenant dashboard, urban integrator channel.

---

## Product Definition

| Product | What it is | Status |
|---------|-----------|--------|
| **Sentinel** (the codec) | LeWM-VC learned video codec — Docker container, CLI, API | Research prototype; needs packaging |
| **Sentinel Edge Gateway** (the hardware) | Jetson Orin NX/AGX appliance pre-loaded with Sentinel codec | Build guide written; no unit built yet |

Both are the same software — the Gateway is just the Jetson form factor for remote sites.

---

## Phase 1: Design Partner Ready (Right Now)

**Goal:** Someone can run Sentinel on their remote industrial footage this week.

### What exists today
- [x] Working encoder/decoder with pretrained checkpoints (`lewmvc-stream/`)
- [x] Flask API server (`src/server.py`) + Python client
- [x] Dockerfile + docker-compose for x86 with CUDA
- [x] Build guide for Jetson Orin NX/AGX deployment (note: Orin Nano lacks NVENC — NX or AGX required)
- [x] Website with clear product positioning
- [x] Design partner program page with contact flow
- [x] Current training run (`85e9d5`) shows balanced loss and no entropy collapse — base codec expected to be functional within days

### What's missing to ship

| Priority | Item | Details |
|----------|------|---------|
| **P0** | **Dual-layer SVC (Base Layer + Enhancement Layer)** | The forensic/legal claim depends on this. Base layer streams semantic latents continuously. Enhancement layer stores deterministically on local NVMe. Fuse on demand for human-viewable reconstruction. Without this, Sentinel is just another learned codec with no legal moat. Training script at `scripts/train_svc.py` — ready to run after base codec finishes. |
| P0 | **Docker image published** | Build and push `mahamaia/sentinel:latest` to GHCR or Docker Hub with bundled checkpoint |
| P0 | **CLI tool** | `sentinel encode input.mp4 output.lewm` and `sentinel decode output.lewm frames/` — single entry point |
| P1 | **Legal admissibility documentation** | Formal specification of deterministic residual enhancement layer and chain-of-custody best practices. Needed for forensic/legal claims. |
| P1 | **Multi-stream** | Handle 4+ camera feeds on a single Orin NX |
| P1 | **Systemd service** | Auto-start on boot, health checks, log rotation (partially specified in build guide) |
| P2 | **OTA updates** | Mechanism to push model updates to field devices. Infrastructure, not feature — design with rollback and code-signing from day one. |

---

## Phase 2: First Design Partner (Weeks 1-8)

### Outreach
- **Target**: 20 remote industrial operators (offshore energy, pipeline networks, island infrastructure, maritime)
- **Not target**: Urban integrators (Convergint, Johnson Controls), smart cities, retail — these sites have cheap fiber
- **Channel**: LinkedIn, industry events (OTC, SPE), warm intros through advisors
- **Pitch**: 8-week trial, pre-configured Gateway shipped to site, $0 upfront
- **Goal**: 5 conversations → 2-3 data shares → 1 signed LOI

### Onboarding Flow
1. Partner shares 2-3 clips (1 min each, typical remote site scenes)
2. Run full comparison: Sentinel vs H.265 at multiple bitrates
3. 30-min results review call
4. If metrics look good: ship Gateway (Orin NX), start pilot on real satellite link

### Hardware Requirement
- **Minimum**: Jetson Orin NX 16GB (~$600 module + carrier)
- **Not Orin Nano** — lacks NVENC, can't do real-time software H.265 at 1080p (3-8 fps)
- Target BOM at Q1: ~$900-1,000 per unit (module + carrier + NVMe + enclosure + PSU)
- Q100 target: ~$700-800 with ODM pricing on the module
- **Prioritize VSAT-only sites** (offshore, pipelines) over Starlink-served sites, as bandwidth savings are larger on VSAT ($3k-10k/Mbps/mo vs Starlink ~$500/Mbps/mo)

### Success Criteria
- Measurable bandwidth savings on a real satellite/VSAT link (not synthetic)
- Partner can operate Gateway without daily support
- Signed LOI with path to paid contract

---

## Phase 3: Production (Months 3-9)

### Product
- [ ] **Multi-camera** — verified with 8+ cameras per Gateway across diverse remote sites
- [ ] **VMS integrations** — RTSP output only. Sentinel appears as a standard RTSP camera. No custom Milestone/Genetec drivers pre-PMF.
- [ ] **Pricing finalized** — **per Gateway/month only.** Metered billing and per-Mbps-saved models are unauditable and add unnecessary infrastructure. Fixed monthly covers hardware lease + software license.
- [ ] **Single Gateway tier** — one product, one price. Differentiate by lease term (monthly vs annual).

### Business
- [ ] First paid customer (convert from design partner)
- [ ] Second design partner onboarded
- [ ] Pricing page live on website
- [ ] Case study published with partner (anonymized if needed)

### Engineering
- [ ] Train on expanded dataset (VIRAT + full PEViD-HD)
- [ ] Release v0.2 checkpoint with published benchmarks
- [ ] Edge optimization: TensorRT export, INT8 quantization
- [ ] CI/CD for Docker builds

### IP
- [ ] Provisional patent filed on dual-layer SVC + JEPA predictor
- [ ] LeWM-Eval open-sourced only after patent filing (not before)
- [ ] Engagement with MPEG VCM for standards alignment

---

## Key Decisions (from May 2026)

1. **Wedge**: Remote industrial with satellite backhaul only. Urban surveillance is out of scope — cheap fiber makes the value prop negligible.
2. **Product**: Sentinel = codec, Sentinel Edge Gateway = Jetson Orin NX appliance. Same software, different form factor.
3. **Claims**: Normalized to ranges (50-65% reduction, 1-4 month payback, $15-40K+ monthly savings) — defensible, not cherry-picked.
4. **Tagline**: "A machine-vision codec" — not "The first." Acknowledges MPEG VCM (2019+) prior art.
5. **Legal language**: "Supports chain-of-custody... can help meet FRE 901." Not "legally admissible." Dual-layer SVC must be built before this claim can be made credibly.
6. **Pricing**: Per Gateway/month. No metered billing, no per-Mbps-saved. One tier, one price.
7. **IP Licensing**: Endgame is licensing to VMS vendors, camera OEMs, or chip companies. Hardware bundle is the near-term delivery mechanism.
8. **Evaluation toolkit**: Open-sourced only after patent filings. Otherwise it arms competitors with our benchmarking methodology for free.
9. **Surprise-gated bit allocation** — differentiator from JPEG AI/MPEG VCM. Uses VOE predictor to allocate more bits to anomalous frames. To be implemented post-SVC.
10. **Legal strategy** — pursue civil/industrial use cases first (lower burden of proof). Criminal admissibility requires a test case that has not occurred. Document the deterministic residual enhancement layer as part of the evidence chain.

---

## Hard No's (What We Are NOT Doing)

- ❌ SaaS dashboard with per-camera billing
- ❌ Per-Mbps-saved pricing
- ❌ Urban surveillance / smart city / retail
- ❌ Custom Milestone/Genetec C++ drivers before first paid customer
- ❌ Orin Nano — use NX or AGX
- ❌ Open-sourcing LeWM-Eval before patent filings
- ❌ Claiming "JEPA is a proven video codec" without published benchmarks

---

## Reference Documents

| File | Content |
|------|---------|
| `reviews/11_productization_plan.md` | 6-week sprint: training data, Docker, CLI, evaluation pipeline (needs updating per above) |
| `reviews/08_100day_plan.md` | Post-seed execution: customer, research, packaging, BD, team |
| `reviews/10_customer_acquisition_plan.md` | Target list, outreach process, LOI template, objection handling (over-indexed on urban integrators — use remote industrial instead) |
| `docs/ROADMAP.md` | Broader 3-6 month acquisition positioning roadmap |
| `gateway-build-guide.md` | Jetson Orin NX hardware BOM, Docker build, systemd service |
| `yc-sprint-plan.md` | 10-week YC batch sprint: pilots, benchmarks, Demo Day prep |
| `reviews/SentinelBusinessViabilityAnalysis.md` | Business viability analysis (recommends IP licensing + remote industrial wedge — this plan adopts that) |
