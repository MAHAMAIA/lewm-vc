# YC Batch Sprint Plan — Sentinel Edge Gateway

## Goal
From acceptance to Demo Day: ship a production-ready edge video codec with 2-3 live design partners and a demonstrable path to revenue.

---

## Week 1-2: Foundation

### Codec Productionization
- [ ] **Dockerize the full pipeline** — `sentinel encode`, `sentinel decode`, `sentinel stream` CLI commands
- [ ] **End-to-end bitstream validation** — verify `bitstream/writer.py` + `reader.py` produce identical encode-decode cycles on 50+ test clips
- [ ] **Arithmetic coding integration** — replace learned entropy rate estimates with actual arithmetic-coded bitstreams
- [ ] **Quantization mode selector** — `--quality` flag (low/medium/high) maps to different `num_levels` and `step_size`

### Hardware
- [ ] **Jetson Orin setup** — flash Orin NX/Nano with JetPack, install Docker runtime
- [ ] **Baseline benchmark** — measure encode/decode fps, power draw, memory usage on Orin with 4 camera streams
- [ ] **RTSP ingestion** — verify Sentinel can read from standard RTSP camera feeds

### Design Partners
- [ ] **Outreach list** — 20 target operators (offshore, pipeline, island), sourced via LinkedIn + industry contacts
- [ ] **Trial agreement** — simple 1-page pilot MOU (not a full legal contract)
- [ ] **Initial calls** — 5 conversations to understand deployment environments

---

## Week 3-4: First Pilot

### Pilot Deployment
- [ ] **Ship first Gateway** — pre-configured Orin to first design partner site
- [ ] **Remote monitoring** — basic telemetry (uptime, throughput, encode stats)
- [ ] **Weekly checkpoint reviews** — compare Sentinel vs H.265 on their footage
- [ ] **Iterate on hardware** — fix any power/thermal/connectivity issues discovered in the field

### Codec Improvements
- [ ] **Hot-swap model reload** — update encoder/predictor weights without restart
- [ ] **Multi-stream support** — handle 4+ camera feeds on single Orin
- [ ] **Failure modes** — handle network dropout, power loss, disk full scenarios gracefully

### Metrics
- [ ] **Publish reproducible benchmarks** — on UVG and MCL-JCV test sets vs H.265
- [ ] **Detection accuracy sweep** — measure YOLO/nanodet mAP at multiple bitrate points
- [ ] **Create benchmark page** — live-updating dashboard on sentinel-web

---

## Week 5-6: Scale to 3 Pilots

### Deployments
- [ ] **Pilot 2** — different vertical (e.g. pipeline vs offshore)
- [ ] **Pilot 3** — different geography (e.g. island vs arctic)
- [ ] **Feedback loop** — structured weekly call with each partner
- [ ] **Hardware iteration** — revise enclosure, cooling, mounting based on field feedback

### Product
- [ ] **Fleet dashboard MVP** — web UI showing all deployed Gateways, per-stream stats, alerts
- [ ] **OTA update mechanism** — push model updates and config changes to field devices
- [ ] **Alerting** — SNS/Slack webhook for device offline, disk near-full, encode errors

### Business
- [ ] **Pricing finalized** — $X/mo per Gateway based on pilot learnings
- [ ] **Customer references** — collect quotes, case study data from pilots (with permission)
- [ ] **First LOI** — letter of intent from at least 1 pilot for paid deployment post-trial

---

## Week 7-8: Demo Day Prep

### Demo
- [ ] **Live demo** — real-time encode/decode comparison on stage: Sentinel vs H.265 side-by-side
- [ ] **ROI calculator** — interactive web tool: enter your Mbps × cost → see savings
- [ ] **Hardware demo** — physical Sentinel Edge Gateway on demo table, running from battery

### Pitch
- [ ] **Demo Day deck** — 10 slides: Problem → Solution → Architecture → Traction → Team → Ask
- [ ] **Executive summary** — 1-page PDF leave-behind with key numbers
- [ ] **Video** — 90-second product demo video (screen recording + hardware footage)

### Fundraising
- [ ] **Target investor list** — 50+ angels/funds (deep tech, infrastructure, climate, defense)
- [ ] **Data room** — benchmarks, pilot results, team bios, market sizing, financial model
- [ ] **Warm intros** — leverage YC network for intros to target investors

---

## Week 9-10: Launch

### Public Launch
- [ ] **Hacker News post** — technical deep-dive with benchmarks
- [ ] **arXiv paper** — release the LeWM-VC architecture and evaluation
- [ ] **GitHub release** — v0.1.0 tagged with pretrained weights, Docker image, eval scripts

### Post-Demo Day
- [ ] **Investor meetings** — 30+ meetings from Demo Day intros
- [ ] **Pilot → Paid conversion** — close first paid customers
- [ ] **Hiring** — post first job descriptions (ML engineer, embedded systems engineer)

---

## Key Milestones

| Milestone | Target | Verification |
|-----------|--------|-------------|
| First Gateway shipped | Week 3 | Deployed at partner site |
| 3 live pilots | Week 6 | All partners sending data |
| Public benchmarks | Week 6 | Published on website + GitHub |
| Demo Day ready | Week 8 | Pitch deck + live demo rehearsed |
| First LOI | Week 8 | Signed letter from pilot |
| v0.1.0 release | Week 10 | GitHub + Docker Hub |

---

## Team Roles

| Person | Role | Focus |
|--------|------|-------|
| Preetam | CEO/ML | Codec, benchmarks, demos, fundraising |
| Soumyajit | CTO/Hardware | Jetson validation, edge deployment, ASIC roadmap |

## Hiring (Post-Demo Day)

- ML Engineer — compression/vision experience, PyTorch, ONNX export
- Embedded Systems Engineer — Jetson, Yocto, camera drivers, Docker
- Customer Engineer — first sales hire, remote infrastructure domain experience

---

## Questions to Answer Before Demo Day

1. What is the right pricing model? Per-Gateway, per-camera, per-Mbps saved?
2. Do we sell direct or through VMS integrators (Milestone, Genetec partners)?
3. Is the target customer offshore energy, or is there a faster path in another vertical?
4. What's the hardware BOM cost at low volume (1-10 units) vs scale (100+)?
5. Do we manufacture the Gateway ourselves or partner with an ODM?
