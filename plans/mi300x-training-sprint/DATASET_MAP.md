# Sentinel GTM — Targeted Dataset Mapping

Prioritized by Tier 1 accounts and waves.

## Tier 1 Immediate Priorities (Fast Wins — Months 1-6)

Highest-ROI targets. Public datasets give strong coverage for **vessel detection/tracking** in real maritime conditions, but **FPSO/offshore platform specifics** are the clear gap.

| Client | Focus | Best Matching Public Datasets | Why It Fits | Priority |
|--------|-------|-------------------------------|-------------|----------|
| **1. SBM Offshore** (FPSO & Offshore Production) | Large floating structures, support vessels, offshore ops | **Shuttle Tanker / FPSO / FSO Dataset** (IEEE DataPort — 580 annotated images) + **SeaDronesSee** | Directly targets FPSO/shuttle tanker visuals (rare publicly). SeaDronesSee adds UAV/open-water context. | **Highest urgency** — public data is limited. Prioritize SBM data-sharing agreement immediately for real footage. |
| **2. Maran Tankers** (Oil Tankers) | Large tanker vessels | **Singapore Maritime Dataset (SMD)** + **ABOships / ABOships-PLUS** | SMD has "Vessel-ship" + larger commercial classes in busy waters; ABOships includes cargoship-like vessels with scale/illumination variation. | **Top bootstrap choice** — onboard + onshore views match operational camera setups. |
| **3. Equinor Wind** (Offshore Wind) | Wind turbines at sea, installation/maintenance vessels | **SeaDronesSee** (primary) + SMD/ABOships | UAV perspective excels for wide-area offshore wind farm surveillance and vessel detection in open water. | Strong complement to vessel-focused sets. |
| **4. Pacific Basin** (Dry Bulk) & **5. FMG Fortescue** (Iron Ore / Bulk Carriers) | Bulk carriers & large cargo vessels | **SMD** + **ABOships** | Good coverage of larger "vessel-ship"/cargoship classes under varied conditions (haze, rain, different times of day). | Excellent for bulk/container-adjacent shipping. |

## Waves 2-4 / Broader Targets (Platform & Scale)

- **Container Shipping** (EPS Eastern Pacific, MSC): Covered well by SMD (vessel-ship classes) + ABOships.
- **Petrobras, Equinor Upstream, Transocean** (Offshore Oil/Gas & Drilling): Same FPSO/offshore gap as SBM → lean on the IEEE FPSO dataset as bridge + push customer footage.
- Integrations (Cognite, Orca AI, Captain's Eye): These datasets help validate maritime context for their workflows.

## Recommended Dataset Stack (Bootstrap + Adapter Training)

Start here this week (easy downloads, ready formats on Roboflow/Kaggle mirrors):

### 1. Singapore Maritime Dataset (SMD) — **#1 recommendation overall**
- 81 videos (40 onshore visible, 11 onboard visible, 30 NIR).
- Singapore waters, real conditions: pre-dawn → night, haze, rain.
- Annotations for object detection + tracking + horizon.
- Classes: Boat, Vessel-ship, Speed boat, Ferry, Sail boat, etc. (larger commercial vessels well-represented).
- Onboard views are especially valuable for ship-mounted or similar camera perspectives.
- Citation: D. K. Prasad et al., "Video Processing from Electro-optical Sensors for Object Detection and Tracking in Maritime Environment: A Survey," *IEEE Trans. Intelligent Transportation Systems*, 18(8), 1993-2016, 2017.
- **Download:**
  - Ground truth description: [Google Drive](https://drive.google.com/file/d/0B10RxHxW3I92NjRjZnN1bjVjelk/view)
  - Visible On-Shore (40 videos, 3 GB): [Google Drive](https://drive.google.com/open?id=1HnHyQzhzzDlYh15y9_K1mNZX3grlSDMM)
  - Visible On-Board (11 videos, 768 MB): [Google Drive](https://drive.google.com/file/d/0B43_rYxEgelVb2VFaXB4cE56RW8/view)
  - Near-IR On-Shore (30 videos, 1.51 GB): [Google Drive](https://drive.google.com/file/d/13wKWzHqkDQHMHjfuUWjgzwhTrnMoEE8P/view)
- Camera: Canon 70D, 1080×1920, Canon EF 70-300mm f/4-5.6 IS USM lens.
- Conditions: pre-sunrise, sunrise, mid-day, afternoon, evening, post-sunset, haze, rain (Jul 2015 - May 2016).

### 2. ABOships / ABOships-PLUS
- ~9,880 images extracted from videos (inshore to offshore transition).
- Captured from moving watercraft in Finnish archipelago.
- 9+ vessel classes (boat, cargoship, cruiseship, ferry, motorboat, sailboat, etc.) + seamarks/misc.
- Strong diversity in background, lighting, occlusion, scale — great for robust adapter training.
- COCO format available in PLUS version.

### 3. SeaDronesSee (Object Detection v2 + Tracking videos)
- UAV/drone perspective over open water — ideal for offshore wind and wide-area maritime surveillance.
- Humans, boats, vessels in realistic SAR-like scenarios.
- Complements the vessel-heavy SMD/ABOships perfectly.

### Niche but high-value for SBM #1
- **Shuttle Tanker, FPSO and FSO Image Dataset** (IEEE DataPort) — 580 annotated images specifically of FPSOs, shuttle tankers, and oil rigs (Pascal VOC with bow/mid/stern labels). Small but directly on-target. Access via free IEEE account.

## Strategic Fit

- **Immediate fix**: Download SMD + ABOships + SeaDronesSee today. Credible domain footage for quick adapter experiments (hours on one GPU) and early mAP validation on real maritime scenes (not just COCO proxies).
- **Architectural fix (adapter)**: Train lightweight adapter on this combined stack → learns to project varied maritime feature distributions into the compressor's canonical space. Per-client fine-tuning becomes overnight.
- **Data problem & moat**: Public datasets are excellent bootstrap/credibility builders, but the real unlock (especially for SBM FPSO, offshore drilling, wind) is **customer footage**. Use publics to de-risk and demonstrate value → then productize the "upload sample → train adapter overnight → deploy" onboarding flow.

## Sequencing Recommendation

- **This week**: Confirm target backbones + download/train on SMD + ABOships first (broadest coverage for most Tier 1 shipping clients). Add SeaDronesSee for wind/offshore angle.
- **Next**: Layer in the IEEE FPSO set + any early SBM pilot footage.
- **Q3 / Pre-seed**: Show <10% mAP delta on these domain datasets + customer samples.
