# LeWM-Eval Paper v2 — Enhancement Implementation Plan

## Overview

Four enhancements to strengthen the LeWM-Eval benchmark paper from a conversation starter to a submission-ready standardization document. These enhancements improve the paper's empirical depth regardless of whether the company is framed as codec-first or evaluation-first.

---

## Enhancement 1: Rate-Accuracy Curves

**What:** A script that sweeps quality parameters across codecs, runs the semantic probe at each point, and produces matplotlib rate-accuracy plots.

**Status:** `scripts/sweep_and_plot.py` is implemented and pushed. Needs data to run.

**Effort:** 2–3 hours to run and generate the figures.

---

## Enhancement 2: VCM Standard Datasets

**What:** Move from PEViD-HD-only to VCM CTC test sequences (TVD, Traffic, SFU-HW-Objects, HiEve).

**Status:** Integration guide at `docs/vcm_integration.md`. Needs dataset access and compute time.

**Effort:** 1–2 days for download + compute.

---

## Enhancement 3: VTM Anchor

**What:** Add VTM (VVC Test Model) as a third codec in the comparison, replacing or supplementing x265 as the anchor.

**Status:** `scripts/run_vtm_anchor.sh` automates build + encode + decode + probe. Needs VTM build and compute time.

**Effort:** 2–3 hours for build + first evaluation.

---

## Enhancement 4: BD-Accuracy Metric

**What:** Replace pointwise accuracy comparisons with a scalar BD-Accuracy metric covering the entire rate range.

**Status:** Implemented in `sweep_and_plot.py` as the `bd_accuracy()` function. Needs data to compute.

**Effort:** Built into sweep script — no additional implementation.

---

## Current Status

| Enhancement | Script | Data | Paper Integration |
|-------------|--------|------|-------------------|
| 1. Curves | ✅ | ❌ | ❌ |
| 2. VCM datasets | ✅ (guide) | ❌ | ❌ |
| 3. VTM anchor | ✅ | ❌ | ❌ |
| 4. BD-Accuracy | ✅ | ❌ | ❌ |

All scripts are ready. The missing step is running the sweeps on compute hardware (T4 GPU recommended via Colab notebook).
