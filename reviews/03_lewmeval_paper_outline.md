# LeWM-Eval — Benchmark Paper Outline
## NeurIPS 2027 Datasets & Benchmarks Track Submission

**Document version:** 1.1  
**Date:** 2026-05-16  
**Target venue:** NeurIPS 2027 Datasets & Benchmarks track  
**Working title:** "LeWM-Eval: A Reproducible Benchmark for Machine-Oriented Video Compression"

---

## Preamble: Relationship to the Codec Paper

The LeWM-VC paper argues: JEPA-based latent prediction is a better temporal model for machine-oriented compression than motion vectors. LeWM-VC is the product.

The LeWM-Eval paper argues: the field needs a standard evaluation framework, here is one, and here is evidence that existing evaluation practice produces misleading rankings. LeWM-Eval is the tool that ships with the codec, and this paper establishes its methodology independently.

The benchmark paper must be written so that a researcher who has never heard of LeWM-VC finds it valuable. The central empirical finding — PSNR-based codec rankings diverge systematically from task-accuracy-based rankings — is the result that will drive citations. The paper's contribution is the framework, and the reference implementation happens to be LeWM-VC.

---

## Paper Structure

**Abstract:** The field lacks a standardized way to measure task accuracy at matched bitrate for video compression. LeWM-Eval fills this gap with a codec-agnostic semantic probing methodology aligned with MPEG VCM CTC. We demonstrate that codec rankings based on PSNR diverge systematically from rankings based on task accuracy — with rank inversions occurring at efficient operating points. LeWM-Eval is available as an open-source framework.

**1. Introduction:** The evaluation gap, fragmentation problem, contribution statement.

**2. The LeWM-Eval Framework:** Five-step protocol (bitrate matching, frame decoding, semantic probing, teacher calibration, metric reporting). VCM CTC alignment.

**3. Empirical Validation:** Two-codec comparison (LeWM-VC vs x265) on PEViD-HD. Probe accuracy table at two operating points.

**4. Rank Inversion: PSNR vs Task Accuracy:** The central finding — the two metrics disagree at the efficient operating point. PSNR says x265 is better; task accuracy says LeWM-VC is better.

**5. Related Work:** Traditional evaluation, machine-oriented compression.

**6. Call for Community Adoption:** Links to both repos, public leaderboard plan.

---

## Key Relationships

- The codec paper (LeWM-VC) is about the architecture. It is the product.
- The benchmark paper (LeWM-Eval) is about the measurement methodology. It is infrastructure that ships with the codec.
- Both reference each other. The benchmark paper uses LeWM-VC as a validation example. The codec paper references the benchmark paper for evaluation methodology.
