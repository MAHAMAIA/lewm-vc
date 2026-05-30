# LeWM-VC: Thesis Validation

## The Central Claim

> Latent-space predictive video coding (JEPA + entropy-encoded residuals) can match or beat x265 at machine-perception tasks (detection, tracking) at lower bitrate.

## Why This Is Not a Wild Goose Chase

### 1. The Foundation Is Published Science

Every component of LeWM-VC is backed by peer-reviewed or major-conference work:

| Piece | Where it comes from | Status |
|-------|-------------------|--------|
| JEPA temporal prediction | V-JEPA (LeCun et al., Meta), LeWorldModel (ICLR 2026) | Published, competitive on Push-T at 94% |
| Hyperprior entropy coding | Balle et al. (ICLR 2017), Google DeepMind | Industry standard for learned codecs |
| Perceptual loss (LPIPS) | Zhang et al. (CVPR 2018) | Standard in all modern learned codecs |
| Rate-distortion optimization | Shannon rate-distortion theory, Balle | Mathematical bedrock |
| SIGReg anti-collapse | LeJEPA (Balestriero & LeCun, NeurIPS 2025) | Published |

**We are not inventing anything unproven.** We are combining proven components into a pipeline that hasn't been done in this specific configuration (JEPA residuals + hyperprior entropy + spatial latent grid).

### 2. The Industry Trajectory Validates the Space

- **Deep Render** (acquired by InterDigital, $15M+): neural codec for streaming, similar approach
- **WaveOne** (acquired by Apple): neural video compression
- **Google's** E2E learned codec (Ballé et al.): hyperprior-based, constant improvements
- **Qualcomm** AI research: learned video compression as a core focus

Commercial exits in this space confirm there's real value, not just academic curiosity. The difference is our emphasis on **machine perception** rather than human visual quality, which is a defensible niche.

### 3. The Moat Is Defensible

The dual-layer SVC (predictive base layer + forensic enhancement layer) is unique:

- **No existing codec** (H.265, H.266, AV1, Deep Render, E2E) offers this for machine vision
- **Chain-of-custody claims** require unbuffered raw reference frames on device + compressed stream for satellite — this is our product architecture
- **Surprise-gated bit allocation** is a novel feature not present in any competitor

We can lose on PSNR to x265 and still win on this use case alone.

### 4. The Risks (Honest Assessment)

| Risk | Why it's manageable |
|------|-------------------|
| **Temporal residual coding gain may be small** | Worst case: codec performs as intra-frame only, still competitive at 0.75 BPP/31dB. Temporal gain is upside, not requirement |
| **Dataset too small (456 VIRAT clips)** | VIRAT is surveillance-specific but enough for proof-of-concept. PEVID + SFU-HW + pilot footage de-risk this |
| **SIGReg not publishing novelty** | Not needed — SIGReg is a training stabilizer, not the product moat. The moat is SVC + surprise-gating |
| **BD-rate may not beat x265 at PSNR** | We don't need to win PSNR. We need to win mAP/detection at matched bitrate on surveillance scenes. The codec is optimized for machine, not human, eyeballs |
| **No design partner LOI yet** | Technical risk, not product risk. Training a working model now gives us something to demo |

### 5. The Stop/Go Criteria

**This project should continue if Phase 1 training shows:**

1. P-frames have lower BPP than I-frames at matched quality (temporal coding works)
2. The decoder produces recognizable reconstructions (no mode collapse)
3. Validation loss tracks training loss (no overfitting to 319 clips)

**This project should pivot if:**

1. P-frames are NOT cheaper than I-frames after Phase 1 (temporal coding doesn't work at all)
2. BD-rate is >200% of x265 at matched PSNR (codec is catastrophically worse)
3. Latent space collapses despite SIGReg (representations are garbage)

We are currently at Phase 0 (warmup). None of the stop criteria apply yet. Phase 1 will be the first real signal.

### Bottom Line

**Not a wild goose chase.** The approach is grounded in published science, the niche is defensible, the market exists (Deep Render exit validates), and there's a clear product architecture (dual-layer SVC + surprise gating) that no competitor offers.

The biggest risk is execution — can we get temporal residual coding to actually save bits? That question gets answered in Phase 1, ~10 hours from now.
