# LeWM-VC Bitstream Specification

## Status

❌ **Not implemented.** Current training estimates bits (BPP via KL divergence)
but never produces a real compressed bitstream. The `scripts/inference.py` and
`src/lewm_vc/codec.py` output placeholder packets with frame_type + bit count
metadata, not actual compressed latent data.

## Overview

LeWM-VC uses a NAL-unit-based bitstream structure inspired by H.264/H.265 but
simplified for the latent-space architecture. Each frame produces one or more
NAL units:
- **I-frame**: quantized latent tensor (192 channels, H/16 × W/16 spatial)
- **P-frame**: quantized residual tensor (same shape)

These tensors are compressed with the trained entropy model (GMM hyperprior)
using a range/arithmetic coder.

## NAL Unit Structure

```
┌──────────────────────────────────────────────┐
│ NAL Unit Header (4 bytes)                    │
├──────────────────────────────────────────────┤
│  sync_byte (1)  |  nal_type (1)  |  size (2) │
│  0x4C (0x4C)    |  see below     |  big-end  │
└──────────────────────────────────────────────┘
┌──────────────────────────────────────────────┐
│ Entropy Parameters (variable)                │
├──────────────────────────────────────────────┤
│  μ offset  |  σ scale  |  component weights  │
│  8 bytes      8 bytes     4×num_comp bytes   │
└──────────────────────────────────────────────┘
┌──────────────────────────────────────────────┐
│ Compressed Latent Data (variable)            │
├──────────────────────────────────────────────┤
│  Arithmetic-coded latent tensor elements     │
│  Scalar per element: Q(index) → bits         │
└──────────────────────────────────────────────┘
```

### NAL Types

| Type | Code | Description |
|------|------|-------------|
| I_FRAME | 0x01 | Complete quantized latent |
| P_FRAME | 0x02 | Quantized residual (predictor context not included) |
| P_CONTEXT | 0x03 | Predictor context state (for random access) |
| EOS | 0x04 | End of sequence |

### Sync Byte

Fixed byte `0x4C` (ASCII 'L' for LeWM) used for stream synchronization.

## Entropy Coding

### GMM Parameter Quantization

The hyperprior entropy model produces per-spatial-position GMM parameters:
- **μ** (mean): 2 components × 192 channels × H/16 × W/16
- **σ** (scale): same shape
- **w** (mixture weight): same shape (normalized via softmax)

These are quantized before transmission:
- μ: 8-bit uniform quantization, range [-1.0, 1.0]
- log(σ²): 8-bit uniform quantization, range [-10, 2]
- w: 6-bit per component (shared CDF across spatial positions)

### Arithmetic Coding (rANS)

Use rANS (range Asymmetric Numeral Systems) for entropy coding:

```python
class RangeEncoder:
    """Encode symbols using entropy model predicted probabilities."""

    def __init__(self, precision: int = 16):
        self.precision = precision  # bits of precision
        self.stream = []
        self.state = 0

    def encode(self, symbol: int, pdf: torch.Tensor):
        """Encode one symbol given its probability distribution over [0, 255]."""
        cdf = torch.cumsum(pdf, dim=-1)
        # rANS encode step
        ...

    def flush(self) -> bytes:
        """Finalize and return compressed bytes."""
        ...

class RangeDecoder:
    def __init__(self, data: bytes, precision: int = 16):
        ...

    def decode(self, cdf: torch.Tensor) -> int:
        """Decode one symbol given CDF."""
        ...
```

### Per-Element Coding

Each latent element (192 channels × 256 spatial = 49,152 for 256×256 frame)
is coded independently using the GMM parameters at that spatial position:

```python
# Encode one element
mu, sigma, weight = entropy_model.get_params(quantized, position)
pdf = mixture_pdf(mu, sigma, weight)  # discretized over [-128, 127] x quant_step
bits = encoder.encode(quantized_element, pdf)

# Decode
cdf = mixture_cdf(mu, sigma, weight)
quantized_element = decoder.decode(cdf)
```

## Frame Assembly

### I-frame

```
NAL(I_FRAME)
  ├── μ quantization table (192 × 2 × 16-bit)
  ├── logσ² quantization table (192 × 2 × 16-bit)
  ├── weight values (2 × 6-bit)
  └── Compressed latent elements (49,152 arith-coded symbols)
```

### P-frame

```
NAL(P_FRAME)
  ├── μ quantization table (192 × 2 × 16-bit)
  ├── logσ² quantization table (192 × 2 × 16-bit)
  ├── weight values (2 × 6-bit)
  └── Compressed residual elements (49,152 arith-coded symbols)
```

### GOP Structure

```
IDR I-frame  → P1 → P2 → P3 → P4 → P5 → P6 → ...
  (IDR resets predictor context)

No P-frames reference outside the current GOP (context_len = 4).
For random access, insert P_CONTEXT NALs which carry the predictor state
without coded frame data (for seeking).
```

## Bitrate Estimation (Current Training)

During training, bitrate is estimated via the KL divergence:

```python
kl = 0.5 * (mu² + σ² - log(σ²) - 1)
kl_bits = kl / log(2)  # theoretical minimum bits
```

This is a continuous approximation of the true discrete entropy. The
arithmetic coder achieves rate within 1-2% of this bound for well-trained
entropy models. The difference is the **coding gap** — negligible for
pilot demos, important for final BD-rate claims.

## Implementation Plan

### Phase 1: Python rANS (2 days)

1. Implement `RangeEncoder`/`RangeDecoder` classes in `src/lewm_vc/codec.py`
2. Integrate with existing entropy model forward pass
3. Verify: encode → decode roundtrip produces identical latents at < 1 cpb overhead
4. Test on 1,000 frames from VIRAT, measure actual bits vs KL estimate

### Phase 2: Bitstream Format (1 day)

1. Define NAL structure as protobuf or flatbuffers schema
2. Implement serialization/deserialization in Python
3. Produce verifiable .lewm files: `ffmpeg -i input.mp4 -c:v lewmvc output.lewm`

### Phase 3: C Integration (3 days)

1. Reimplement rANS in C (for FFmpeg plugin performance)
2. Port NAL serialization to C
3. Benchmark: C rANS vs Python rANS (target: 10× faster)

### Phase 4: Validation (2 days)

1. End-to-end test: raw video → encode → bitstream → decode → raw video
2. Verify lossless roundtrip (encoder output == decoder input at matched QP)
3. Measure: encode time, decode time, bitrate vs KL estimate for 100 sequences

## Open Questions

| Question | Decision needed |
|----------|-----------------|
| rANS vs Range coding? | rANS is simpler, Range has better IP protection |
| Fixed Huffman for μ/σ headers? | Headers are small (< 50KB), fixed Huffman is fast |
| Per-sequence entropy model? | Current per-frame model adapts, but adds overhead |
| Resync marker interval? | Insert sync every N frames for error resilience |
