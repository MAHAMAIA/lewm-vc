# TensorRT Export Plan

## Status

❌ **Not started.** Current model runs in PyTorch (`torch.nn.Module` with
`torch.no_grad()`). On MI300X this is fine for training. On Jetson Orin NX,
TensorRT is needed for real-time inference (30+ fps).

## Architecture Compatibility

### Ops Used by LeWM-VC Models

| Module | PyTorch Ops | TensorRT Support |
|--------|-------------|------------------|
| **Encoder** | Conv2d, LayerNorm, GELU, Linear, MultiheadAttention | ✅ Conv2d, LayerNorm, GELU. Attention via FlashAttention (TRT 8.6+) |
| **Predictor** | Conv2d, LayerNorm, GELU, TransformerEncoder | ✅ Same as encoder |
| **Decoder** | ConvTranspose2d, Conv2d, InstanceNorm, GELU, Sigmoid | ✅ All supported (ConvTranspose needs static shape) |
| **Entropy Model** | Conv2d, ReLU, Softplus | ✅ All supported |
| **Quantizer** | Round, Clamp, Straight-Through | ❌ STE not supported by TRT. Option: custom plugin or pre/post-processing |

### Risk Areas

1. **STE Quantizer:** TensorRT doesn't support the `detach()` pattern.
   Solution: move quantization to pre/post-processing (CPU or CUDA kernel).

2. **MultiheadAttention** (encoder): FlashAttention requires TRT 8.6+.
   On Jetson JetPack 6.x (TRT 8.5+), use manual attention or fallback to
   grouped MHCA plugin.

3. **ConvTranspose2d** (decoder): Static shape required. Input resolution
   must be fixed at export time. For variable-resolution support, export
   at max resolution and use region-of-interest cropping.

## Export Strategy

### Step 1: ONNX Export (1 day)

```python
def export_to_onnx(encoder, predictor, decoder, sample_input):
    """Export each module to ONNX independently."""
    torch.onnx.export(
        encoder,
        sample_input,  # [1, 3, 256, 256]
        "encoder.onnx",
        opset_version=17,
        input_names=["input"],
        output_names=["latent", "surprise"],
        dynamic_axes={"input": {0: "batch"}},
    )
    # Similar for predictor, decoder, entropy_model
```

Challenges:
- Encoder has optional `return_surprise` flag — export without it
- Predictor takes a list of tensors (variable length) — need to pad to
  fixed context_len and use a mask

### Step 2: TensorRT Build (1 day)

```python
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, precision="fp16"):
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(builder.create_network(), TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    # Optimize for Jetson
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    return builder.build_serialized_network(builder.network, config)
```

Precision options:

| Precision | Speed vs PyTorch | Quality impact |
|-----------|-----------------|----------------|
| FP32 | 1.5-2× | None |
| FP16 | 3-4× | Negligible (< 0.1 dB PSNR) |
| INT8 | 5-6× | Calibration needed (~0.3 dB) |

Target: **FP16** for pilot deployments. INT8 after validation.

### Step 3: Custom Quantizer Plugin (2 days)

TensorRT doesn't support STE. Two approaches:

**Option A: Pre-processing (Recommended)**
```cuda
// CUDA kernel: quantize latent before TRT inference
__global__ void quantize_kernel(float* input, float* output, float step, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float rounded = roundf(input[idx] / step) * step;
        output[idx] = rounded;
    }
}

// Decoder receives pre-quantized tensor — no quantization ops inside TRT
```

**Option B: TRT Plugin**
```cpp
class QuantizePlugin : public IPluginV2DynamicExt {
    // Implement STE gradient for training, hard round for inference
    // More complex but integrated end-to-end
};
```

Recommend **Option A** — simpler, zero quality loss, and matches the inference
path (hard rounding outside the model graph).

### Step 4: Pipeline Integration (1 day)

```cpp
// Encode pipeline (C++ with TRT + CUDA):
// 1. Preprocess: RGB → float tensor, resize to 256×256
// 2. encoder_engine.execute(latent_output)
// 3. quantize_kernel(latent_output, quantized)  // custom CUDA
// 4. entropy_model_engine.execute(quantized, mu_output, sigma_output)
// 5. Predictor for P-frames: concat latents, execute predictor_engine
// 6. residual = latent - pred_mean
// 7. quantize_kernel(residual → quant_residual)

// Decode pipeline:
// 1. I-frame: decoder_engine.execute(quant_latent → frame)
// 2. P-frame: predictor_engine.execute(ctx → pred_mean)
// 3. recon_latent = pred_mean + quant_residual
// 4. decoder_engine.execute(recon_latent → frame)
```

## Multi-Model Engine Management

The codec requires 4 engines running together per frame:

```
Encoder Engine (1× per frame)
Predictor Engine (1× per P-frame)
Decoder Engine (1× per frame)
Entropy Model Engine (1× per frame)
```

Architecture:

```cpp
class LeWMInference {
    trt::UniqueEngine encoder_engine;
    trt::UniqueEngine predictor_engine;
    trt::UniqueEngine decoder_engine;
    trt::UniqueEngine entropy_engine;

    // CUDA streams for parallel execution
    cudaStream_t encode_stream;
    cudaStream_t decode_stream;

    // Context cache (separate context per engine for thread safety)
    // Jetson Orin NX: 2× DLA + 1× GPU → use DLA for encoder/decoder
};
```

## Jetson Optimization

| Optimization | Expected Gain | Effort |
|-------------|---------------|--------|
| DLA offload (encoder) | 30% latency reduction | 1 day |
| FP16 precision | 3-4× vs PyTorch | 0 (config flag) |
| CUDA graph capture (static sequence length) | 10-15% throughput | 1 day |
| Batch 1 optimization | Standard for edge | Default |

### Memory Budget (Jetson Orin NX 16GB)

| Component | Memory (FP16) |
|-----------|--------------|
| Encoder | ~300 MB |
| Predictor | ~400 MB |
| Decoder | ~200 MB |
| Entropy | ~300 MB |
| CUDA runtime + buffers | ~500 MB |
| **Total** | **~1.7 GB** |

Well within the 16GB budget.

## Testing Plan

| Test | Criteria | When |
|------|----------|------|
| PyTorch → ONNX | All ops export without warnings | Step 1 |
| ONNX → TRT | Build succeeds, no fallback kernels | Step 2 |
| TRT → PyTorch bit exact | max diff < 1e-3 on 100 frames | Step 3 |
| End-to-end latency | < 33 ms per frame (30 fps) | Step 4 |
| Memory | < 4 GB during encode+decode | Step 4 |

## Fallback

If TensorRT export fails for any module, fall back to **LibTorch C++ API**
for that module. Mix TRT (where compatible) with LibTorch (for
unsupported ops). This is slower but guaranteed to work.
