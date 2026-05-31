# FFmpeg Plugin Integration

## Status

🔧 **C code exists, Python bridge exists, needs compilation on target.**

The FFmpeg plugin at `ffmpeg/lewm_vc_encoder.c` is a standard AVCodec
plugin that registers encoder and decoder with FFmpeg. It embeds Python
via `Python.h` and calls into `lewm_vc.codec.get_encoder()` which loads
the trained model and runs inference.

## Architecture

```
ffmpeg -i input.mp4 -c:v lewmvc output.lewm

        │
        ▼
┌───────────────────┐
│ FFmpeg            │
│  AVCodec plugin   │
│  ff_lewmvc_encoder │
└───────┬───────────┘
        │ Python.h C API
        │
┌───────▼───────────┐
│ Python interpreter │
│  lewm_vc.codec     │
│  get_encoder()     │
└───────┬───────────┘
        │
┌───────▼───────────┐
│ PyTorch (CUDA)     │
│  encode/decode     │
│  best.pt           │
└───────────────────┘
```

## Compilation

### Prerequisites (Jetson Orin NX / Jetson)

```bash
# JetPack includes these by default:
apt install libavcodec-dev libavutil-dev libavformat-dev
apt install python3-dev python3-pip

# FFmpeg 7.x or 8.x (JetPack 6.x ships FFmpeg 6.x, install from source for 7+)
# For FFmpeg 8.x: callbacks (.init, .encode) removed from public AVCodec API
#   → need to use internal FFCodec or ship FFmpeg 7.x compatibility layer

# Python packages (inside Python environment)
pip install torch torchvision numpy opencv-python
```

### Build

```bash
cd ffmpeg

# Set FFmpeg include path for Jetson
export FFMPEG_PREFIX=/usr
export PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

make CC=aarch64-linux-gnu-gcc \
     CFLAGS="-O2 -fPIC -Wall -I$FFMPEG_PREFIX/include -I$PYTHON_INCLUDE -D__STDC_CONSTANT_MACROS" \
     LDFLAGS="-L$FFMPEG_PREFIX/lib -lavcodec -lavutil -lpython3.10"
```

### FFmpeg 8.x Compatibility

FFmpeg 8.x (libavcodec major 62+) removed the `.init`, `.encode`, `.close`
fields from the `AVCodec` public struct. The current C code has a
compile-time check:

```c
#if LIBAVCODEC_VERSION_MAJOR < 62
    .init = lewmvc_encoder_init,
    .encode = lewmvc_encoder_encode,
    .close = lewmvc_encoder_close,
#endif
```

For FFmpeg 8.x, two options:

1. **Patch FFmpeg** — register callbacks via internal FFCodec struct
   (requires recompiling FFmpeg from source with minor patch)
2. **Use `ffmpeg` CLI wrapper** — instead of a plugin, provide a Python
   script that wraps ffmpeg:

```python
# lewmvc_wrap.py — wraps ffmpeg pipe
# Usage: ffmpeg -i input.mp4 -f rawvideo pipe: | python lewmvc_wrap.py encode > output.lewm
import sys, subprocess
from lewm_vc.codec import get_encoder

enc = get_encoder()
while True:
    frame_bytes = sys.stdin.buffer.read(WIDTH * HEIGHT * 3 // 2)  # YUV420
    if not frame_bytes: break
    packet = enc.encode(WIDTH, HEIGHT, frame_bytes)
    sys.stdout.buffer.write(packet)
```

Recommend **option 2** for pilot deployments (10 lines of Python, no FFmpeg
recompilation). Move to option 1 for production.

## Decoder Plugin

The decoder C plugin (`ffmpeg/lewm_vc_decoder.c`) mirrors the encoder:

```
ffmpeg -i input.lewm -c:v lewmvc_decoder output.mp4
```

Current status: stub implementation (same Python embedding approach).
Full decoder requires the bitstream specification to be implemented first
(see `md/bitstream-spec.md`).

## Testing

### Unit Test (Python → Plugin)

```bash
# Encode a single frame
python3 -c "
from lewm_vc.codec import get_encoder
import numpy as np

enc = get_encoder()
frame = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
import cv2
yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420).tobytes()
packet = enc.encode(256, 256, yuv)
print(f'Encoded: {len(packet)} bytes')
"
```

### Integration Test

```bash
# Raw video → lewm bitstream → decoded frames
python3 lewmvc_wrap.py encode < input.yuv > output.lewm
python3 lewmvc_wrap.py decode < output.lewm > decoded.yuv
diff <(md5sum input.yuv) <(md5sum decoded.yuv)  # lossless!
```

### Performance Benchmark

```bash
# Measure encode throughput (PyTorch GPU)
python3 -c "
from lewm_vc.codec import get_encoder
import time, numpy as np, cv2

enc = get_encoder()
frame = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV_I420).tobytes()

start = time.perf_counter()
for _ in range(100):
    enc.encode(256, 256, yuv)
elapsed = time.perf_counter() - start
print(f'{100/elapsed:.1f} fps')
"
```

Target: 30+ fps on Jetson Orin NX (with TensorRT, see `tensorrt-export.md`).

## Deployment

### Container Build

```dockerfile
FROM nvcr.io/nvidia/jetson:r36.3.0-py3  # JetPack 6.x

# Install deps
RUN apt update && apt install -y ffmpeg libavcodec-dev python3-pip
RUN pip install torch torchvision numpy opencv-python

# Copy codec
COPY lewm_vc/ /app/lewm_vc/
COPY ffmpeg/ /app/ffmpeg/
COPY weights/best.pt /app/weights/

# Build plugin
RUN cd /app/ffmpeg && make

# Register plugin with FFmpeg
ENV AVCODEC_PLUGINS=/app/ffmpeg/liblewmvc.so
```

### Pipeline Check

- [ ] C code compiles on Jetson (Python.h + FFmpeg headers)
- [ ] Python codec module loads and finds best.pt
- [ ] `ffmpeg -c:v lewmvc` lists in codec list
- [ ] Single-frame encode → decode roundtrip
- [ ] 10-second clip encode → decode with PSNR > 30 dB
- [ ] Benchmark: fps > 30 on Orin NX
