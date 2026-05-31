# LeWM-Eval: Semantic Probing Pipeline

## How to Evaluate Any Codec

The semantic probing pipeline is codec-agnostic. It measures how well a compressed video preserves task-relevant information regardless of the codec used to compress it.

### Step 1: Decode your video frames

Encode and decode your video using any codec (x265, VVC, AV1, your own learned codec), then write the decoded frames to disk as PNGs or YUV:

```bash
# Example: encode + decode with x265
x265 --input input.yuv --input-res 256x256 --crf 28 -o encoded.h265
ffmpeg -i encoded.h265 decoded_frames/%04d.png
```

### Step 2: Run the probe

```bash
python semantic_probe.py \
    --frames decoded_frames/ \
    --teacher yolov5s \
    --output results.json
```

### Step 3: Compare at matched bitrate

The probe reports objectness accuracy and class accuracy. Compare across codecs at the same BPP (sweep CRF/quality parameter) to produce rate-accuracy curves.

## Supported Teachers

- YOLOv5s (Ultralytics auto-download)
- YOLOv5su (Ultralytics auto-download)

## Adding Custom Codec Wrappers

See `experiment/common.py` for the dataset and probe infrastructure. The probe pipeline is:
1. Load decoded frames → `torch.Tensor [T, 3, H, W]`
2. Run frozen teacher detector → pseudo-labels
3. Train lightweight CNN probe (3 conv layers, 128→64→32 channels)
4. Report objectness accuracy + class accuracy

Any codec that produces decoded frames at a known resolution can be evaluated.
