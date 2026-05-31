
# CRITICAL PATH EXPERIMENT: Probe Accuracy at Operational Bitrate
# LeWM-VC Intra-Frame Probe Evaluation at λ=0.05 (~0.11 BPP)
# Date: 2026-05-12
# Objective: Determine if the 86.5% class accuracy holds at the codec's 
#            actual efficient operating point, or if it decays.

## 1. EXPERIMENTAL OVERVIEW

The paper's central claim pairs "13.7× storage reduction" (from temporal 
compression at 0.087 BPP) with "86.5% class accuracy" (from intra-frame 
probe at ~1.95 BPP). These are DIFFERENT operating points.

This experiment closes that gap by:
  A. Training a probe on LeWM-VC intra-frame latents at λ=0.05 (0.109 BPP)
  B. Training a matched probe on x265-decoded frames at ~0.11 BPP
  C. Evaluating both against YOLOv5s teacher outputs on uncompressed frames
  D. (Secondary) Testing probe on P-frame residuals at 0.087 BPP

## 2. PREREQUISITES

### Hardware
- NVIDIA GPU with 16GB+ VRAM (T4, A10, RTX 3090/4090, or MI300X)
- ~50GB disk space for checkpoints and intermediate latents

### Software
- PyTorch 2.0+
- torchvision, opencv-python, pandas, matplotlib
- FFmpeg (for x265 encoding)
- CompressAI (for entropy coding verification)
- YOLOv5s weights ( Ultralytics or original Jocher repo)

### Data
- PEViD-HD dataset (already used in paper)
- Training videos: walking_day_outdoor_1_1, droppingBag_day_indoor_1_1
- Evaluation: held-out 100-frame segments (same as paper)
- IMPORTANT: Use the EXACT same train/test split as Section 4.1

## 3. STEP-BY-STEP PROTOCOL

-----------------------------------------------------------------------------
STEP 1: GENERATE LeWM-VC LATENTS AT λ=0.05
-----------------------------------------------------------------------------

Load the Milestone 1 (intra-frame) checkpoint at λ=0.05.
From Table 1: this produces 0.109 BPP at 25.21 dB PSNR.

```python
# pseudo-code — adapt to your actual model API
import torch
from lewm_vc import Encoder, AffineNorm, Quantizer, GMMENTropy

encoder = Encoder(vit_tiny, hidden_dim=192, num_layers=6)
affine_norm = AffineNorm(num_channels=192)
quantizer = Quantizer(step_size=2/255)
entropy = GMMENTropy(num_components=2)

# Load checkpoint
checkpoint = torch.load("checkpoints/m1_lambda_0.05.pt")
encoder.load_state_dict(checkpoint["encoder"])
affine_norm.load_state_dict(checkpoint["affine_norm"])

# Generate latents for ALL frames (train + eval)
latents = {}
for frame_id, frame in dataset:
    with torch.no_grad():
        z = encoder(frame)           # [B, 192, 16, 16]
        z = affine_norm(z)
        z_q = quantizer(z)            # quantized latent
    latents[frame_id] = z_q.cpu()

# Verify bitrate
bits = entropy.estimate_rate(z_q)    # cross-entropy lower bound
total_pixels = num_frames * 256 * 256 * 3
bpp = bits / total_pixels
assert 0.10 <= bpp <= 0.12, f"BPP {bpp} out of expected range"
```

Save latents to disk as .pt files. You will reuse these for probe training.

-----------------------------------------------------------------------------
STEP 2: GENERATE x265 FRAMES AT MATCHED BPP (~0.11)
-----------------------------------------------------------------------------

From Table 1: x265 at CRF=33 achieves 0.011 BPP. That's 10× LOWER than 
0.11 BPP. We need to find the CRF that produces ~0.11 BPP.

CRF is not linear in BPP, so you must search:

```bash
# Binary search for CRF that yields ~0.11 BPP on PEViD-HD
for CRF in 18 20 22 24 26 28 30 32 33; do
    ffmpeg -i input.mp4 -c:v libx265 -crf $CRF -preset medium            -pix_fmt yuv420p -an output_crf${CRF}.mp4

    # Calculate BPP
    filesize_bits=$(stat -f%z output_crf${CRF}.mp4 2>/dev/null || stat -c%s output_crf${CRF}.mp4)
    frames=$(ffprobe -v error -count_frames -select_streams v:0              -show_entries stream=nb_read_frames -of csv=p=0 input.mp4)
    bpp=$(python -c "print($filesize_bits / ($frames * 256 * 256 * 3))")
    echo "CRF=$CRF -> BPP=$bpp"
done
```

Expected result: CRF ~24–26 should land near 0.11 BPP. 
If not, interpolate between CRF values.

Once you identify the target CRF, decode back to RGB:

```bash
ffmpeg -i output_crf${TARGET}.mp4 -pix_fmt rgb24 frame_%04d.png
```

Load these as your "x265 pixel probe" training data.

-----------------------------------------------------------------------------
STEP 3: GENERATE YOLOv5s TEACHER LABELS
-----------------------------------------------------------------------------

You need ground-truth objectness scores and class logits from YOLOv5s 
running on the UNCOMPRESSED original frames. These are your soft targets.

```python
import torch
from yolov5.models.common import DetectMultiBackend

model = DetectMultiBackend("yolov5s.pt", device=device)
model.eval()

teacher_outputs = {}
for frame_id, frame in dataset:  # ORIGINAL uncompressed frames
    with torch.no_grad():
        pred = model(frame)  # [num_detections, 85] -> xyxy, conf, cls

    # Extract objectness and class logits for the PRIMARY detection
    # (or aggregate across detections — match your Section 4.4 protocol)
    if len(pred) > 0:
        best = pred[pred[:, 4].argmax()]  # highest confidence
        objectness = best[4].item()
        class_logits = best[5:].cpu()      # 80-class COCO logits
    else:
        objectness = 0.0
        class_logits = torch.zeros(80)

    teacher_outputs[frame_id] = {
        "objectness": objectness,
        "class_logits": class_logits,
        "bbox": best[:4].cpu() if len(pred) > 0 else torch.zeros(4)
    }
```

CRITICAL: Use the EXACT same teacher extraction logic as in your 
original Section 4.4 experiment. Any change here invalidates comparison.

-----------------------------------------------------------------------------
STEP 4: TRAIN PROBE ON LeWM-VC LATENTS (0.109 BPP)
-----------------------------------------------------------------------------

Probe architecture: identical to Section 4.4.
Three GELU conv layers: 128 → 64 → 32 channels, plus heads.

```python
import torch.nn as nn

class LatentProbe(nn.Module):
    def __init__(self, latent_dim=192, num_classes=80):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(latent_dim, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # global spatial pool
        )
        self.objectness_head = nn.Linear(32, 1)
        self.class_head = nn.Linear(32, num_classes)

    def forward(self, z):
        # z: [B, 192, 16, 16]
        feat = self.backbone(z).flatten(1)  # [B, 32]
        obj = torch.sigmoid(self.objectness_head(feat))
        cls = self.class_head(feat)  # raw logits
        return obj, cls

# Training
probe_latent = LatentProbe().to(device)
optimizer = torch.optim.AdamW(probe_latent.parameters(), lr=1e-3)
criterion_obj = nn.BCELoss()
criterion_cls = nn.CrossEntropyLoss()

for epoch in range(50):  # match Section 4.4
    for batch in train_loader:  # 200 frames
        z = batch["latent"].to(device)      # LeWM-VC latent
        obj_target = batch["objectness"].to(device)
        cls_target = batch["class_idx"].to(device)

        obj_pred, cls_pred = probe_latent(z)
        loss = criterion_obj(obj_pred.squeeze(), obj_target) +                criterion_cls(cls_pred, cls_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Train on 200 frames, validate on 50. Save best checkpoint.

-----------------------------------------------------------------------------
STEP 5: TRAIN PROBE ON x265 PIXELS (~0.11 BPP)
-----------------------------------------------------------------------------

Identical architecture adapted for RGB input:

```python
class PixelProbe(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.objectness_head = nn.Linear(32, 1)
        self.class_head = nn.Linear(32, num_classes)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        obj = torch.sigmoid(self.objectness_head(feat))
        cls = self.class_head(feat)
        return obj, cls

# Training identical to Step 4, but input is x265-decoded RGB
```

-----------------------------------------------------------------------------
STEP 6: EVALUATE BOTH PROBES
-----------------------------------------------------------------------------

```python
def evaluate_probe(probe, dataloader, device):
    probe.eval()
    obj_correct = 0
    cls_correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["input"].to(device)
            obj_target = batch["objectness"].to(device)
            cls_target = batch["class_idx"].to(device)

            obj_pred, cls_pred = probe(x)

            # Objectness: threshold at 0.5
            obj_pred_bin = (obj_pred.squeeze() > 0.5).float()
            obj_correct += (obj_pred_bin == obj_target).sum().item()

            # Class: argmax
            cls_pred_idx = cls_pred.argmax(dim=1)
            cls_correct += (cls_pred_idx == cls_target).sum().item()
            total += len(x)

    return {
        "objectness_acc": obj_correct / total,
        "class_acc": cls_correct / total,
    }

# Run evaluation
latent_results = evaluate_probe(probe_latent, test_loader_latent, device)
pixel_results = evaluate_probe(probe_pixel, test_loader_pixel, device)

print(f"LeWM-VC Latent (0.109 BPP):  Obj={latent_results['objectness_acc']:.3f}  Cls={latent_results['class_acc']:.3f}")
print(f"x265 Pixel (~0.11 BPP):      Obj={pixel_results['objectness_acc']:.3f}  Cls={pixel_results['class_acc']:.3f}")
```

-----------------------------------------------------------------------------
STEP 7: (SECONDARY) PROBE ON P-FRAME RESIDUALS AT 0.087 BPP
-----------------------------------------------------------------------------

This tests the ACTUAL operating point of the full temporal codec.

```python
# Load Milestone 2 (temporal) checkpoint
predictor = JEPAPredictor(num_layers=8, hidden_dim=256)
temporal_ckpt = torch.load("checkpoints/m2_temporal.pt")
predictor.load_state_dict(temporal_ckpt["predictor"])

# For each P-frame in GOP:
# 1. Encode frame to latent z_t
# 2. Predict z_hat_t from previous decoded latents
# 3. Compute residual r_t = z_t - z_hat_t
# 4. Quantize and entropy-code residual
# 5. Decode: z_tilde_t = z_hat_t + r_tilde_t

# Train probe on z_tilde_t (reconstructed latent after residual decoding)
# NOT on z_t (original unquantized latent)

# Architecture identical to Step 4, but input is z_tilde_t
```

This is the most technically demanding step because the predictor 
must be run in the loop, and latents must be decoded through the 
quantization/entropy pipeline, not taken from encoder output.

## 4. EXPECTED OUTCOMES & DECISION MATRIX

After completing Steps 1–6 (intra-frame at 0.11 BPP), you will have:

| Scenario | LeWM Class Acc | x265 Class Acc | Interpretation |
|----------|---------------|----------------|----------------|
| A (Best) | ≥82%          | ≤75%           | Claim VALIDATED. 13.7× storage + high accuracy is real. |
| B (Good) | 75–82%        | 70–78%         | Claim PARTIALLY validated. Latent still wins but margin shrinks. |
| C (Marginal)| 65–75%     | 60–70%         | Claim WEAKENS. Need to qualify deck: "accuracy trade-off for storage." |
| D (Fail) | <65%          | Any            | Claim BROKEN. Do not pair 13.7× with accuracy. Separate them. |

For Step 7 (temporal P-frames at 0.087 BPP):

| Scenario | P-Frame Probe Acc | Action |
|----------|-------------------|--------|
| ≥75%     | Temporal pipeline validated | Add to deck as "operational accuracy" |
| 60–75%   | Usable but degraded         | Flag as "early results, optimization ongoing" |
| <60%     | Temporal probe fails         | Do NOT claim probe works on P-frames. Stick to I-frame probe + temporal compression as separate benefits. |

## 5. TIMELINE

| Day | Task |
|-----|------|
| 1   | Generate LeWM latents at λ=0.05; verify BPP |
| 1   | Generate x265 frames at matched BPP |
| 2   | Generate YOLOv5s teacher labels |
| 2–3 | Train latent probe (50 epochs, ~4 hours on T4) |
| 3–4 | Train pixel probe (50 epochs, ~4 hours on T4) |
| 4   | Evaluate both; generate comparison table |
| 5   | (If time) Run P-frame residual probe experiment |
| 5   | Update deck with actual numbers; decide go/no-go |

Total wall time: 5 days with one GPU.

## 6. DOCUMENTATION FOR INVESTORS

Produce a one-page "Experimental Validation Memo" with:
1. Exact BPP values achieved for both codecs
2. Train/test split sizes
3. Probe architecture diagram (same as Section 4.4)
4. Accuracy numbers with confidence intervals (run 3 seeds if possible)
5. Qualitative examples: side-by-side of x265 decode vs. latent probe detection
6. Clear statement: "These results were produced at the codec's operational 
   bitrate of 0.109 BPP, not at the higher-fidelity point used in the original 
   preprint probe comparison."

## 7. RISK MITIGATION

If the experiment fails (Scenario C or D):
- Do NOT hide the result. Investors will find out.
- Pivot pitch to: "JEPA-based latent compression achieves 13.7× storage reduction 
  with graceful degradation; probe accuracy is [X]% at operational bitrate, 
  improving to 86.5% at higher fidelity."
- Emphasize the architectural moat (no motion vectors, GMM entropy, surprise metric) 
  rather than the absolute accuracy number.

## 8. REPOSITORY STRUCTURE

```
experiments/
├── probe_low_bitrate/
│   ├── 01_generate_latents.py
│   ├── 02_generate_x265.py
│   ├── 03_generate_teacher_labels.py
│   ├── 04_train_latent_probe.py
│   ├── 05_train_pixel_probe.py
│   ├── 06_evaluate.py
│   ├── 07_pframe_residual_probe.py (optional)
│   └── results/
│       ├── latent_probe_acc.json
│       ├── pixel_probe_acc.json
│       └── comparison_table.csv
```

Execute in order. All scripts should be deterministic (set seeds).

---
END OF PROTOCOL
