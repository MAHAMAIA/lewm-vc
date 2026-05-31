### Milestone 4: Machine Perception Benchmark

You asked. Here it is. This script takes a trained LeWM-VC model and an x265 baseline, compresses a set of video frames at matched bitrates, runs an object detector on the reconstructions, and compares detection accuracy. The metric is whether LeWM-VC preserves more task-relevant information than x265 when both are constrained to the same bit budget.

```python
#!/usr/bin/env python3
"""
milestone4_perception.py
Machine perception benchmark: detection mAP on original vs compressed frames.
Compares LeWM-VC vs x265 at matched bitrates.
Usage: python3 milestone4_perception.py
"""

import os, sys, glob, subprocess, shutil, json, csv
import cv2, numpy as np
import torch, torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from lewm_vc.encoder import LeWMEncoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESOLUTION = 256
QUANT_STEP = 2.0 / 255

# --- Architecture (same as milestone1) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        r = x
        x = torch.nn.functional.gelu(self.norm1(x)); x = self.conv1(x)
        x = torch.nn.functional.gelu(self.norm2(x)); x = self.conv2(x)
        return x + r

class LeWMDecoder(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4,2,1); self.res1 = ResidualBlock(hidden_dim//2)
        self.up2 = nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4,2,1); self.res2 = ResidualBlock(hidden_dim//4)
        self.up3 = nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4,2,1); self.res3 = ResidualBlock(hidden_dim//8)
        self.up4 = nn.ConvTranspose2d(hidden_dim//8, hidden_dim//16, 4,2,1); self.res4 = ResidualBlock(hidden_dim//16)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim//16, hidden_dim//32, 3,1,1), nn.InstanceNorm2d(hidden_dim//32),
            nn.GELU(), nn.Conv2d(hidden_dim//32, 3,3,1,1),
        )
    def forward(self, latent, target_size=None):
        x = self.proj(latent); x = self.up1(x); x = self.res1(x)
        x = self.up2(x); x = self.res2(x); x = self.up3(x); x = self.res3(x)
        x = self.up4(x); x = self.res4(x); x = torch.sigmoid(self.final(x))
        if target_size: x = torch.nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class AffineNormalization(nn.Module):
    def __init__(self, n): super().__init__(); self.s = nn.Parameter(torch.ones(1,n,1,1)); self.b = nn.Parameter(torch.zeros(1,n,1,1))
    def forward(self, x): return x * self.s + self.b

class GMMEntropyModel(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=256, nc=2):
        super().__init__(); self.latent_dim = latent_dim; self.nc = nc
        self.hp = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hyper_channels, latent_dim * nc * 3, 3, padding=1),
        )
        self.sp = nn.Softplus()
    def forward(self, x):
        p = self.hp(x); B, C, H, W = p.shape; cp = C // self.nc
        p = p.view(B, self.nc, cp, H, W)
        mu = p[:,:,:self.latent_dim]; ls = p[:,:,self.latent_dim:2*self.latent_dim]
        lw = p[:,:,2*self.latent_dim:3*self.latent_dim]
        return mu, self.sp(ls)+1e-5, torch.softmax(lw, dim=1)

def gmm_bits(y, mu, scale, weight, step, eps=1e-12):
    B,C,H,W = y.shape; nc = mu.shape[1]; ye = y.unsqueeze(1).expand(-1,nc,-1,-1,-1)
    n = Normal(mu, scale)
    pmf = torch.clamp(n.cdf(ye+0.5*step)-n.cdf(ye-0.5*step), min=eps, max=1.0)
    return (-torch.log((weight*pmf).sum(dim=1)).mean() / np.log(2)) * y.numel()

class LeWMCompressor(nn.Module):
    """Lightweight wrapper: encode -> quantize -> bits -> decode."""
    def __init__(self, ae_path, ent_path):
        super().__init__()
        self.ae = self._build_ae()
        self.ae.load_state_dict(torch.load(ae_path, map_location=DEVICE, weights_only=False), strict=False)
        self.ae.eval()
        self.entropy = GMMEntropyModel().to(DEVICE)
        sd = torch.load(ent_path, map_location=DEVICE, weights_only=False)
        for k in list(sd.keys()):
            if 'mask' in k: del sd[k]
        self.entropy.load_state_dict(sd, strict=False)
        self.entropy.eval()
    @staticmethod
    def _build_ae():
        class AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = LeWMEncoder(latent_dim=192, semantic_surprise=True)
                self.decoder = LeWMDecoder(latent_dim=192)
                self.affine = AffineNormalization(192)
            def encode(self, x): return self.affine(self.encoder(x, return_surprise=False))
            def decode(self, l, ts): return self.decoder(l, target_size=ts)
        return AE().to(DEVICE)
    def compress(self, x, target_bpp=None):
        """If target_bpp given, search quantization step to match."""
        with torch.no_grad():
            latent = self.ae.encode(x)
            if target_bpp is not None:
                # Binary search step size to hit target bpp
                lo, hi = QUANT_STEP * 0.01, QUANT_STEP * 100
                for _ in range(20):
                    mid = (lo + hi) / 2
                    q = torch.round(latent / mid) * mid
                    mu, scale, weight = self.entropy(q)
                    bits = gmm_bits(q, mu, scale, weight, step=mid)
                    bpp = bits / x.numel()
                    if bpp > target_bpp: lo = mid
                    else: hi = mid
                step = hi
            else:
                step = QUANT_STEP
            q = torch.round(latent / step) * step
            bits = gmm_bits(q, *self.entropy(q), step=step)
            return self.ae.decode(q, target_size=(RESOLUTION, RESOLUTION)), bits.item() / x.numel()

# --- YOLO detector (pretrained) ---
def load_detector():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
    model.to(DEVICE); model.eval()
    return model

def detect(model, frame_bgr):
    """Run YOLOv5 on a BGR frame (0-255 uint8). Returns list of {class, conf, bbox}."""
    results = model(frame_bgr)
    dets = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        dets.append({'class': int(cls), 'conf': float(conf), 'bbox': [float(x) for x in xyxy]})
    return dets

# --- x265 compressor at target bpp ---
def compress_x265(frame_list, target_bpp, tmpdir='/tmp/x265_perception'):
    """Compress a list of BGR uint8 frames with x265 to approximate target_bpp."""
    os.makedirs(tmpdir, exist_ok=True)
    h, w = frame_list[0].shape[:2]
    # Write frames to raw video
    raw_path = os.path.join(tmpdir, 'raw.y4m')
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    # Convert to YUV420p via ffmpeg pipe
    pipe = subprocess.Popen(['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                              '-s', f'{w}x{h}', '-r', '25', '-i', '-',
                              '-pix_fmt', 'yuv420p', raw_path],
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frame_list:
        pipe.stdin.write(f.tobytes())
    pipe.stdin.close(); pipe.wait()

    # Binary search CRF for target bpp
    lo, hi = 0, 51
    best_crf = 23
    best_bpp = float('inf')
    for _ in range(12):
        mid = (lo + hi) / 2
        out_path = os.path.join(tmpdir, f'x265_{mid}.mp4')
        subprocess.run(['ffmpeg', '-y', '-i', raw_path, '-c:v', 'libx265',
                        '-crf', str(int(mid)), '-preset', 'medium', out_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        file_bits = os.path.getsize(out_path) * 8
        bpp = file_bits / (len(frame_list) * w * h * 3)
        if abs(bpp - target_bpp) < abs(best_bpp - target_bpp):
            best_crf = int(mid); best_bpp = bpp
        if bpp > target_bpp: hi = mid
        else: lo = mid
    # Decode best
    dec_dir = os.path.join(tmpdir, 'decoded')
    os.makedirs(dec_dir, exist_ok=True)
    best_out = os.path.join(tmpdir, f'x265_{best_crf}.mp4')
    subprocess.run(['ffmpeg', '-y', '-i', best_out, os.path.join(dec_dir, 'frame_%06d.png')],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    decoded = []
    for p in sorted(glob.glob(os.path.join(dec_dir, '*.png'))):
        decoded.append(cv2.imread(p))
    shutil.rmtree(tmpdir)
    return decoded, best_bpp

# --- Frame extraction from videos ---
def extract_frames(video_path, n=50):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (RESOLUTION, RESOLUTION))
        frames.append(frame)  # BGR uint8
    cap.release()
    return frames

# --- Main benchmark ---
if __name__ == '__main__':
    # Load LeWM-VC
    ae_path = 'checkpoints_milestone1/ae_lambda_0.05_final.pt'
    ent_path = 'checkpoints_milestone1/entropy_lambda_0.05_final.pt'
    if not os.path.exists(ae_path):
        print("Run milestone1 first. Checkpoint not found.")
        exit(1)
    lewm = LeWMCompressor(ae_path, ent_path)

    # Load detector
    detector = load_detector()

    # Extract frames from test video
    video_path = glob.glob('datasets/pevid-hd/walking*.mpg')[0]
    orig_frames = extract_frames(video_path, n=30)  # 30 frames for speed
    orig_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in orig_frames]

    # Get detection baseline on original frames
    print("Running detector on original frames...")
    orig_dets = [detect(detector, f) for f in tqdm(orig_frames)]

    # Determine LeWM-VC BPP at default quantization (from milestone1)
    with torch.no_grad():
        test_x = torch.from_numpy(orig_frames[0].astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
        _, lewm_default_bpp = lewm.compress(test_x, target_bpp=None)
    print(f"LeWM-VC default BPP: {lewm_default_bpp:.4f}")

    # Test at this BPP and one lower
    target_bpps = [lewm_default_bpp * 0.5, lewm_default_bpp]
    results = []

    for target_bpp in target_bpps:
        print(f"\n--- Testing at {target_bpp:.4f} bpp ---")
        # LeWM-VC compression (frame by frame)
        lewm_decoded = []
        lewm_bpp = 0.0
        for frame in tqdm(orig_rgb, desc="LeWM-VC compress"):
            x = torch.from_numpy(frame.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
            dec, bpp = lewm.compress(x, target_bpp=target_bpp)
            dec_np = (dec.squeeze(0).permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
            lewm_decoded.append(cv2.cvtColor(dec_np, cv2.COLOR_RGB2BGR))
            lewm_bpp += bpp
        lewm_bpp /= len(orig_frames)

        # x265 compression
        x265_decoded, x265_bpp = compress_x265(orig_frames, target_bpp)

        # Detection on reconstructions
        lewm_dets = [detect(detector, f) for f in tqdm(lewm_decoded, desc="LeWM detection")]
        x265_dets = [detect(detector, f) for f in tqdm(x265_decoded, desc="x265 detection")]

        # Compute mAP-like metric: average number of detections preserved per frame
        orig_count = np.mean([len(d) for d in orig_dets])
        lewm_count = np.mean([len(d) for d in lewm_dets])
        x265_count = np.mean([len(d) for d in x265_dets])

        # Also check if same classes are detected
        orig_classes = set()
        for d in orig_dets:
            for det in d:
                orig_classes.add(det['class'])
        lewm_classes = set(); x265_classes = set()
        for d in lewm_dets:
            for det in d: lewm_classes.add(det['class'])
        for d in x265_dets:
            for det in d: x265_classes.add(det['class'])

        class_recall_lewm = len(lewm_classes & orig_classes) / max(1, len(orig_classes))
        class_recall_x265 = len(x265_classes & orig_classes) / max(1, len(orig_classes))

        results.append({
            'target_bpp': target_bpp,
            'lewm_bpp': lewm_bpp, 'x265_bpp': x265_bpp,
            'orig_dets_per_frame': orig_count,
            'lewm_dets_per_frame': lewm_count,
            'x265_dets_per_frame': x265_count,
            'lewm_class_recall': class_recall_lewm,
            'x265_class_recall': class_recall_x265,
        })

        print(f"LeWM: {lewm_bpp:.4f} bpp, {lewm_count:.1f} dets/frame, class recall {class_recall_lewm:.2f}")
        print(f"x265:  {x265_bpp:.4f} bpp, {x265_count:.1f} dets/frame, class recall {class_recall_x265:.2f}")

    # Save results
    os.makedirs('benchmark_milestone4', exist_ok=True)
    with open('benchmark_milestone4/perception_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print("\nResults saved to benchmark_milestone4/perception_results.csv")
```

---

## What to Expect at the Conclusion of Milestones 1–4

Here is the state your project will be in after each milestone, what numbers to expect, and what each milestone enables you to claim in the paper.

### After Milestone 1: RD Curve + BD-rate

**What you will have:**
- A CSV file with 6 (BPP, PSNR) pairs for LeWM-VC and 5 for x265.
- A BD-rate percentage. If the GMM entropy model is well-trained, I expect **negative BD-rate (bitrate savings) in the range of -5% to -25%** relative to x265 at the lower end of the quality range (25–35 dB PSNR). At higher PSNR, traditional codecs often outperform learned ones, so the curve may cross. This is normal and publishable if properly contextualized.

**Falsifiable claim:**
> "LeWM-VC with GMM entropy model achieves a BD-rate of -X% relative to x265 on PEViD-HD surveillance footage at 256×256 resolution, demonstrating that JEPA-style latent representations are competitive with block-based hybrid coding for machine-oriented video."

**If BD-rate is positive (LeWM-VC worse than x265 at all rates):** The paper shifts to a machine-perception argument: "While LeWM-VC does not outperform x265 on PSNR, the latent representations preserve more semantic information for downstream tasks, as shown in Milestone 4." This is a harder sell but still publishable with the perception data.

**If the RD points do not form a monotonic curve (higher lambda gives higher bpp with lower PSNR):** Training failed. Debug the entropy model's sigma floor and the affine normalization scale. Do not proceed to Milestone 2.

---

### After Milestone 2: Temporal Coding

**What you will have:**
- A trained codec that does both I‑frame and P‑frame coding.
- A single number: the **P/I bit ratio**. This is the average bits spent on a P‑frame divided by the average bits on an I‑frame, at matched reconstruction quality. For temporal coding to work, this must be **below 1.0**. I expect **0.3–0.7** if the JEPA predictor learns useful dynamics. Below 0.3 is excellent; above 0.8 means the predictor is not helping much.

**Falsifiable claim:**
> "JEPA-based temporal prediction reduces bitrate by X% on P‑frames relative to I‑frames at equivalent reconstruction quality, without explicit motion vectors."

**If P/I > 1.0:** The predictor is not learning useful dynamics, or the residual distribution is harder to model than the raw latent distribution. Possible fixes: (a) increase predictor capacity, (b) pretrain the predictor for more epochs before joint RD training, (c) check that the predictor's context window includes enough frames, (d) verify that the latent space is stationary across time (if affine normalization drifts, temporal prediction becomes impossible). If it still fails after debugging, drop the temporal claim and reposition the paper as a still-image codec with a JEPA-derived architecture — that is a weaker but honest paper.

---

### After Milestone 3: Surprise Gating

**What you will have:**
- BPP measurements on normal vs. anomaly videos with surprise gating active.
- A **BPP ratio (anomaly/normal) that should exceed 1.0**. Currently it is 0.81× (anomaly gets fewer bits because it is visually simpler). After gating, I expect **1.1–1.5×** — anomaly frames should receive more bits, and the anomaly video's overall BPP should be higher than the normal video's.
- A breakdown of high-surprise vs. low-surprise frame counts and average bits for each.

**Falsifiable claim:**
> "VOE surprise gating allocates X% more bits to anomalous frames while maintaining Y% detection accuracy on downstream tasks, demonstrating semantic bitrate allocation without hand-crafted rules."

**If the BPP ratio remains below 1.0:** The surprise signal is not correlated with anomaly content. Possible causes: (a) the VOE predictor was trained on a different dataset or distribution and does not generalize, (b) the "anomaly" in the dropping bag video is too subtle in latent space — the predictor might anticipate the bag being present and just not care where it is, producing low surprise, (c) the fixed thresholds (τ = 0.3, 0.7) are miscalibrated. Fix: plot a histogram of surprise scores for normal vs. anomaly frames and set thresholds based on percentiles.

**If surprise gating works but the overall BPP does not decrease relative to ungated coding on normal video:** That is actually fine — the point is not to reduce average bitrate on everything, it is to reallocate bits toward important content. Frame the result as "matched average bitrate with X% improvement in anomaly detection recall."

---

### After Milestone 4: Perception Benchmark

**What you will have:**
- Detection counts (or mAP if you use a dataset with ground‑truth bounding boxes) on original frames, LeWM-VC reconstructions, and x265 reconstructions at matched bitrates.
- Class recall: what fraction of object classes detected in the original are still detected after compression.
- At matched BPP, I expect **LeWM-VC to preserve more detections and higher class recall than x265**, especially at low bitrates where x265's blocking artifacts destroy small objects. This is the core machine-to-machine argument.

**Falsifiable claim:**
> "At 0.X bpp, LeWM-VC preserves Y% of object detections vs. Z% for x265, a Δ of W percentage points, demonstrating that semantic latent compression is more task-preserving than pixel-centric hybrid coding at low bitrates."

**If x265 outperforms LeWM-VC on detection preservation:** Your thesis is wrong. The entire machine-to-machine argument collapses. In that case, you have an intra‑frame codec that compresses okay and a bunch of disconnected components. That is still a Master's thesis but not a startup and not a top‑tier paper. The honest move is to publish the architectural exploration with negative results and move on.

**If LeWM-VC wins on detection but loses on PSNR:** That is the paper. That is the entire argument. You lead with that trade-off and frame it as "PSNR is the wrong metric for machine-consumed video."

---

### Final State After All Four Milestones

If all four succeed, you will have:

| Asset | Content |
|-------|---------|
| RD curve | 6 LeWM-VC points + 5 x265 points, BD-rate computed |
| Temporal ablation | All-intra vs. IPPP BPP comparison, P/I bit ratio |
| Surprise gating | BPP ratio anomaly/normal > 1.0, high-surprise frame analysis |
| Perception benchmark | Detection preservation LeWM-VC vs. x265 at matched bitrate |
| Code | Clean, single‑script‑to‑reproduce for each claim |
| Paper | 8‑page CVPR/DCC/ICLR workshop submission with all four experiments |

That paper is defensible, reproducible, and makes a genuine contribution. It also passes investor technical diligence.

If any milestone fails, the paper scope contracts to what succeeded. Do not fabricate results to compensate. The four scripts above are designed so that each stands alone — a failure in one does not invalidate the others.