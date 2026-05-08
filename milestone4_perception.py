#!/usr/bin/env python3
"""
milestone4_perception.py
Machine perception benchmark: detection preservation on original vs compressed frames.
Compares LeWM-VC vs x265 at matched bitrates.
Uses pretrained YOLOv5s on COCO classes.
Usage: python3 milestone4_perception.py
"""

import os, sys, glob, subprocess, shutil, csv, datetime
import cv2, numpy as np
import torch, torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESOLUTION = 256
QUANT_STEP = 2.0 / 255
DATASET_DIR = 'datasets/pevid-hd'
BENCHMARK_DIR = 'benchmark_milestone4'
N_FRAMES = 30
os.makedirs(BENCHMARK_DIR, exist_ok=True)

LOG_FILE = os.path.join(BENCHMARK_DIR, f'perception_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()
sys.stdout = Tee(sys.stdout, open(LOG_FILE, 'w'))
print(f"Logging to {LOG_FILE}")
print(f"Device: {DEVICE}")

# --- Architecture (same as milestone1) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels); self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels); self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        r = x; x = torch.nn.functional.gelu(self.norm1(x)); x = self.conv1(x)
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
        x = self.proj(latent); x = self.up1(x); x = self.res1(x); x = self.up2(x); x = self.res2(x)
        x = self.up3(x); x = self.res3(x); x = self.up4(x); x = self.res4(x)
        x = torch.sigmoid(self.final(x))
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
    def __init__(self, ae_path, ent_path):
        super().__init__()
        self.ae = self._build_ae()
        self.ae.load_state_dict(torch.load(ae_path, map_location=DEVICE, weights_only=False), strict=False)
        self.ae.to(DEVICE).eval()
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
            def decode(self, l, ts): return self.decoder(l, ts)
        return AE()
    def compress(self, x, target_bpp=None):
        with torch.no_grad():
            latent = self.ae.encode(x)
            if target_bpp is not None:
                lo, hi = QUANT_STEP * 0.01, QUANT_STEP * 100
                for _ in range(20):
                    mid = (lo + hi) / 2
                    xq = torch.round(latent / mid) * mid
                    q = xq + (latent - xq.detach()) * 0.5
                    mu, scale, weight = self.entropy(q)
                    bits = gmm_bits(q, mu, scale, weight, step=mid)
                    if bits / x.numel() > target_bpp: lo = mid
                    else: hi = mid
                step = hi
            else:
                step = QUANT_STEP
            xq = torch.round(latent / step) * step
            q = xq + (latent - xq.detach()) * 0.5
            bits = gmm_bits(q, *self.entropy(q), step=step)
            decoded = self.ae.decode(q, (RESOLUTION, RESOLUTION))
        return decoded, bits.item() / x.numel()

def load_detector():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
    model.to(DEVICE).eval()
    return model

def detect(model, frame_bgr):
    results = model(frame_bgr)
    dets = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        dets.append({'class': int(cls), 'conf': float(conf), 'bbox': [float(x) for x in xyxy]})
    return dets

def compress_x265(frame_list, target_bpp, tmpdir='/tmp/x265_perception'):
    import tempfile
    os.makedirs(tmpdir, exist_ok=True)
    h, w = frame_list[0].shape[:2]
    n_frames = len(frame_list)

    # Write frames as PNG images
    frame_dir = os.path.join(tmpdir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frame_list):
        cv2.imwrite(os.path.join(frame_dir, f'frame_{i:06d}.png'), frame)

    # Encode with x265 at different CRFs, find closest to target_bpp
    lo, hi = 0, 51
    best_crf, best_bpp = 23, float('inf')
    best_out = None

    for _ in range(10):
        mid = int((lo + hi) / 2)
        out_path = os.path.join(tmpdir, f'x265_crf{mid}.mp4')
        cmd = [
            'ffmpeg', '-y',
            '-framerate', '25',
            '-i', os.path.join(frame_dir, 'frame_%06d.png'),
            '-c:v', 'libx265',
            '-crf', str(mid),
            '-preset', 'medium',
            '-x265-params', 'log-level=error',
            out_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0 or not os.path.exists(out_path):
            hi = mid
            continue
        file_bits = os.path.getsize(out_path) * 8
        bpp = file_bits / (n_frames * w * h * 3)
        if abs(bpp - target_bpp) < abs(best_bpp - target_bpp):
            best_crf, best_bpp = mid, bpp
            if best_out:
                os.remove(best_out)
            best_out = out_path
        else:
            os.remove(out_path)
        if bpp > target_bpp:
            lo = mid
        else:
            hi = mid
        if lo >= hi - 1:
            break

    if best_out is None:
        shutil.rmtree(tmpdir)
        raise RuntimeError("x265 encoding failed for all CRF values")

    # Decode to frames
    dec_dir = os.path.join(tmpdir, 'decoded')
    os.makedirs(dec_dir, exist_ok=True)
    subprocess.run([
        'ffmpeg', '-y', '-i', best_out,
        os.path.join(dec_dir, 'frame_%06d.png')
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    decoded = []
    for p in sorted(glob.glob(os.path.join(dec_dir, '*.png'))):
        decoded.append(cv2.imread(p))

    shutil.rmtree(tmpdir)
    return decoded, best_bpp

def extract_frames(video_path, n=N_FRAMES):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (RESOLUTION, RESOLUTION))
        frames.append(frame)
    cap.release()
    return frames

if __name__ == '__main__':
    print("Loading LeWM-VC checkpoint...")
    ae_path = 'checkpoints_milestone1/ae_lambda_0.05_final.pt'
    ent_path = 'checkpoints_milestone1/entropy_lambda_0.05_final.pt'
    if not os.path.exists(ae_path):
        print(f"ERROR: {ae_path} not found. Run milestone1 first."); exit(1)
    lewm = LeWMCompressor(ae_path, ent_path)

    print("Loading YOLOv5s detector...")
    detector = load_detector()
    print("Detector ready.")

    video_path = sorted(glob.glob(os.path.join(DATASET_DIR, 'walking*.mpg')))
    if not video_path: video_path = sorted(glob.glob(os.path.join(DATASET_DIR, '*.mpg')))
    video_path = video_path[0]
    print(f"Test video: {video_path}")

    orig_frames = extract_frames(video_path)
    orig_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in orig_frames]
    print(f"Extracted {len(orig_frames)} frames")

    print("Running detector on original frames...")
    orig_dets = [detect(detector, f) for f in tqdm(orig_frames)]
    orig_count = np.mean([len(d) for d in orig_dets])
    orig_classes = set()
    for d in orig_dets:
        for det in d: orig_classes.add(det['class'])
    print(f"Original: {orig_count:.1f} detections/frame, {len(orig_classes)} classes")

    # Determine default BPP
    with torch.no_grad():
        test_x = torch.from_numpy(orig_frames[0].astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
        _, default_bpp = lewm.compress(test_x, target_bpp=None)
    print(f"LeWM-VC default BPP: {default_bpp:.4f}")

    # Test at default BPP and one lower
    target_bpps = [default_bpp * 0.5, default_bpp]
    results = []

    for target_bpp in target_bpps:
        print(f"\n{'='*60}")
        print(f"Testing at target BPP={target_bpp:.4f}")
        print(f"{'='*60}")

        # LeWM-VC
        lewm_decoded, lewm_bpp_total = [], 0.0
        for frame in tqdm(orig_rgb, desc="LeWM-VC"):
            x = torch.from_numpy(frame.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
            dec, bpp = lewm.compress(x, target_bpp=target_bpp)
            dec_np = (dec.squeeze(0).permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
            lewm_decoded.append(cv2.cvtColor(dec_np, cv2.COLOR_RGB2BGR))
            lewm_bpp_total += bpp
        lewm_bpp = lewm_bpp_total / len(orig_frames)

        # x265
        x265_decoded, x265_bpp = compress_x265(orig_frames, target_bpp)

        # Detection
        lewm_dets = [detect(detector, f) for f in tqdm(lewm_decoded, desc="LeWM detections")]
        x265_dets = [detect(detector, f) for f in tqdm(x265_decoded, desc="x265 detections")]

        lewm_count = np.mean([len(d) for d in lewm_dets])
        x265_count = np.mean([len(d) for d in x265_dets])

        lewm_classes = set(); x265_classes = set()
        for d in lewm_dets:
            for det in d: lewm_classes.add(det['class'])
        for d in x265_dets:
            for det in d: x265_classes.add(det['class'])

        lewm_class_recall = len(lewm_classes & orig_classes) / max(1, len(orig_classes))
        x265_class_recall = len(x265_classes & orig_classes) / max(1, len(orig_classes))

        results.append({
            'target_bpp': target_bpp,
            'lewm_bpp': lewm_bpp, 'x265_bpp': x265_bpp,
            'orig_dets': orig_count,
            'lewm_dets': lewm_count, 'x265_dets': x265_count,
            'lewm_class_recall': lewm_class_recall, 'x265_class_recall': x265_class_recall,
        })

        print(f"LeWM: {lewm_bpp:.4f} bpp, {lewm_count:.1f} dets/frame, class recall={lewm_class_recall:.2f}")
        print(f"x265:  {x265_bpp:.4f} bpp, {x265_count:.1f} dets/frame, class recall={x265_class_recall:.2f}")
        print(f"Detection preservation: LeWM={lewm_count/orig_count:.2f}x, x265={x265_count/orig_count:.2f}x")

    print(f"\n{'='*60}")
    print(f"PERCEPTION BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"Target BPP={r['target_bpp']:.4f}:")
        print(f"  LeWM: {r['lewm_bpp']:.4f} bpp, {r['lewm_dets']:.1f} dets, class recall={r['lewm_class_recall']:.2f}")
        print(f"  x265:  {r['x265_bpp']:.4f} bpp, {r['x265_dets']:.1f} dets, class recall={r['x265_class_recall']:.2f}")

    with open(f'{BENCHMARK_DIR}/perception_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f"\nResults saved to {BENCHMARK_DIR}/perception_results.csv")
    print(f"Log saved to {LOG_FILE}")
