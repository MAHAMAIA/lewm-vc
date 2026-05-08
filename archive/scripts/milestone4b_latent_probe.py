#!/usr/bin/env python3
"""
milestone4b_latent_probe.py
Compares LeWM-VC latent representations vs x265-decoded pixels for
downstream machine perception without full pixel reconstruction.
Trains a lightweight probe on the latent grid to predict YOLO detections.
Usage: python3 milestone4b_latent_probe.py
"""

import os, sys, glob, subprocess, shutil, csv, datetime
import cv2, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.distributions.normal import Normal
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESOLUTION = 256
QUANT_STEP = 2.0 / 255
LATENT_H, LATENT_W = 16, 16
LATENT_DIM = 192
DATASET_DIR = 'datasets/pevid-hd'
BENCHMARK_DIR = 'benchmark_milestone4b'
EPOCHS_PROBE = 50
BATCH_SIZE = 8
N_TRAIN_FRAMES = 200
N_TEST_FRAMES = 50
os.makedirs(BENCHMARK_DIR, exist_ok=True)

LOG_FILE = os.path.join(BENCHMARK_DIR, f'latent_probe_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()
sys.stdout = Tee(sys.stdout, open(LOG_FILE, 'w'))
print(f"Logging to {LOG_FILE}")
print(f"Device: {DEVICE}")

# --- LeWM encoder + GMM (no decoder needed) ---
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

def load_detector():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
    model.to(DEVICE).eval()
    return model


def detect(model, frame_bgr):
    results = model(frame_bgr)
    dets = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        dets.append({"class": int(cls), "conf": float(conf), "bbox": [float(x) for x in xyxy]})
    return dets
# --- Probe: lightweight convnet that maps latent grid -> detection heatmap ---
class LatentProbe(nn.Module):
    """Predicts objectness heatmap + class logits from latent grid."""
    def __init__(self, latent_dim=LATENT_DIM, num_classes=80):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.obj_head = nn.Conv2d(32, 1, 1)        # objectness logit
        self.cls_head = nn.Conv2d(32, num_classes, 1)  # class logits
        self.act = nn.GELU()

    def forward(self, latent):
        # latent: [B, 192, 16, 16]
        x = self.act(self.conv1(latent))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        obj = self.obj_head(x)      # [B, 1, 16, 16]
        cls = self.cls_head(x)      # [B, 80, 16, 16]
        return obj, cls

# --- Pixel probe: same architecture but on decoded images ---
class PixelProbe(nn.Module):
    """Predicts objectness heatmap + class logits from image."""
    def __init__(self, num_classes=80):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.obj_head = nn.Conv2d(128, 1, 1)
        self.cls_head = nn.Conv2d(128, num_classes, 1)
        self.act = nn.GELU()

    def forward(self, img):
        # img: [B, 3, 256, 256] -> grid: [B, 128, 16, 16]
        x = self.act(self.conv1(img))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        obj = self.obj_head(x)
        cls = self.cls_head(x)
        return obj, cls

# --- Generate detection targets from YOLO ---
def generate_targets(dets, grid_h=LATENT_H, grid_w=LATENT_W):
    """Convert YOLO detections to spatial grid targets.
    Returns: obj_target [H, W], cls_target [H, W] (class indices, -1 for no obj)"""
    obj_t = torch.zeros(grid_h, grid_w)
    cls_t = torch.full((grid_h, grid_w), -1, dtype=torch.long)
    for d in dets:
        x1, y1, x2, y2 = d['bbox']
        cx = (x1 + x2) / 2 / RESOLUTION
        cy = (y1 + y2) / 2 / RESOLUTION
        gx = min(int(cx * grid_w), grid_w - 1)
        gy = min(int(cy * grid_h), grid_h - 1)
        obj_t[gy, gx] = 1.0
        cls_t[gy, gx] = d['class']
    return obj_t, cls_t

# --- Dataset: latent grids + detection targets ---
class LatentDetectionDataset(Dataset):
    def __init__(self, frames, encoder, affine, detector, compressed=False, x265_frames=None):
        self.frames = frames
        self.encoder = encoder
        self.affine = affine
        self.detector = detector
        self.compressed = compressed
        self.x265_frames = x265_frames

    def __len__(self): return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.compressed and self.x265_frames is not None:
            # Use x265-decoded frame as input to pixel probe
            img = self.x265_frames[idx]
            x = torch.from_numpy(img.astype(np.float32)/255.0).permute(2,0,1)
        else:
            # Original frame
            x = torch.from_numpy(frame.astype(np.float32)/255.0).permute(2,0,1)

        # Run detector on original frame for ground truth
        orig_bgr = frame  # already BGR uint8
        dets = detect(self.detector, orig_bgr)
        obj_t, cls_t = generate_targets(dets)
        return x, obj_t, cls_t

# --- Extract latent for original frame ---
def extract_latent(frame, encoder, affine):
    x = torch.from_numpy(frame.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        latent = affine(encoder(x, return_surprise=False))
    return latent.squeeze(0).cpu()

# --- Train probe ---
def train_probe(probe, train_loader, val_loader, epochs, label):
    probe.to(DEVICE)
    opt = optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    obj_loss_fn = nn.BCEWithLogitsLoss()
    cls_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_val = float('inf')
    for epoch in range(1, epochs+1):
        probe.train()
        total_loss = 0.0; n = 0
        for x, obj_t, cls_t in tqdm(train_loader, desc=f"{label} epoch {epoch}/{epochs}"):
            x = x.to(DEVICE); obj_t = obj_t.to(DEVICE); cls_t = cls_t.to(DEVICE)
            opt.zero_grad()
            obj_pred, cls_pred = probe(x)
            loss_obj = obj_loss_fn(obj_pred.squeeze(1), obj_t)
            loss_cls = cls_loss_fn(cls_pred.permute(0,2,3,1).reshape(-1, cls_pred.shape[1]),
                                    cls_t.reshape(-1))
            loss = loss_obj + loss_cls
            loss.backward()
            opt.step()
            total_loss += loss.item(); n += 1
        sched.step()

        # Validation
        probe.eval()
        val_loss = 0.0; vn = 0
        with torch.no_grad():
            for x, obj_t, cls_t in val_loader:
                x = x.to(DEVICE); obj_t = obj_t.to(DEVICE); cls_t = cls_t.to(DEVICE)
                obj_pred, cls_pred = probe(x)
                loss_obj = obj_loss_fn(obj_pred.squeeze(1), obj_t)
                loss_cls = cls_loss_fn(cls_pred.permute(0,2,3,1).reshape(-1, cls_pred.shape[1]),
                                        cls_t.reshape(-1))
                val_loss += (loss_obj + loss_cls).item(); vn += 1
        avg_val = val_loss / vn
        print(f"{label} epoch {epoch}: train_loss={total_loss/n:.4f}, val_loss={avg_val:.4f}")
        if avg_val < best_val:
            best_val = avg_val

    probe.eval()
    return best_val

# --- Evaluate probe: detection accuracy on grid ---
def evaluate_probe(probe, loader, label):
    probe.eval()
    total_obj_correct = 0; total_obj = 0
    total_cls_correct = 0; total_cls = 0
    with torch.no_grad():
        for x, obj_t, cls_t in tqdm(loader, desc=f"Evaluating {label}"):
            x = x.to(DEVICE); obj_t = obj_t.to(DEVICE); cls_t = cls_t.to(DEVICE)
            obj_pred, cls_pred = probe(x)
            obj_pred_bin = (torch.sigmoid(obj_pred.squeeze(1)) > 0.5).float()
            # Objectness accuracy
            total_obj_correct += (obj_pred_bin == obj_t).sum().item()
            total_obj += obj_t.numel()
            # Class accuracy (only where object present)
            mask = obj_t == 1.0
            if mask.sum() > 0:
                cls_pred_labels = cls_pred.permute(0,2,3,1)[mask].argmax(dim=-1)
                total_cls_correct += (cls_pred_labels == cls_t[mask]).sum().item()
                total_cls += mask.sum().item()

    obj_acc = total_obj_correct / max(1, total_obj)
    cls_acc = total_cls_correct / max(1, total_cls)
    return obj_acc, cls_acc

if __name__ == '__main__':
    # Load encoder from milestone 1
    ae_path = 'checkpoints_milestone1/ae_lambda_0.05_final.pt'
    if not os.path.exists(ae_path):
        print("Run milestone1 first."); exit(1)

    encoder = LeWMEncoder(latent_dim=192, semantic_surprise=True).to(DEVICE)
    decoder = LeWMDecoder(latent_dim=192).to(DEVICE)
    affine = AffineNormalization(192).to(DEVICE)
    ae_state = torch.load(ae_path, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in ae_state.items() if k.startswith('encoder.')}, strict=False)
    decoder.load_state_dict({k.replace('decoder.', ''): v for k, v in ae_state.items() if k.startswith('decoder.')}, strict=False)
    affine.load_state_dict({k.replace('affine.', ''): v for k, v in ae_state.items() if k.startswith('affine.')}, strict=False)
    encoder.eval(); decoder.eval(); affine.eval()

    detector = load_detector()
    print("Models loaded.")

    # Load frames
    video_path = sorted(glob.glob(os.path.join(DATASET_DIR, 'walking*.mpg')))[0]
    cap = cv2.VideoCapture(video_path)
    all_frames = []
    while len(all_frames) < N_TRAIN_FRAMES + N_TEST_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (RESOLUTION, RESOLUTION))
        all_frames.append(frame)
    cap.release()
    print(f"Loaded {len(all_frames)} frames")

    train_frames = all_frames[:N_TRAIN_FRAMES]
    test_frames = all_frames[N_TRAIN_FRAMES:N_TRAIN_FRAMES+N_TEST_FRAMES]

    # --- Generate compressed versions of test frames for x265 baseline ---
    # Use CRF 28 from milestone1 (BPP ~0.02) — this is where x265 gives decent quality
    # For fair comparison, we need x265 at similar BPP to LeWM-VC default
    # LeWM default BPP ~1.95 from earlier run. We'll match that.
    print("Generating x265 compressed frames at matched BPP...")
    x265_decoded_test = []
    target_bpp = 1.95
    # Use compress_x265 function (simplified version)
    import tempfile
    tmpdir = '/tmp/x265_probe'
    os.makedirs(tmpdir, exist_ok=True)
    frame_dir = os.path.join(tmpdir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(test_frames):
        cv2.imwrite(os.path.join(frame_dir, f'frame_{i:06d}.png'), frame)
    # Find CRF for target BPP
    lo, hi = 0, 51
    best_crf, best_bpp = 23, float('inf')
    best_out = None
    for _ in range(10):
        mid = int((lo + hi) / 2)
        out_path = os.path.join(tmpdir, f'x265_crf{mid}.mp4')
        cmd = ['ffmpeg', '-y', '-framerate', '25', '-i', os.path.join(frame_dir, 'frame_%06d.png'),
               '-c:v', 'libx265', '-crf', str(mid), '-preset', 'medium',
               '-x265-params', 'log-level=error', out_path]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0: hi = mid; continue
        bits = os.path.getsize(out_path) * 8
        bpp = bits / (len(test_frames) * RESOLUTION * RESOLUTION * 3)
        if abs(bpp - target_bpp) < abs(best_bpp - target_bpp):
            best_crf, best_bpp = mid, bpp
            if best_out: os.remove(best_out)
            best_out = out_path
        else: os.remove(out_path)
        if bpp > target_bpp: lo = mid
        else: hi = mid
        if lo >= hi - 1: break
    # Decode
    dec_dir = os.path.join(tmpdir, 'decoded')
    os.makedirs(dec_dir, exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-i', best_out, os.path.join(dec_dir, 'frame_%06d.png')],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    for p in sorted(glob.glob(os.path.join(dec_dir, '*.png'))):
        x265_decoded_test.append(cv2.imread(p))
    shutil.rmtree(tmpdir)
    print(f"x265 test frames: {len(x265_decoded_test)} at {best_bpp:.4f} bpp (CRF={best_crf})")

    # --- Prepare datasets ---
    # Latent probe dataset: extract latents from original frames
    train_latents = [extract_latent(f, encoder, affine) for f in tqdm(train_frames, desc="Extracting train latents")]
    test_latents = [extract_latent(f, encoder, affine) for f in tqdm(test_frames, desc="Extracting test latents")]

    # Targets from detector on original frames
    train_targets = [generate_targets(detect(detector, f)) for f in tqdm(train_frames, desc="Train targets")]
    test_targets = [generate_targets(detect(detector, f)) for f in tqdm(test_frames, desc="Test targets")]

    # DataLoaders for latent probe
    class PrecomputedDataset(Dataset):
        def __init__(self, latents, targets):
            self.latents = latents
            self.targets = targets
        def __len__(self): return len(self.latents)
        def __getitem__(self, idx):
            return self.latents[idx], self.targets[idx][0], self.targets[idx][1]

    train_loader_latent = DataLoader(PrecomputedDataset(train_latents, train_targets),
                                     batch_size=BATCH_SIZE, shuffle=True)
    test_loader_latent = DataLoader(PrecomputedDataset(test_latents, test_targets),
                                    batch_size=BATCH_SIZE, shuffle=False)

    # DataLoaders for pixel probe (on x265-decoded test frames + original train frames)
    # Train pixel probe on original frames, test on x265-decoded frames
    pixel_train_dataset = LatentDetectionDataset(train_frames, encoder, affine, detector, compressed=False)
    pixel_test_dataset = LatentDetectionDataset(test_frames, encoder, affine, detector, compressed=True, x265_frames=x265_decoded_test)
    train_loader_pixel = DataLoader(pixel_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader_pixel = DataLoader(pixel_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Train latent probe ---
    print("\n=== Training Latent Probe (on LeWM-VC latents) ===")
    latent_probe = LatentProbe()
    val_loss_latent = train_probe(latent_probe, train_loader_latent, test_loader_latent, EPOCHS_PROBE, "LatentProbe")
    obj_acc_latent, cls_acc_latent = evaluate_probe(latent_probe, test_loader_latent, "LatentProbe")
    print(f"Latent Probe: Obj Acc={obj_acc_latent:.4f}, Cls Acc={cls_acc_latent:.4f}")

    # --- Train pixel probe ---
    print("\n=== Training Pixel Probe (on x265-decoded frames) ===")
    pixel_probe = PixelProbe()
    val_loss_pixel = train_probe(pixel_probe, train_loader_pixel, test_loader_pixel, EPOCHS_PROBE, "PixelProbe")
    obj_acc_pixel, cls_acc_pixel = evaluate_probe(pixel_probe, test_loader_pixel, "PixelProbe")
    print(f"Pixel Probe: Obj Acc={obj_acc_pixel:.4f}, Cls Acc={cls_acc_pixel:.4f}")

    # --- Results ---
    print(f"\n{'='*60}")
    print(f"LATENT PROBE RESULTS")
    print(f"{'='*60}")
    print(f"Test frames: {len(test_frames)}")
    print(f"x265 BPP: {best_bpp:.4f} (CRF={best_crf})")
    print(f"LeWM-VC latent BPP: ~{1.95:.4f} (from milestone4 default)")
    print(f"\nLatent Probe (LeWM-VC latents):")
    print(f"  Objectness Accuracy: {obj_acc_latent:.4f}")
    print(f"  Class Accuracy:      {cls_acc_latent:.4f}")
    print(f"\nPixel Probe (x265-decoded pixels):")
    print(f"  Objectness Accuracy: {obj_acc_pixel:.4f}")
    print(f"  Class Accuracy:      {cls_acc_pixel:.4f}")

    with open(f'{BENCHMARK_DIR}/latent_probe_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'obj_acc', 'cls_acc', 'bpp'])
        w.writerow(['latent_probe', obj_acc_latent, cls_acc_latent, 1.95])
        w.writerow(['pixel_probe_x265', obj_acc_pixel, cls_acc_pixel, best_bpp])
    print(f"\nResults saved to {BENCHMARK_DIR}/latent_probe_results.csv")
