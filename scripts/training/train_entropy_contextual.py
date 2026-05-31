#!/usr/bin/env python3
"""
Final entropy model training with:
- Enhanced hyperprior (down/up + skip)
- Checkerboard autoregressive context
- Laplacian likelihood
Target: 2.5–3.5 bpp
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.distributions.laplace import Laplace
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Affine autoencoder (same as before) ----------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        residual = x
        x = torch.nn.functional.gelu(self.norm1(x))
        x = self.conv1(x)
        x = torch.nn.functional.gelu(self.norm2(x))
        x = self.conv2(x)
        return x + residual

class LeWMDecoder(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(latent_dim, hidden_dim, 1)
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4,2,1)
        self.res1 = ResidualBlock(hidden_dim//2)
        self.up2 = nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4,2,1)
        self.res2 = ResidualBlock(hidden_dim//4)
        self.up3 = nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4,2,1)
        self.res3 = ResidualBlock(hidden_dim//8)
        self.up4 = nn.ConvTranspose2d(hidden_dim//8, hidden_dim//16, 4,2,1)
        self.res4 = ResidualBlock(hidden_dim//16)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim//16, hidden_dim//32, 3,1,1),
            nn.InstanceNorm2d(hidden_dim//32),
            nn.GELU(),
            nn.Conv2d(hidden_dim//32, 3, 3,1,1),
        )
    def forward(self, latent, target_size=None):
        x = self.proj(latent)
        x = self.up1(x); x = self.res1(x)
        x = self.up2(x); x = self.res2(x)
        x = self.up3(x); x = self.res3(x)
        x = self.up4(x); x = self.res4(x)
        x = self.final(x)
        x = torch.sigmoid(x)
        if target_size:
            x = torch.nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class AffineNormalization(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x):
        return x * self.scale + self.shift

class VideoAutoencoderWithAffine(nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.affine = AffineNormalization(latent_dim)
    def encode(self, x):
        latent = self.encoder(x, return_surprise=False)
        return self.affine(latent)

# ---------- Contextual entropy model (hyperprior + checkerboard) ----------
class CheckerboardContext(nn.Module):
    """Checkerboard autoregressive context for latent refinement."""
    def __init__(self, channels, hidden_dim=128):
        super().__init__()
        self.mask_conv = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.refine = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1),
        )
        # Precompute checkerboard mask (static)
        self.register_buffer('mask', None)
        
    def forward(self, x, full_context=False):
        # x: latent features [B, C, H, W]
        # For simplicity, we use a 2x2 checkerboard pattern (half the positions are masked)
        if self.mask is None or self.mask.shape[2:] != x.shape[2:]:
            h, w = x.shape[2], x.shape[3]
            mask = torch.zeros(1, 1, h, w, device=x.device)
            mask[..., 0::2, 0::2] = 1
            mask[..., 1::2, 1::2] = 1
            self.mask = mask
        # Apply masked convolution (we implement by zeroing out masked positions after conv)
        out = self.mask_conv(x)
        out = self.refine(out)
        if not full_context:
            # Zero out positions that are not in the current checkerboard step
            out = out * self.mask
        return out

class ContextualEntropyModel(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=512, context_hidden=128):
        super().__init__()
        # Hyperprior CNN (down + up + skip)
        self.down = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2, stride=2),
            nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2),
            nn.GELU(),
        )
        self.skip_proj = nn.Conv2d(latent_dim, hyper_channels, 1)
        self.head = nn.Conv2d(hyper_channels, latent_dim * 2, 1)  # mu and log_scale
        
        # Checkerboard context
        self.context = CheckerboardContext(latent_dim, context_hidden)
        
        # Additional refinement after context
        self.refine_mu = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        self.refine_scale = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        
    def forward(self, x):
        # x: quantized latents [B, C, H, W]
        # 1. Hyperprior base prediction
        x_down = self.down(x)
        x_up = self.up(x_down)
        x_skip = nn.functional.interpolate(x, size=x_up.shape[2:], mode='bilinear', align_corners=False)
        x_skip = self.skip_proj(x_skip)
        features = x_up + x_skip
        base_params = self.head(features)
        mu_base = base_params[:, :192, :, :]
        log_scale_base = base_params[:, 192:, :, :]
        
        # 2. Checkerboard context (refinement)
        # Use base mu as input to context? Usually context refines both mu and scale.
        # We'll compute a residual offset for mu and scale.
        ctx = self.context(x)  # refine from quantized latents (with masking)
        mu_offset = self.refine_mu(ctx)
        scale_offset = self.refine_scale(ctx)
        
        mu = mu_base + mu_offset
        log_scale = log_scale_base + scale_offset
        return mu, log_scale

# ---------- Load frozen autoencoder ----------
autoencoder = VideoAutoencoderWithAffine(latent_dim=192).to(device)
ae_checkpoint = '/root/le-maia/checkpoints_affine/autoencoder_affine_final.pt'
autoencoder.load_state_dict(torch.load(ae_checkpoint, map_location=device, weights_only=False), strict=False)
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False
print("Autoencoder loaded and frozen.")

# ---------- Dataset (same) ----------
class LatentDataset(Dataset):
    def __init__(self, video_paths, frame_size=(256,256), frames_per_clip=4):
        self.videos = video_paths
        self.frame_size = frame_size
        self.frames_per_clip = frames_per_clip
    def __len__(self):
        return len(self.videos) * 200
    def __getitem__(self, idx):
        video_idx = idx % len(self.videos)
        cap = cv2.VideoCapture(self.videos[video_idx])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = np.random.randint(0, max(1, total - self.frames_per_clip))
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(self.frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2,0,1))
            frames.append(frame)
        cap.release()
        return torch.from_numpy(np.stack(frames)).float()

video_paths = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')
dataset = LatentDataset(video_paths, frame_size=(256,256), frames_per_clip=4)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

# ---------- Quantizer step ----------
QUANT_STEP = 2.0 / 255
print(f"Quantizer step = {QUANT_STEP:.6f}")

# ---------- Entropy model and quantizer ----------
entropy_model = ContextualEntropyModel(latent_dim=192, hyper_channels=512, context_hidden=128).to(device)
quantizer = Quantizer(num_levels=256, mode='training').to(device)

# Resume if checkpoint exists
checkpoint_dir = '/root/le-maia/checkpoints_entropy_contextual'
os.makedirs(checkpoint_dir, exist_ok=True)
resume_ckpt = f"{checkpoint_dir}/entropy_final.pt"
start_epoch = 1
if os.path.exists(resume_ckpt):
    entropy_model.load_state_dict(torch.load(resume_ckpt, map_location=device, weights_only=False))
    print(f"Resumed from {resume_ckpt}")
    if os.path.exists(f"{checkpoint_dir}/epoch.txt"):
        with open(f"{checkpoint_dir}/epoch.txt", 'r') as f:
            start_epoch = int(f.read().strip()) + 1
        print(f"Resuming from epoch {start_epoch}")

# ---------- Laplacian likelihood with discrete bins ----------
def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.05, epsilon=1e-9):
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    scale = torch.clamp(scale, min=sigma_floor, max=5.0)
    laplace = Laplace(mu, scale)
    cdf_upper = laplace.cdf(y + 0.5 * step)
    cdf_lower = laplace.cdf(y - 0.5 * step)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    nll = -torch.log(pmf)
    return nll.mean()

# ---------- Optimizer (higher LR) ----------
optimizer = optim.AdamW(entropy_model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

EPOCHS = 100

for epoch in range(start_epoch, EPOCHS+1):
    sigma_floor = max(0.08, 0.25 - epoch * 0.003)
    entropy_model.train()
    total_nll = 0.0
    total_bpp = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    for batch in pbar:
        batch = batch.to(device)
        B, T, C, H, W = batch.shape
        batch_flat = batch.view(B*T, C, H, W)
        
        with torch.no_grad():
            latent_norm = autoencoder.encode(batch_flat)
        
        quantized = quantizer(latent_norm)
        mu, log_scale = entropy_model(quantized)
        
        nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP, sigma_floor=sigma_floor)
        loss = nll  # no extra regularizer
        
        optimizer.zero_grad()
        loss.backward()
        
        if torch.isnan(loss):
            print(f"⚠️ NaN loss at epoch {epoch}, batch {num_batches}, skipping")
            continue
        
        torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 1.0)
        optimizer.step()
        
        total_nll += nll.item()
        bpp = nll.item() / np.log(2)
        total_bpp += bpp
        num_batches += 1
        
        with torch.no_grad():
            scale = torch.nn.functional.softplus(log_scale) + sigma_floor
            scale = torch.clamp(scale, min=sigma_floor, max=5.0)
        
        pbar.set_postfix(
            bpp=f"{bpp:.4f}",
            scale=f"{scale.mean().item():.4f}"
        )
    
    scheduler.step()
    avg_bpp = total_bpp / num_batches
    print(f"Epoch {epoch}: Avg BPP = {avg_bpp:.4f}")
    
    if epoch % 10 == 0 or epoch == EPOCHS:
        torch.save(entropy_model.state_dict(), f'{checkpoint_dir}/entropy_epoch{epoch}.pt')
        with open(f"{checkpoint_dir}/epoch.txt", 'w') as f:
            f.write(str(epoch))

torch.save(entropy_model.state_dict(), f'{checkpoint_dir}/entropy_final.pt')
print("✅ Contextual entropy model training complete.")
