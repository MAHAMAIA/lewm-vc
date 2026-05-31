#!/usr/bin/env python3
"""
Experiment 1: Fine-tune best checkpoint with λ=50 to force bitrate down.
Loads ae_lambda_0.05_best.pt and continues training with high rate weight.
"""

import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.distributions.laplace import Laplace
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Model definitions (same as training) ----------
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

class VideoAutoencoder(nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.affine = AffineNormalization(latent_dim)
    def encode(self, x):
        latent = self.encoder(x, return_surprise=False)
        return self.affine(latent)
    def decode(self, latent_norm, target_size):
        return self.decoder(latent_norm, target_size=target_size)

class CheckerboardContext(nn.Module):
    def __init__(self, channels, hidden_dim=128):
        super().__init__()
        self.mask_conv = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.refine = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1),
        )
        self.register_buffer('mask', None)
    def forward(self, x, full_context=False):
        if self.mask is None or self.mask.shape[2:] != x.shape[2:]:
            h, w = x.shape[2], x.shape[3]
            mask = torch.zeros(1, 1, h, w, device=x.device)
            mask[..., 0::2, 0::2] = 1
            mask[..., 1::2, 1::2] = 1
            self.mask = mask
        out = self.mask_conv(x)
        out = self.refine(out)
        if not full_context:
            out = out * self.mask
        return out

class ContextualEntropyModel(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=512, context_hidden=128):
        super().__init__()
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
        self.head = nn.Conv2d(hyper_channels, latent_dim * 2, 1)
        self.context = CheckerboardContext(latent_dim, context_hidden)
        self.refine_mu = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        self.refine_scale = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
    def forward(self, x):
        x_down = self.down(x)
        x_up = self.up(x_down)
        x_skip = nn.functional.interpolate(x, size=x_up.shape[2:], mode='bilinear', align_corners=False)
        x_skip = self.skip_proj(x_skip)
        features = x_up + x_skip
        base_params = self.head(features)
        mu_base = base_params[:, :192, :, :]
        log_scale_base = base_params[:, 192:, :, :]
        ctx = self.context(x)
        mu_offset = self.refine_mu(ctx)
        scale_offset = self.refine_scale(ctx)
        mu = mu_base + mu_offset
        log_scale = log_scale_base + scale_offset
        return mu, log_scale

# ---------- Dataset (same as training) ----------
class VideoDataset(Dataset):
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
if not video_paths:
    raise FileNotFoundError("No videos found")
dataset = VideoDataset(video_paths, frame_size=(256,256), frames_per_clip=4)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

# Validation split
val_size = max(1, int(len(dataset) * 0.1))
_, val_dataset = random_split(dataset, [len(dataset)-val_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

# ---------- Load best checkpoint ----------
checkpoint_dir = '/root/le-maia/checkpoints_joint_phase0'
ae_ckpt = os.path.join(checkpoint_dir, 'ae_lambda_0.05_best.pt')
ent_ckpt = os.path.join(checkpoint_dir, 'entropy_lambda_0.05_best.pt')

if not os.path.exists(ae_ckpt) or not os.path.exists(ent_ckpt):
    raise FileNotFoundError(f"Best checkpoints not found")

autoencoder = VideoAutoencoder().to(device)
autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=False), strict=False)
autoencoder.train()

entropy_model = ContextualEntropyModel().to(device)
state = torch.load(ent_ckpt, map_location=device, weights_only=False)
for key in list(state.keys()):
    if 'mask' in key:
        del state[key]
entropy_model.load_state_dict(state, strict=False)
entropy_model.train()

quantizer = Quantizer(num_levels=256, mode='training').to(device)

# ---------- Loss functions ----------
criterion_mse = nn.MSELoss()
try:
    import lpips
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    use_perceptual = True
except:
    use_perceptual = False

QUANT_STEP = 2.0 / 255

def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.003, epsilon=1e-9):
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    laplace = Laplace(mu, scale)
    cdf_upper = laplace.cdf(y + 0.5 * step)
    cdf_lower = laplace.cdf(y - 0.5 * step)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    nll = -torch.log(pmf)
    return nll.mean()

def quantize_with_temp(x, step, temp):
    x_quant = torch.round(x / step) * step
    return x_quant + (x - x_quant.detach()) * temp

# ---------- High λ fine‑tuning ----------
LAMBDA = 50.0  # much higher rate weight
EPOCHS = 20
optimizer = optim.AdamW([
    {'params': autoencoder.parameters(), 'lr': 1e-6},
    {'params': entropy_model.parameters(), 'lr': 1e-5},
], weight_decay=1e-6)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_loss = float('inf')
best_val_psnr = 0

for epoch in range(1, EPOCHS+1):
    temp = max(0.1, 1.0 - epoch * 0.045)  # temperature decay over 20 epochs
    autoencoder.train()
    entropy_model.train()
    total_loss = 0
    total_rate = 0
    total_mse = 0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"λ={LAMBDA} Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        B, T, C, H, W = batch.shape
        batch_flat = batch.view(B*T, C, H, W)
        latent_norm = autoencoder.encode(batch_flat)
        quantized = quantize_with_temp(latent_norm, QUANT_STEP, temp)
        mu, log_scale = entropy_model(quantized)
        nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP, sigma_floor=0.003)
        rate_per_pixel = (nll * quantized.numel() / np.log(2)) / batch_flat.numel()
        recon = autoencoder.decode(quantized, target_size=(H, W))
        recon = recon.view(B, T, C, H, W)
        mse = criterion_mse(recon, batch)
        if use_perceptual and epoch > 5:
            recon_4d = recon.view(B*T, C, H, W)
            batch_4d = batch.view(B*T, C, H, W)
            perceptual = perceptual_loss_fn(recon_4d*2-1, batch_4d*2-1).mean()
            distortion = mse + 0.1 * perceptual
        else:
            distortion = mse
        loss = LAMBDA * rate_per_pixel + distortion
        loss.backward()
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_rate += rate_per_pixel.item()
        total_mse += mse.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item(), rate=rate_per_pixel.item(), mse=mse.item())
    scheduler.step()
    avg_loss = total_loss / num_batches
    avg_rate = total_rate / num_batches
    avg_mse = total_mse / num_batches
    psnr = 10 * np.log10(1.0 / avg_mse) if avg_mse > 0 else 0
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Rate={avg_rate:.4f} bpp, MSE={avg_mse:.6f}, PSNR={psnr:.2f} dB")
    # Validation
    autoencoder.eval()
    entropy_model.eval()
    val_mse = 0
    val_rate = 0
    val_batches = 0
    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = val_batch.to(device)
            Bv, Tv, Cv, Hv, Wv = val_batch.shape
            val_flat = val_batch.view(Bv*Tv, Cv, Hv, Wv)
            latent_val = autoencoder.encode(val_flat)
            quant_val = quantize_with_temp(latent_val, QUANT_STEP, temp=0.0)
            mu_val, log_scale_val = entropy_model(quant_val)
            nll_val = laplace_likelihood_discrete(quant_val, mu_val, log_scale_val, step=QUANT_STEP, sigma_floor=0.003)
            rate_val = (nll_val * quant_val.numel() / np.log(2)) / val_flat.numel()
            recon_val = autoencoder.decode(quant_val, target_size=(Hv, Wv))
            recon_val = recon_val.view(Bv, Tv, Cv, Hv, Wv)
            val_mse += criterion_mse(recon_val, val_batch).item()
            val_rate += rate_val.item()
            val_batches += 1
    val_psnr = 10 * np.log10(1.0 / (val_mse / val_batches))
    val_bpp = val_rate / val_batches
    print(f"  Val: PSNR={val_psnr:.2f} dB, BPP={val_bpp:.4f}")
    if val_psnr > best_val_psnr:
        best_val_psnr = val_psnr
        torch.save(autoencoder.state_dict(), f'/root/le-maia/checkpoints_finetuned/ae_lambda_{LAMBDA}_best.pt')
        torch.save(entropy_model.state_dict(), f'/root/le-maia/checkpoints_finetuned/entropy_lambda_{LAMBDA}_best.pt')
    autoencoder.train()
    entropy_model.train()

print(f"Fine-tuning complete. Best validation PSNR: {best_val_psnr:.2f} dB")
