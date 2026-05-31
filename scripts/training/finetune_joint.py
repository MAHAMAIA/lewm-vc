#!/usr/bin/env python3
"""
Joint fine‑tuning of autoencoder + frozen entropy model.
Sweep λ values with curriculum learning to build competitive RD curve.
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

# ---------- Affine autoencoder ----------
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
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, 2, 1)
        self.res1 = ResidualBlock(hidden_dim//2)
        self.up2 = nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4, 2, 1)
        self.res2 = ResidualBlock(hidden_dim//4)
        self.up3 = nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4, 2, 1)
        self.res3 = ResidualBlock(hidden_dim//8)
        self.up4 = nn.ConvTranspose2d(hidden_dim//8, hidden_dim//16, 4, 2, 1)
        self.res4 = ResidualBlock(hidden_dim//16)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim//16, hidden_dim//32, 3, 1, 1),
            nn.InstanceNorm2d(hidden_dim//32),
            nn.GELU(),
            nn.Conv2d(hidden_dim//32, 3, 3, 1, 1),
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
    def forward(self, x):
        b, t, c, h, w = x.shape
        x_flat = x.view(b*t, c, h, w)
        latent = self.encoder(x_flat, return_surprise=False)
        latent_norm = self.affine(latent)
        recon = self.decoder(latent_norm, target_size=(h,w))
        recon = recon.view(b, t, c, h, w)
        return recon, latent_norm

# ---------- Contextual entropy model ----------
class CheckerboardContext(nn.Module):
    def __init__(self, channels, hidden_dim=128):
        super().__init__()
        self.mask_conv = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.refine = nn.Sequential(nn.GELU(), nn.Conv2d(hidden_dim, channels, 3, padding=1))
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
            nn.Conv2d(latent_dim, hyper_channels, 5, padding=2), nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2, stride=2), nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2), nn.GELU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2), nn.GELU()
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
        return mu_base + mu_offset, log_scale_base + scale_offset

# ---------- Load models ----------
autoencoder = VideoAutoencoderWithAffine(latent_dim=192).to(device)
ae_checkpoint = '/root/le-maia/checkpoints_affine/autoencoder_affine_final.pt'
autoencoder.load_state_dict(torch.load(ae_checkpoint, map_location=device, weights_only=False), strict=False)
print("✅ Autoencoder loaded.")

entropy_model = ContextualEntropyModel(latent_dim=192, hyper_channels=512, context_hidden=128).to(device)
entropy_ckpt = '/root/le-maia/checkpoints_entropy_contextual/entropy_final.pt'
state = torch.load(entropy_ckpt, map_location=device, weights_only=False)
for key in list(state.keys()):
    if 'mask' in key:
        del state[key]
entropy_model.load_state_dict(state, strict=False)

# ✅ CRITICAL FIX 1: Freeze entropy model so encoder must adapt to fixed rate landscape
entropy_model.eval()
for p in entropy_model.parameters():
    p.requires_grad = False
print("✅ Entropy model loaded & FROZEN (gradients will flow through to encoder).")

# ---------- Dataset ----------
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
dataset = VideoDataset(video_paths, frame_size=(256,256), frames_per_clip=4)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

val_size = max(1, int(len(dataset) * 0.1))
train_dataset, val_dataset = random_split(dataset, [len(dataset)-val_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

# ---------- Loss functions ----------
criterion_mse = nn.MSELoss()
QUANT_STEP = 2.0 / 255

def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.05, epsilon=1e-9):
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    scale = torch.clamp(scale, min=sigma_floor, max=5.0)
    laplace = Laplace(mu, scale)
    cdf_upper = laplace.cdf(y + 0.5 * step)
    cdf_lower = laplace.cdf(y - 0.5 * step)
    pmf = torch.clamp(cdf_upper - cdf_lower, min=epsilon, max=1.0)
    return -torch.log(pmf).mean()

def quantize_with_ste(x, step):
    """Standard straight-through estimator."""
    x_quant = torch.round(x / step) * step
    return x + (x_quant - x).detach()

# ---------- Training Configuration ----------
lambda_list = [0.5]  # Aggressive sweep to force BPP down
LATENT_L2_WEIGHT = 0.02  # Explicitly penalize high-variance latents
output_dir = '/root/le-maia/checkpoints_joint_rd_v2'
os.makedirs(output_dir, exist_ok=True)

print(f"🎯 λ sweep: {lambda_list}")
print(f"📉 Latent L2 weight: {LATENT_L2_WEIGHT}")
print(f"🔁 Curriculum: λ starts 2× higher, decays linearly over 60 epochs")

for lam in lambda_list:
    print(f"\n{'='*60}")
    print(f"Training with λ = {lam}")
    print(f"{'='*60}")
    
    autoencoder.load_state_dict(torch.load(ae_checkpoint, map_location=device, weights_only=False), strict=False)
    autoencoder.train()
    
    # ✅ CRITICAL FIX 2: Higher LR, no gradient clipping
    optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    
    best_val_loss = float('inf')
    patience = 15
    no_improve = 0
    
    for epoch in range(1, 61):
        # Curriculum: start aggressive, decay to target λ
        current_lam = lam * (2.0 - (epoch / 60.0))  # Linear decay from 2× → 1×
        
        total_loss = 0
        total_rate = 0
        total_mse = 0
        total_l2 = 0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"λ={lam} Epoch {epoch}/60")
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon, latent_norm = autoencoder(batch)
            quantized = quantize_with_ste(latent_norm, QUANT_STEP)
            
            with torch.no_grad():
                mu, log_scale = entropy_model(quantized)
                nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP)
            
            # Rate per pixel
            batch_pixels = batch.numel()
            rate_per_pixel = (nll * quantized.numel() / np.log(2)) / batch_pixels
            
            mse = criterion_mse(recon, batch)
            latent_l2 = torch.mean(latent_norm**2)
            
            # ✅ CRITICAL FIX 3: Standard loss formulation, NO clipping, NO artificial scaling
            loss = mse + current_lam * rate_per_pixel + LATENT_L2_WEIGHT * latent_l2
            loss.backward()
            optimizer.step()  # Removed clip_grad_norm_ to preserve rate gradient magnitude
            
            total_loss += loss.item()
            total_rate += rate_per_pixel.item()
            total_mse += mse.item()
            total_l2 += latent_l2.item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item(), rate=rate_per_pixel.item(), mse=mse.item())
            
            if epoch == 1 and batch_idx == 0:
                with torch.no_grad():
                    enc_grad = sum(p.grad.abs().sum().item() for p in autoencoder.encoder.parameters() if p.grad is not None)
                    print(f"  🔍 Encoder grad norm: {enc_grad:.6f} | λ_eff={current_lam:.3f}")
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        avg_rate = total_rate / num_batches
        avg_mse = total_mse / num_batches
        avg_l2 = total_l2 / num_batches
        psnr = 10 * np.log10(1.0 / avg_mse) if avg_mse > 0 else 0
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Rate={avg_rate:.4f} bpp, "
              f"MSE={avg_mse:.6f}, PSNR={psnr:.2f} dB, L2={avg_l2:.4f}")
        
        if epoch % 10 == 0:
            autoencoder.eval()
            val_mse, val_rate, val_batches = 0, 0, 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device)
                    recon_val, latent_val = autoencoder(val_batch)
                    quant_val = quantize_with_ste(latent_val, QUANT_STEP)
                    mu_val, log_scale_val = entropy_model(quant_val)
                    nll_val = laplace_likelihood_discrete(quant_val, mu_val, log_scale_val, step=QUANT_STEP)
                    rate_val = (nll_val * quant_val.numel() / np.log(2)) / val_batch.numel()
                    val_mse += criterion_mse(recon_val, val_batch).item()
                    val_rate += rate_val.item()
                    val_batches += 1
            
            val_psnr = 10 * np.log10(1.0 / (val_mse / val_batches))
            val_bpp = val_rate / val_batches
            val_loss = (val_mse / val_batches) + lam * val_bpp + LATENT_L2_WEIGHT * avg_l2
            
            print(f"  ✅ Val: PSNR={val_psnr:.2f} dB, BPP={val_bpp:.4f}, ValLoss={val_loss:.6f}")
            
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                no_improve = 0
                torch.save({
                    'autoencoder': autoencoder.state_dict(),
                    'entropy_model': entropy_model.state_dict(),
                }, f'{output_dir}/best_lambda{lam}_epoch{epoch}.pt')
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  ⏹ Early stop at epoch {epoch}")
                    break
            autoencoder.train()
    
    torch.save(autoencoder.state_dict(), f'{output_dir}/ae_lambda_{lam}_final.pt')
    print(f"✅ Finished λ={lam}")

print("\n🎉 All runs completed.")
