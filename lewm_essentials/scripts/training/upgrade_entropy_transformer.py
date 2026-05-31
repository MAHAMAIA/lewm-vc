#!/usr/bin/env python3
"""
FIXED Upgrade Script: Transformer Entropy Model + Content‑Adaptive Inference
- Correct BPP calculation matching training
- Proper quantization step (4.0/255)
- Fixed sigma clamping
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
import csv

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16
print(f"Device: {device}, Dtype: {dtype}")

# ============================================================================
# 1. Model Definitions (Autoencoder + Transformer Entropy Model)
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        return x + self.conv2(torch.nn.functional.gelu(self.norm2(
            self.conv1(torch.nn.functional.gelu(self.norm1(x))))))

class LeWMDecoder(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512, num_layers=4):
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
        x = torch.sigmoid(self.final(x))
        if target_size:
            x = torch.nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

class AffineNormalization(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x): return x * self.scale + self.shift

class VideoAutoencoderWithAffine(nn.Module):
    def __init__(self, latent_dim=192, decoder_layers=4):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim, num_layers=decoder_layers)
        self.affine = AffineNormalization(latent_dim)
    def encode(self, x):
        latent = self.encoder(x, return_surprise=False)
        return self.affine(latent)
    def decode(self, latent_norm, target_size):
        return self.decoder(latent_norm, target_size=target_size)

# ----------------------------------------------------------------------------
# Transformer-based Context Entropy Model (MaskCRT-style)
# ----------------------------------------------------------------------------
class TransformerContext(nn.Module):
    """Lightweight transformer to capture long-range spatial dependencies."""
    def __init__(self, channels, hidden_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.proj = nn.Conv2d(channels, hidden_dim, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, hidden_dim, 16, 16) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim*2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Conv2d(hidden_dim, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x) + self.pos_embed[:, :, :H, :W]
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.transformer(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        return self.out_proj(x)

class ContextualEntropyModelTransformer(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=1024, context_hidden=256):
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
        self.context = TransformerContext(latent_dim, hidden_dim=context_hidden)
        self.refine_mu = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        self.refine_scale = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)

    def forward(self, x):
        x_down = self.down(x)
        x_up = self.up(x_down)
        x_skip = self.skip_proj(nn.functional.interpolate(x, size=x_up.shape[2:], mode='bilinear', align_corners=False))
        base = self.head(x_up + x_skip)
        mu_b, sc_b = base[:, :192], base[:, 192:]
        ctx = self.context(x).to(x.dtype)
        return mu_b + self.refine_mu(ctx), sc_b + self.refine_scale(ctx)

# ============================================================================
# 2. Dataset (same as training)
# ============================================================================
class VideoDataset(Dataset):
    def __init__(self, video_paths, frame_size=(256,256), frames_per_clip=2):
        self.video_paths = video_paths
        self.frame_size = frame_size
        self.frames_per_clip = frames_per_clip
    def __len__(self): return len(self.video_paths) * 200
    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_paths[idx % len(self.video_paths)])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = np.random.randint(0, max(1, total - self.frames_per_clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frame = cv2.resize(frame, self.frame_size)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
        cap.release()
        return torch.from_numpy(np.stack(frames)).permute(0,3,1,2).to(dtype)

# ============================================================================
# 3. Loss Functions & Helpers (FIXED)
# ============================================================================
QUANT_STEP = 4.0 / 255        # ✅ MATCHES TRAINING
SIGMA_FLOOR = 0.01            # ✅ MATCHES TRAINING
SIGMA_MAX = 10.0              # ✅ MATCHES TRAINING

def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=SIGMA_FLOOR, epsilon=1e-9):
    """Exactly matches training."""
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    scale = torch.clamp(scale, min=sigma_floor, max=SIGMA_MAX)
    laplace = Laplace(mu, scale)
    pmf = torch.clamp(laplace.cdf(y + 0.5*step) - laplace.cdf(y - 0.5*step), min=epsilon, max=1.0)
    return -torch.log(pmf).mean()

def compute_bpp(nll, quantized, num_pixels):
    """BPP calculation exactly matching training."""
    return (nll.item() * quantized.numel() / np.log(2)) / num_pixels

# ============================================================================
# 4. Content-Adaptive Inference Helper
# ============================================================================
def adapt_entropy_biases(model, latent_sample, steps=20, lr=1e-2):
    """Quickly adapt biases of entropy model to the given latent sample."""
    model.train()
    biases = [p for n, p in model.named_parameters() if 'bias' in n and p.requires_grad]
    if not biases:
        return
    opt = torch.optim.Adam(biases, lr=lr)
    for _ in range(steps):
        mu, log_scale = model(latent_sample)
        loss = laplace_likelihood_discrete(latent_sample, mu, log_scale, step=QUANT_STEP)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

# ============================================================================
# 5. Main Upgrade Routine (FIXED BPP CALCULATION)
# ============================================================================
def upgrade_lambda(lam, autoencoder_ckpt, output_dir, fine_tune_epochs=30, batch_size=64):
    """Upgrade a single λ checkpoint with transformer entropy model."""
    print(f"\n{'='*60}\nUpgrading λ={lam}\n{'='*60}")
    
    # Load frozen autoencoder
    decoder_layers = 6 if lam >= 0.01 else 4
    autoencoder = VideoAutoencoderWithAffine(latent_dim=192, decoder_layers=decoder_layers).to(device).to(dtype)
    autoencoder.load_state_dict(torch.load(autoencoder_ckpt, map_location=device), strict=False)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False
    print(f"✅ Loaded autoencoder from {autoencoder_ckpt}")
    
    # Dataset
    video_paths = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')
    dataset = VideoDataset(video_paths, frame_size=(256,256), frames_per_clip=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2)
    
    quantizer_train = Quantizer(num_levels=256, mode='training').to(device)
    
    # Initialize transformer entropy model
    entropy_model = ContextualEntropyModelTransformer(latent_dim=192, hyper_channels=1024).to(device).to(dtype)
    optimizer = optim.AdamW(entropy_model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs)
    
    # Fine-tune entropy model
    for epoch in range(1, fine_tune_epochs+1):
        entropy_model.train()
        total_bpp = 0
        n_batches = 0
        pbar = tqdm(dataloader, desc=f"Entropy FT λ={lam} Epoch {epoch}/{fine_tune_epochs}")
        for batch in pbar:
            batch = batch.to(device)
            with torch.no_grad():
                b, t, c, h, w = batch.shape
                x_flat = batch.view(b*t, c, h, w)
                latent_norm = autoencoder.encode(x_flat)
                quantized = quantizer_train(latent_norm)
            mu, log_scale = entropy_model(quantized)
            nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP)
            loss = nll
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 1.0)
            optimizer.step()
            
            # ✅ FIXED: BPP calculation matches training
            num_pixels = x_flat.numel()
            bpp = (nll.item() * quantized.numel() / np.log(2)) / num_pixels
            total_bpp += bpp
            n_batches += 1
            pbar.set_postfix(bpp=f"{bpp:.4f}")
        scheduler.step()
        avg_bpp = total_bpp / n_batches
        print(f"Epoch {epoch}: Avg BPP = {avg_bpp:.4f}")
    
    # Save upgraded model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(entropy_model.state_dict(), f'{output_dir}/entropy_transformer_lambda_{lam}.pt')
    
    # Evaluate with content-adaptive inference
    print("\n📊 Running Content-Adaptive Evaluation...")
    test_video = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')[0]
    cap = cv2.VideoCapture(test_video)
    frames = []
    target_size = (256,256)
    while len(frames) < 150:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    entropy_model.eval()
    total_bits = 0
    total_mse = 0
    
    for frame in tqdm(frames, desc="Adaptive encoding"):
        frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
        frame_t = frame_t.to(device).to(dtype)
        with torch.no_grad():
            latent_norm = autoencoder.encode(frame_t)
            # ✅ FIXED: Hard quantization matching training
            quantized = torch.round(latent_norm / QUANT_STEP) * QUANT_STEP
        
        adapt_entropy_biases(entropy_model, quantized, steps=10, lr=1e-2)
        
        with torch.no_grad():
            mu, log_scale = entropy_model(quantized)
            nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP)
            # ✅ FIXED: BPP calculation matches training
            num_pixels = frame_t.numel()
            bits = nll.item() * quantized.numel() / np.log(2)
            total_bits += bits
            recon = autoencoder.decode(quantized, target_size=target_size)
            mse = torch.nn.functional.mse_loss(recon.float(), frame_t.float()).item()
            total_mse += mse
    
    avg_bpp = total_bits / (len(frames) * target_size[0] * target_size[1])
    avg_psnr = 10 * np.log10(1.0 / (total_mse / len(frames))) if total_mse > 0 else 100
    print(f"✅ Upgraded λ={lam}: BPP = {avg_bpp:.4f}, PSNR = {avg_psnr:.2f} dB")
    return lam, avg_bpp, avg_psnr

# ============================================================================
# 6. Run Upgrade on Best λ
# ============================================================================
if __name__ == "__main__":
    LAMBDA = 0.01
    CKPT_DIR = '/root/le-maia/checkpoints_corrected'
    OUTPUT_DIR = '/root/le-maia/checkpoints_upgraded'
    
    autoencoder_ckpt = f'{CKPT_DIR}/ae_lambda_{LAMBDA}_final.pt'
    if not os.path.exists(autoencoder_ckpt):
        print(f"❌ Checkpoint {autoencoder_ckpt} not found.")
        sys.exit(1)
    
    lam, bpp, psnr = upgrade_lambda(LAMBDA, autoencoder_ckpt, OUTPUT_DIR, fine_tune_epochs=30)
    
    with open(f'{OUTPUT_DIR}/upgrade_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['λ', 'BPP', 'PSNR'])
        writer.writerow([lam, bpp, psnr])
    print(f"\n🎉 Upgrade complete. Result saved to {OUTPUT_DIR}/upgrade_result.csv")
