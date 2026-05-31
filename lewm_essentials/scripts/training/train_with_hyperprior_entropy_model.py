#!/usr/bin/env python3
"""
Corrected joint training with a hyperprior entropy model.
Aims to produce variable bitrate across λ values.
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

# ---------- Autoencoder ----------
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

# ---------- Hyperprior Entropy Model ----------
class Hyperprior(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=256):
        super().__init__()
        self.hyperprior_cnn = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hyper_channels, latent_dim * 2, 3, padding=1),
        )
    def forward(self, x):
        return self.hyperprior_cnn(x)

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
if not video_paths:
    raise FileNotFoundError("No videos found")
dataset = VideoDataset(video_paths, frame_size=(256,256), frames_per_clip=4)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

val_size = max(1, int(len(dataset) * 0.1))
_, val_dataset = random_split(dataset, [len(dataset)-val_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

# ---------- Loss functions ----------
criterion_mse = nn.MSELoss()
try:
    import lpips
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    use_perceptual = True
    print("LPIPS perceptual loss enabled")
except:
    use_perceptual = False
    print("LPIPS not available, using MSE only")

QUANT_STEP = 2.0 / 255

def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.003, epsilon=1e-9):
    """Compute negative log-likelihood for discrete Laplace distribution.
    Returns: nats per element (mean over all elements)"""
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    laplace = Laplace(mu, scale)
    cdf_upper = laplace.cdf(y + 0.5 * step)
    cdf_lower = laplace.cdf(y - 0.5 * step)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    nll = -torch.log(pmf)
    return nll.mean()  # Mean over all elements

def quantize_with_temp(x, step, temp):
    """Soft quantization with temperature annealing"""
    x_quant = torch.round(x / step) * step
    return x_quant + (x - x_quant.detach()) * temp

@torch.no_grad()
def validate(autoencoder, entropy_model, val_loader, lam):
    """Validation loop to compute true metrics"""
    autoencoder.eval()
    entropy_model.eval()
    total_loss = 0
    total_rate_bpp = 0
    total_mse = 0
    num_batches = 0
    
    for batch in val_loader:
        batch = batch.to(device)
        B, T, C, H, W = batch.shape
        batch_flat = batch.view(B*T, C, H, W)
        
        # Encode
        latent_norm = autoencoder.encode(batch_flat)
        
        # Quantize (deterministic for validation)
        quantized = torch.round(latent_norm / QUANT_STEP) * QUANT_STEP
        
        # Get hyperprior parameters
        params = entropy_model(quantized)
        mu = params[:, :192, :, :]
        log_scale = params[:, 192:, :, :]
        
        # Compute likelihood
        nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP)
        
        # CORRECTED: Convert nats per latent element to bits per original pixel
        nats_per_latent = nll
        bits_per_latent = nats_per_latent / np.log(2)
        
        # Calculate compression ratio between latent and original
        latent_elements = quantized.numel()
        original_pixels = batch_flat.numel()
        latent_to_original_ratio = latent_elements / original_pixels
        
        rate_bpp = bits_per_latent * latent_to_original_ratio
        
        # Decode
        recon = autoencoder.decode(quantized, target_size=(H, W))
        recon = recon.view(B, T, C, H, W)
        
        # Distortion
        mse = criterion_mse(recon, batch)
        
        # Loss
        loss = lam * rate_bpp + mse
        
        total_loss += loss.item()
        total_rate_bpp += rate_bpp.item()
        total_mse += mse.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_rate = total_rate_bpp / num_batches
    avg_mse = total_mse / num_batches
    psnr = 10 * np.log10(1.0 / avg_mse) if avg_mse > 0 else 0
    
    return avg_loss, avg_rate, avg_mse, psnr

# ---------- Training for each λ ----------
lambda_list = [0.001, 0.01, 0.1, 1.0, 10.0]
output_dir = '/root/le-maia/checkpoints_rd_scratch'
os.makedirs(output_dir, exist_ok=True)

# Load Phase 0 autoencoder (initial weights)
ae_phase0 = '/root/le-maia/checkpoints_rd_scratch/autoencoder_final.pt'

num_epochs = 13  # Only 13 epochs per run

for lam in lambda_list:
    print(f"\n{'='*60}")
    print(f"Training with λ = {lam}")
    print(f"{'='*60}")
    
    # Initialize models
    autoencoder = VideoAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load(ae_phase0, map_location=device, weights_only=False))
    autoencoder.train()
    
    entropy_model = Hyperprior(latent_dim=192, hyper_channels=256).to(device)
    quantizer = Quantizer(num_levels=256, mode='training').to(device)
    
    # Optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': autoencoder.parameters(), 'lr': 1e-5},
        {'params': entropy_model.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-6)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_psnr = 0
    
    for epoch in range(1, num_epochs + 1):
        # Temperature annealing (less aggressive)
        temp = max(0.3, 1.0 - epoch * (0.7 / num_epochs))  # Decays to 0.3 by final epoch
        
        autoencoder.train()
        entropy_model.train()
        
        total_loss = 0
        total_rate_bpp = 0
        total_mse = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"λ={lam} Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            B, T, C, H, W = batch.shape
            batch_flat = batch.view(B*T, C, H, W)
            
            # Encode
            latent_norm = autoencoder.encode(batch_flat)
            
            # Quantize with temperature
            quantized = quantize_with_temp(latent_norm, QUANT_STEP, temp)
            
            # Get hyperprior parameters
            params = entropy_model(quantized)
            mu = params[:, :192, :, :]
            log_scale = params[:, 192:, :, :]
            
            # Compute likelihood (nats per latent element)
            nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP)
            
            # CORRECTED: Convert to bits per original pixel
            bits_per_latent = nll / np.log(2)
            
            # Calculate compression ratio between latent and original
            latent_elements = quantized.numel()
            original_pixels = batch_flat.numel()
            latent_to_original_ratio = latent_elements / original_pixels
            
            rate_bpp = bits_per_latent * latent_to_original_ratio
            
            # Decode
            recon = autoencoder.decode(quantized, target_size=(H, W))
            recon = recon.view(B, T, C, H, W)
            
            # Distortion (always use perceptual if available, no sudden switch)
            mse = criterion_mse(recon, batch)
            
            if use_perceptual:
                # Use perceptual loss from the beginning, but with smaller weight
                recon_4d = recon.view(B*T, C, H, W)
                batch_4d = batch.view(B*T, C, H, W)
                perceptual = perceptual_loss_fn(recon_4d * 2 - 1, batch_4d * 2 - 1).mean()
                distortion = mse + 0.05 * perceptual  # Reduced weight for stability
            else:
                distortion = mse
            
            # Rate-distortion loss
            loss = lam * rate_bpp + distortion
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_rate_bpp += rate_bpp.item()
            total_mse += mse.item()
            num_batches += 1
            
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                rate=f"{rate_bpp.item():.4f}",
                mse=f"{mse.item():.6f}",
                temp=f"{temp:.3f}"
            )
        
        scheduler.step()
        
        # Training metrics
        avg_loss = total_loss / num_batches
        avg_rate = total_rate_bpp / num_batches
        avg_mse = total_mse / num_batches
        psnr = 10 * np.log10(1.0 / avg_mse) if avg_mse > 0 else 0
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Rate={avg_rate:.4f} bpp, "
              f"MSE={avg_mse:.6f}, PSNR={psnr:.2f} dB, Temp={temp:.3f}")
        
        # Validation every epoch (since only 13 epochs)
        val_loss, val_rate, val_mse, val_psnr = validate(
            autoencoder, entropy_model, val_loader, lam
        )
        print(f"  Val: Loss={val_loss:.4f}, Rate={val_rate:.4f} bpp, "
              f"MSE={val_mse:.6f}, PSNR={val_psnr:.2f} dB")
        
        # Save best model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(autoencoder.state_dict(), 
                      f'{output_dir}/ae_lambda_{lam}_best.pt')
            torch.save(entropy_model.state_dict(), 
                      f'{output_dir}/entropy_lambda_{lam}_best.pt')
            print(f"  Best model saved (PSNR: {val_psnr:.2f} dB)")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs:
            torch.save(autoencoder.state_dict(), 
                      f'{output_dir}/ae_lambda_{lam}_epoch{epoch}.pt')
            torch.save(entropy_model.state_dict(), 
                      f'{output_dir}/entropy_lambda_{lam}_epoch{epoch}.pt')
    
    # Save final models
    torch.save(autoencoder.state_dict(), f'{output_dir}/ae_lambda_{lam}_final.pt')
    torch.save(entropy_model.state_dict(), f'{output_dir}/entropy_lambda_{lam}_final.pt')
    
    print(f"Finished λ={lam}. Best validation PSNR: {best_val_psnr:.2f} dB")
    print(f"{'='*60}\n")
