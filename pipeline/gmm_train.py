#!/usr/bin/env python3
"""
GMM entropy model — per-λ joint training of autoencoder + 2-component GMM.
Usage:
    python3 gmm_train.py               # train all 6 λ models
    python3 gmm_train.py --evaluate    # evaluate final checkpoints
"""

import os
import sys
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.distributions.normal import Normal
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Same autoencoder as pipeline ----------
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

# ---------- GMM Entropy Model (2 components) ----------
class GMMEntropyModel(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=256, num_components=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        self.hyperprior = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(hyper_channels, hyper_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hyper_channels, latent_dim * num_components * 3, 3, padding=1),
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        params = self.hyperprior(x)
        B, C, H, W = params.shape
        channels_per_comp = C // self.num_components
        params = params.view(B, self.num_components, channels_per_comp, H, W)
        mu = params[:, :, :self.latent_dim, :, :]
        log_scale = params[:, :, self.latent_dim:2*self.latent_dim, :, :]
        log_weight = params[:, :, 2*self.latent_dim:3*self.latent_dim, :, :]
        scale = self.softplus(log_scale) + 1e-5
        weight = torch.softmax(log_weight, dim=1)
        return mu, scale, weight

def gmm_likelihood_discrete(y, mu, scale, weight, step, epsilon=1e-9):
    B, C, H, W = y.shape
    num_comp = mu.shape[1]
    y_expanded = y.unsqueeze(1).expand(-1, num_comp, -1, -1, -1)
    normal = Normal(mu, scale)
    cdf_upper = normal.cdf(y_expanded + 0.5 * step)
    cdf_lower = normal.cdf(y_expanded - 0.5 * step)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    mixture_pmf = (weight * pmf).sum(dim=1)
    nll = -torch.log(mixture_pmf)
    return nll.mean()

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

BASE_DIR = os.environ.get('LEWM_BASE', '/root/le-maia')
DATASET_DIR = os.environ.get('LEWM_DATASET', os.path.join(BASE_DIR, 'datasets/pevid-hd'))
CHECKPOINT_DIR = os.environ.get('LEWM_CHECKPOINT_DIR', os.path.join(BASE_DIR, 'checkpoints_gmm'))
PHASE0_CKPT = os.environ.get('LEWM_PHASE0_CKPT', os.path.join(BASE_DIR, 'checkpoints_phase0/autoencoder_final.pt'))

video_paths = glob.glob(os.path.join(DATASET_DIR, '*.mpg'))
if not video_paths:
    raise FileNotFoundError(f"No videos found in {DATASET_DIR}")
dataset = VideoDataset(video_paths, frame_size=(256,256), frames_per_clip=4)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

val_size = max(1, int(len(dataset) * 0.1))
train_dataset, val_dataset = random_split(dataset, [len(dataset)-val_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

# ---------- Loss ----------
criterion_mse = nn.MSELoss()
QUANT_STEP = 2.0 / 255

def quantize_with_temp(x, step, temp):
    x_quant = torch.round(x / step) * step
    return x_quant + (x - x_quant.detach()) * temp

# ---------- Per-λ training ----------
lambda_list = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
output_dir = CHECKPOINT_DIR
os.makedirs(output_dir, exist_ok=True)

def train():
    for LAMBDA in lambda_list:
        print(f"\n========== GMM training with λ = {LAMBDA} ==========")
        autoencoder = VideoAutoencoder().to(device)
        if os.path.exists(PHASE0_CKPT):
            autoencoder.load_state_dict(torch.load(PHASE0_CKPT, map_location=device, weights_only=False), strict=False)
            print("Loaded Phase 0 autoencoder")
        autoencoder.train()

        entropy_model = GMMEntropyModel().to(device)
        quantizer = Quantizer(num_levels=256, mode='training').to(device)

        optimizer = optim.AdamW([
            {'params': autoencoder.parameters(), 'lr': 5e-5},
            {'params': entropy_model.parameters(), 'lr': 1e-4},
        ], weight_decay=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        best_val_loss = float('inf')
        patience = 10
        no_improve = 0

        for epoch in range(1, 31):
            temp = max(0.1, 1.0 - epoch * 0.03)
            autoencoder.train()
            entropy_model.train()
            total_loss = 0
            total_rate = 0
            total_mse = 0
            num_batches = 0
            pbar = tqdm(dataloader, desc=f"GMM λ={LAMBDA} Epoch {epoch}/30")
            for batch in pbar:
                batch = batch.to(device)
                optimizer.zero_grad()
                B, T, C, H, W = batch.shape
                batch_flat = batch.view(B*T, C, H, W)
                latent_norm = autoencoder.encode(batch_flat)
                quantized = quantize_with_temp(latent_norm, QUANT_STEP, temp)
                mu, scale, weight = entropy_model(quantized)
                nll = gmm_likelihood_discrete(quantized, mu, scale, weight, step=QUANT_STEP)
                rate_per_pixel = (nll * quantized.numel() / np.log(2)) / batch_flat.numel()
                recon = autoencoder.decode(quantized, target_size=(H, W))
                recon = recon.view(B, T, C, H, W)
                mse = criterion_mse(recon, batch)
                loss = LAMBDA * rate_per_pixel + mse
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
            # Validation every 5 epochs
            if epoch % 5 == 0:
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
                        mu_val, scale_val, weight_val = entropy_model(quant_val)
                        nll_val = gmm_likelihood_discrete(quant_val, mu_val, scale_val, weight_val, step=QUANT_STEP)
                        rate_val = (nll_val * quant_val.numel() / np.log(2)) / val_flat.numel()
                        recon_val = autoencoder.decode(quant_val, target_size=(Hv, Wv))
                        recon_val = recon_val.view(Bv, Tv, Cv, Hv, Wv)
                        val_mse += criterion_mse(recon_val, val_batch).item()
                        val_rate += rate_val.item()
                        val_batches += 1
                val_psnr = 10 * np.log10(1.0 / (val_mse / val_batches))
                val_bpp = val_rate / val_batches
                val_loss = (val_mse / val_batches) + LAMBDA * (val_rate / val_batches)
                print(f"  Val: PSNR={val_psnr:.2f} dB, BPP={val_bpp:.4f}, Loss={val_loss:.6f}")
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    no_improve = 0
                    torch.save(autoencoder.state_dict(), f'{output_dir}/ae_lambda_{LAMBDA}_best.pt')
                    torch.save(entropy_model.state_dict(), f'{output_dir}/entropy_lambda_{LAMBDA}_best.pt')
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"  Early stop at epoch {epoch}")
                        break
                autoencoder.train()
                entropy_model.train()
            if epoch % 10 == 0:
                torch.save(autoencoder.state_dict(), f'{output_dir}/ae_lambda_{LAMBDA}_epoch{epoch}.pt')
                torch.save(entropy_model.state_dict(), f'{output_dir}/entropy_lambda_{LAMBDA}_epoch{epoch}.pt')
        torch.save(autoencoder.state_dict(), f'{output_dir}/ae_lambda_{LAMBDA}_final.pt')
        torch.save(entropy_model.state_dict(), f'{output_dir}/entropy_lambda_{LAMBDA}_final.pt')
        print(f"Finished GMM λ={LAMBDA}.")

# ---------- Evaluation ----------
def evaluate():
    test_video = glob.glob(os.path.join(DATASET_DIR, '*.mpg'))[0]
    cap = cv2.VideoCapture(test_video)
    frames = []
    target_size = (256, 256)
    while len(frames) < 150:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames")

    quantizer_eval = Quantizer(num_levels=256, mode='inference').to(device)

    for LAMBDA in lambda_list:
        ae_ckpt = os.path.join(output_dir, f'ae_lambda_{LAMBDA}_final.pt')
        ent_ckpt = os.path.join(output_dir, f'entropy_lambda_{LAMBDA}_final.pt')
        if not os.path.exists(ae_ckpt) or not os.path.exists(ent_ckpt):
            print(f"Missing checkpoints for λ={LAMBDA}, skipping")
            continue
        autoencoder = VideoAutoencoder().to(device)
        autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=False), strict=False)
        autoencoder.eval()
        entropy_model = GMMEntropyModel().to(device)
        entropy_model.load_state_dict(torch.load(ent_ckpt, map_location=device, weights_only=False))
        entropy_model.eval()
        total_bits = 0
        total_mse = 0
        for frame in tqdm(frames, desc=f"GMM λ={LAMBDA}"):
            frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
            frame_t = frame_t.to(device)
            with torch.no_grad():
                latent_norm = autoencoder.encode(frame_t)
                quantized = quantizer_eval(latent_norm)
                mu, scale, weight = entropy_model(quantized)
                nll = gmm_likelihood_discrete(quantized, mu, scale, weight, step=QUANT_STEP)
                bits = nll * quantized.numel() / np.log(2)
                total_bits += bits.item()
                recon = autoencoder.decode(quantized, target_size=target_size)
                mse = torch.nn.functional.mse_loss(recon, frame_t).item()
                total_mse += mse
        bpp = total_bits / (len(frames) * target_size[0] * target_size[1])
        psnr = 20 * np.log10(1.0 / np.sqrt(total_mse / len(frames)))
        print(f"GMM λ={LAMBDA}: bpp={bpp:.4f}, PSNR={psnr:.2f} dB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true', help='Evaluate final checkpoints')
    args = parser.parse_args()
    if args.evaluate:
        evaluate()
    else:
        train()
