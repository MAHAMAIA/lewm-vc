#!/usr/bin/env python3
"""
Train the JEPA predictor on frozen autoencoder latents.
FIXED: Keep batch dimension for predictor (4D tensors).
"""

import os, sys, glob, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2, numpy as np
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.predictor import LeWMPredictor
from lewm_vc.working_decoder import LeWMDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16
print(f"Device: {device}, Dtype: {dtype}")

# ---------- Autoencoder (frozen) ----------
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
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.affine = AffineNormalization(latent_dim)
    def encode(self, x):
        latent = self.encoder(x, return_surprise=False)
        return self.affine(latent)

# ---------- Dataset: Raw Frames ----------
class FrameSequenceDataset(Dataset):
    def __init__(self, video_paths, frame_size=(256,256), seq_len=4):
        self.video_paths = video_paths
        self.frame_size = frame_size
        self.seq_len = seq_len
        print("Extracting frame sequences...")
        self.sequences = []
        for vp in tqdm(video_paths):
            cap = cv2.VideoCapture(vp)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.resize(frame, frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                frames.append(torch.from_numpy(frame).permute(2,0,1))
            cap.release()
            for i in range(0, len(frames) - seq_len, seq_len // 2):
                self.sequences.append(torch.stack(frames[i:i+seq_len]))
        print(f"Extracted {len(self.sequences)} sequences")
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx]

# ---------- Load Frozen Autoencoder ----------
print("Loading autoencoder...")
ae_ckpt = '/root/le-maia/checkpoints_corrected/ae_lambda_0.01_final.pt'
autoencoder = VideoAutoencoderWithAffine(latent_dim=192).to(device).to(dtype)
autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device), strict=False)
autoencoder.eval()
for p in autoencoder.parameters():
    p.requires_grad = False
print("✅ Autoencoder frozen")

# ---------- Dataset ----------
video_paths = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')
dataset = FrameSequenceDataset(video_paths, seq_len=4)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

# ---------- Predictor ----------
predictor = LeWMPredictor(
    latent_dim=192,
    hidden_dim=256,
    num_layers=4,
    num_heads=4,
    context_len=3
).to(device).to(dtype)

optimizer = optim.AdamW(predictor.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.MSELoss()

# ---------- Training ----------
EPOCHS = 20
output_dir = '/root/le-maia/checkpoints'
os.makedirs(output_dir, exist_ok=True)

for epoch in range(1, EPOCHS+1):
    predictor.train()
    autoencoder.eval()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for frames in pbar:
        frames = frames.to(device).to(dtype)  # (B, 4, 3, 256, 256)
        B, T, C, H, W = frames.shape
        with torch.no_grad():
            latents = []
            for t in range(T):
                lat = autoencoder.encode(frames[:, t])  # (B, 192, 16, 16)
                latents.append(lat)
        # Process each batch item
        batch_loss = 0
        for b in range(B):
            # ✅ FIX: add batch dimension with [b:b+1] -> (1, 192, 16, 16)
            seq = [latents[t][b:b+1] for t in range(T)]
            pred_mean, _ = predictor(seq[:3])
            loss = criterion(pred_mean, seq[3])
            batch_loss += loss
        loss = batch_loss / B
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.6f}")
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}: Avg Loss = {avg_loss:.6f}")
    if epoch % 5 == 0:
        torch.save(predictor.state_dict(), f'{output_dir}/predictor_epoch{epoch}.pt')

torch.save(predictor.state_dict(), f'{output_dir}/predictor_final.pt')
print("✅ Predictor training complete")
