import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Original autoencoder (no affine)
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

class VideoAutoencoder(nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
    def encode(self, x):
        return self.encoder(x, return_surprise=False)

# Load pre-trained autoencoder (frozen)
autoencoder = VideoAutoencoder(latent_dim=192).to(device)
checkpoint_orig = '/root/le-maia/checkpoints/autoencoder_final.pt'
autoencoder.load_state_dict(torch.load(checkpoint_orig, map_location=device))
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False
print("Original autoencoder loaded and frozen.")

# Dataset to generate latents on the fly
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

# Entropy model and quantizer
entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
quantizer = Quantizer(num_levels=256, mode='training').to(device)

# Use NLL loss with sigma clipping
def nll_loss(mu, sigma, x):
    sigma = torch.clamp(sigma, min=0.01)  # prevent sigma from becoming too small
    var = sigma ** 2
    loss = 0.5 * ((x - mu) ** 2 / var + torch.log(var) + np.log(2 * np.pi))
    return loss.mean()

optimizer = optim.AdamW(entropy_model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

EPOCHS = 100
os.makedirs('/root/le-maia/checkpoints_entropy_original', exist_ok=True)

for epoch in range(1, EPOCHS+1):
    entropy_model.train()
    total_loss = 0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        batch = batch.to(device)
        # Get latents from frozen autoencoder
        with torch.no_grad():
            B, T, C, H, W = batch.shape
            batch_flat = batch.view(B*T, C, H, W)
            latent = autoencoder.encode(batch_flat)  # [B*T, 192, 16, 16]
        # Quantize
        quantized = quantizer(latent)
        # Get mu, sigma
        params = entropy_model.hyperprior_cnn(quantized)
        mu = params[:, :192, :, :]
        log_sigma = params[:, 192:, :, :]
        sigma = torch.nn.functional.softplus(log_sigma) + 1e-5
        # NLL loss
        loss = nll_loss(mu, sigma, quantized)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item())
    scheduler.step()
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}: NLL loss = {avg_loss:.6f}")
    if epoch % 20 == 0:
        torch.save(entropy_model.state_dict(), f'/root/le-maia/checkpoints_entropy_original/entropy_epoch{epoch}.pt')

torch.save(entropy_model.state_dict(), '/root/le-maia/checkpoints_entropy_original/entropy_final.pt')
print("Entropy model training complete.")
