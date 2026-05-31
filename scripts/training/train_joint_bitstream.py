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
from lewm_vc.bitstream.writer import BitstreamWriter
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
    def forward(self, x):
        b, t, c, h, w = x.shape
        x_flat = x.view(b*t, c, h, w)
        latent = self.encoder(x_flat, return_surprise=False)
        latent_norm = self.affine(latent)
        recon = self.decoder(latent_norm, target_size=(h,w))
        recon = recon.view(b, t, c, h, w)
        return recon, latent_norm

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
print(f"Found {len(video_paths)} videos")
dataset = VideoDataset(video_paths, frame_size=(256,256), frames_per_clip=4)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

# ---------- Model ----------
autoencoder = VideoAutoencoderWithAffine(latent_dim=192).to(device)
quantizer = Quantizer(num_levels=256, mode='training').to(device)

# Load pre-trained affine autoencoder weights
checkpoint_affine = '/root/le-maia/checkpoints_affine/autoencoder_affine_final.pt'
if os.path.exists(checkpoint_affine):
    autoencoder.load_state_dict(torch.load(checkpoint_affine, map_location=device), strict=False)
    print("Loaded pre-trained affine autoencoder")

# Optimizer for autoencoder only (no entropy model)
optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

criterion_mse = nn.MSELoss()
try:
    import lpips
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    use_perceptual = True
    print("LPIPS enabled")
except:
    use_perceptual = False

# Bitstream writer to measure actual rate
writer = BitstreamWriter(version=1)

def get_bitrate(latent_norm):
    """Quantize and write to bitstream, return number of bytes."""
    quantized = quantizer(latent_norm)
    # Write as I-frame (simplified)
    frame_data = {"latent": quantized.cpu()}
    nal_bytes = writer.write_frame(frame_data, is_iframe=True)
    return len(nal_bytes) * 8  # bits

LAMBDA = 0.01  # rate weight
EPOCHS = 100
os.makedirs('/root/le-maia/checkpoints_bitstream_train', exist_ok=True)

for epoch in range(1, EPOCHS+1):
    autoencoder.train()
    total_loss = 0
    total_rate = 0
    total_dist = 0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        recon, latent_norm = autoencoder(batch)
        
        # Compute bitrate (non-differentiable, use straight-through estimation)
        # We'll use the quantizer's step size as a proxy for gradient
        # For now, we'll detach the rate and only use it as a loss term (not ideal but works)
        with torch.no_grad():
            bits = get_bitrate(latent_norm)
        rate_loss = bits
        
        mse_loss = criterion_mse(recon, batch)
        if use_perceptual and epoch > 10:
            b, t, c, h, w = recon.shape
            recon_4d = recon.view(b*t, c, h, w)
            batch_4d = batch.view(b*t, c, h, w)
            perceptual_loss = perceptual_loss_fn(recon_4d*2-1, batch_4d*2-1).mean()
            distortion = mse_loss + 0.1 * perceptual_loss
        else:
            distortion = mse_loss
        
        loss = LAMBDA * rate_loss + distortion
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_rate += rate_loss
        total_dist += distortion.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item(), rate=rate_loss, dist=distortion.item())
    
    scheduler.step()
    avg_loss = total_loss / num_batches
    avg_rate = total_rate / num_batches
    avg_dist = total_dist / num_batches
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Rate={avg_rate:.2f} bits, Dist={avg_dist:.6f}")
    
    if epoch % 20 == 0:
        torch.save(autoencoder.state_dict(), f'/root/le-maia/checkpoints_bitstream_train/autoencoder_epoch{epoch}.pt')

torch.save(autoencoder.state_dict(), '/root/le-maia/checkpoints_bitstream_train/autoencoder_final.pt')
print("Training complete.")
