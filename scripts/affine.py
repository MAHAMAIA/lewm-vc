# retrain_with_affine.py
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

class AffineNormalization(nn.Module):
    """Per-channel affine transform to map latent to zero mean, unit variance."""
    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x):
        # x: [B, C, H, W]
        return x * self.scale + self.shift
    def inverse(self, y):
        return (y - self.shift) / self.scale

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
        # We could quantize here, but for training we just pass through
        recon = self.decoder(latent_norm, target_size=(h,w))
        recon = recon.view(b, t, c, h, w)
        return recon, latent_norm, self.affine.scale, self.affine.shift

# Load existing autoencoder
class VideoAutoencoder(nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)

autoencoder = VideoAutoencoder().to(device)
checkpoint_auto = '/root/le-maia/checkpoints/autoencoder_final.pt'
autoencoder.load_state_dict(torch.load(checkpoint_auto, map_location=device))
autoencoder.eval()
print("Loaded pre-trained autoencoder")

# Create new model with affine layer
model = VideoAutoencoderWithAffine(latent_dim=192).to(device)
# Copy encoder and decoder weights
model.encoder.load_state_dict(autoencoder.encoder.state_dict())
model.decoder.load_state_dict(autoencoder.decoder.state_dict())
# Affine layer is randomly initialized (scale=1, shift=0) – that's fine for fine-tuning
print("Initialized model with affine layer")

# Dataset
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
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

# Training setup
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.L1Loss()

# Optional perceptual loss
try:
    import lpips
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    use_perceptual = True
    print("LPIPS enabled")
except:
    use_perceptual = False

EPOCHS = 30
os.makedirs('/root/le-maia/checkpoints_affine', exist_ok=True)

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, _, _, _ = model(batch)
        loss_l1 = criterion(recon, batch)
        if use_perceptual and epoch > 5:
            b, t, c, h, w = recon.shape
            recon_4d = recon.view(b*t, c, h, w)
            batch_4d = batch.view(b*t, c, h, w)
            loss_perc = perceptual_loss_fn(recon_4d*2-1, batch_4d*2-1).mean()
            loss = loss_l1 + 0.1 * loss_perc
        else:
            loss = loss_l1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=loss.item())
    scheduler.step()
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}")
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'/root/le-maia/checkpoints_affine/autoencoder_affine_e{epoch}.pt')

torch.save(model.state_dict(), '/root/le-maia/checkpoints_affine/autoencoder_affine_final.pt')
print("Training complete.")
