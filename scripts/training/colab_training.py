# LeWM-VC GPU Training on Google Colab (Improved Version)
# Runtime: Runtime -> Change runtime type -> GPU
# Improvements: Skip connections, residual blocks, perceptual loss, higher resolution

# Install dependencies
!pip install torch torchvision numpy opencv-python tqdm lpips

# Clone the repo
!git clone https://github.com/thepreetam/le-maia.git
%cd le-maia

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Mount Google Drive and use local videos
print("Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

# Use videos from Google Drive
print("Using videos from /content/drive/My Drive/le-maia/datasets/pevid-hd")
VIDEO_DIR = '/content/drive/My Drive/le-maia/datasets/pevid-hd'

print(f"\nCUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

from lewm_vc.encoder import LeWMEncoder


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.nn.functional.gelu(self.norm1(x))
        x = self.conv1(x)
        x = torch.nn.functional.gelu(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class LeWMDecoder(nn.Module):
    """Improved decoder with residual blocks and deeper architecture."""

    def __init__(self, latent_dim=192, hidden_dim=512, output_channels=3):
        super().__init__()
        self.proj = nn.Conv2d(latent_dim, hidden_dim, kernel_size=1)

        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1)
        self.res1 = ResidualBlock(hidden_dim // 2)

        self.up2 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1)
        self.res2 = ResidualBlock(hidden_dim // 4)

        self.up3 = nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, 2, 1)
        self.res3 = ResidualBlock(hidden_dim // 8)

        self.up4 = nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, 4, 2, 1)
        self.res4 = ResidualBlock(hidden_dim // 16)

        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim // 16, hidden_dim // 32, 3, padding=1),
            nn.InstanceNorm2d(hidden_dim // 32),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 32, output_channels, 3, padding=1),
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


class VideoDataset(Dataset):
    def __init__(self, video_paths, frame_size=(512, 512), frames_per_clip=4):
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
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)

        cap.release()
        return torch.from_numpy(np.stack(frames)).float()


class VideoAutoencoder(nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x_flat = x.view(batch_size * num_frames, c, h, w)
        latent = self.encoder(x_flat, return_surprise=False)
        reconstructed = self.decoder(latent, target_size=(h, w))
        reconstructed = reconstructed.view(batch_size, num_frames, c, h, w)
        return reconstructed, latent


def train(epochs=100, batch_size=2, lr=1e-4, resolution=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    print(f"Resolution: {resolution}x{resolution}")

    import glob
    video_paths = glob.glob(f'{VIDEO_DIR}/*.mpg') + glob.glob(f'{VIDEO_DIR}/*.mp4') + glob.glob(f'{VIDEO_DIR}/*.avi')

    if not video_paths:
        print("No videos found!")
        return

    print(f"Found {len(video_paths)} videos: {video_paths}")

    dataset = VideoDataset(video_paths, frame_size=(resolution, resolution), frames_per_clip=4)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VideoAutoencoder(latent_dim=192).to(device)

    # Setup perceptual loss (LPIPS)
    try:
        import lpips
        perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
        use_perceptual = True
        print("LPIPS perceptual loss enabled")
    except Exception as e:
        print(f"LPIPS not available: {e}, using only L1 loss")
        perceptual_loss_fn = None
        use_perceptual = False

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    l1_loss_fn = nn.L1Loss()

    import os
    os.makedirs('checkpoints_improved', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_l1 = 0
        total_perc = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            reconstructed, latent = model(batch)

            # Combined loss: L1 + perceptual (if available)
            loss_l1 = l1_loss_fn(reconstructed, batch)

            if use_perceptual and perceptual_loss_fn is not None and epoch > 5:
                # LPIPS expects 4D: flatten batch and frames
                b, f, c, h, w = reconstructed.shape
                reconstructed_4d = reconstructed.view(b * f, c, h, w)
                batch_4d = batch.view(b * f, c, h, w)
                # LPIPS expects [-1, 1] range
                loss_perc = perceptual_loss_fn(reconstructed_4d * 2 - 1, batch_4d * 2 - 1).mean()
                loss = loss_l1 + 0.1 * loss_perc
                total_perc += loss_perc.item()
            else:
                loss_perc = torch.tensor(0.0, device=device)
                loss = loss_l1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_l1 += loss_l1.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'l1': f'{loss_l1.item():.4f}'})

        scheduler.step()
        avg_loss = total_loss / num_batches
        avg_l1 = total_l1 / num_batches
        avg_perc = total_perc / max(1, num_batches)

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, L1 = {avg_l1:.4f}, Perceptual = {avg_perc:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, f'checkpoints_improved/autoencoder_e{epoch+1}.pt')

            # Quick validation
            model.eval()
            with torch.no_grad():
                val_batch = next(iter(dataloader)).to(device)
                val_recon, _ = model(val_batch)
                val_loss = l1_loss_fn(val_recon, val_batch)
                print(f"Validation L1 loss: {val_loss.item():.4f}")

    torch.save({
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict(),
    }, 'checkpoints_improved/autoencoder_final.pt')

    print("Training complete!")
    from google.colab import files
    for f in sorted(glob.glob('checkpoints_improved/*.pt')):
        files.download(f)


if __name__ == "__main__":
    # Train with 512x512 resolution for better quality
    train(epochs=100, batch_size=2, lr=1e-4, resolution=512)