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
from torch.distributions.normal import Normal  # ← ADD THIS

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- ResidualBlock ----------
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

# ---------- LeWMDecoder ----------
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

# ---------- AffineNormalization ----------
class AffineNormalization(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x):
        return x * self.scale + self.shift

# ---------- VideoAutoencoderWithAffine ----------
class VideoAutoencoderWithAffine(nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.affine = AffineNormalization(latent_dim)
    def encode(self, x):
        latent = self.encoder(x, return_surprise=False)
        return self.affine(latent)

# Load frozen affine autoencoder
autoencoder = VideoAutoencoderWithAffine(latent_dim=192).to(device)
ae_checkpoint = '/root/le-maia/checkpoints_affine/autoencoder_affine_final.pt'
# ✅ FIX: Add weights_only=False to suppress warning (or use safe weights_only=True if checkpoint is trusted)
autoencoder.load_state_dict(torch.load(ae_checkpoint, map_location=device, weights_only=False), strict=False)
autoencoder.eval()
for param in autoencoder.parameters():
    param.requires_grad = False
print("Autoencoder loaded and frozen.")

# ---------- Dataset ----------
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

# ---------- Entropy model and quantizer ----------
entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
quantizer = Quantizer(num_levels=256, mode='training').to(device)

# ---------- Latent statistics diagnostic ----------
print("\n=== Latent Statistics Diagnostic ===")
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        if i >= 3: break
        batch = batch.to(device)
        B, T, C, H, W = batch.shape
        batch_flat = batch.view(B * T, C, H, W)
        latent_norm = autoencoder.encode(batch_flat)
        quantized = quantizer(latent_norm)
        print(f"Batch {i}: latent mean={latent_norm.mean().item():.4f}, std={latent_norm.std().item():.4f}")
        print(f"       quantized mean={quantized.mean().item():.4f}, std={quantized.std().item():.4f}, "
              f"min={quantized.min().item():.4f}, max={quantized.max().item():.4f}")
print("===================================\n")

# Optional resume
checkpoint_dir = '/root/le-maia/checkpoints_entropy_nll'
os.makedirs(checkpoint_dir, exist_ok=True)
resume_ckpt = f"{checkpoint_dir}/entropy_final.pt"
if os.path.exists(resume_ckpt):
    entropy_model.load_state_dict(torch.load(resume_ckpt, map_location=device, weights_only=False))
    print(f"✅ Resumed from previous checkpoint: {resume_ckpt}")

# ---------- ✅ CORRECTED: Discrete Gaussian Likelihood via CDF ----------
def gaussian_likelihood_discrete(y, mu, log_sigma, epsilon=1e-9):
    """
    Compute -log P(y) for quantized integer y under Gaussian(μ, σ).
    Uses CDF difference for proper discrete probability mass.
    """
    sigma = torch.nn.functional.softplus(log_sigma) + 1e-3  # small floor for stability
    sigma = torch.clamp(sigma, min=1e-3, max=10.0)
    
    # Standardize the bounds for CDF
    lower = (y - 0.5 - mu) / sigma
    upper = (y + 0.5 - mu) / sigma
    
    # Use stable CDF from torch.distributions
    normal = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    cdf_upper = normal.cdf(upper)
    cdf_lower = normal.cdf(lower)
    
    # Probability mass in bin [y-0.5, y+0.5]
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)  # prevent log(0)
    
    nll = -torch.log(pmf)
    return nll.mean()

optimizer = optim.AdamW(entropy_model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

EPOCHS = 100

for epoch in range(1, EPOCHS + 1):
    entropy_model.train()
    total_nll = 0.0
    num_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")

    for batch in pbar:
        batch = batch.to(device)
        B, T, C, H, W = batch.shape
        batch_flat = batch.view(B * T, C, H, W)

        with torch.no_grad():
            latent_norm = autoencoder.encode(batch_flat)

        # Quantize with straight-through estimator (handled inside Quantizer)
        quantized = quantizer(latent_norm)

        # Predict entropy parameters
        params = entropy_model.hyperprior_cnn(quantized)
        mu = params[:, :192, :, :]
        log_sigma = params[:, 192:, :, :]

        # ✅ Use discrete likelihood, NOT continuous NLL
        nll = gaussian_likelihood_discrete(quantized, mu, log_sigma)
        
        loss = nll  # ← No ad-hoc regularization needed; CDF formulation is stable

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 1.0)
        optimizer.step()

        total_nll += nll.item()
        num_batches += 1

        # ✅ BPP is now always positive and meaningful
        bpp = nll.item() / np.log(2)
        sigma_val = torch.nn.functional.softplus(log_sigma).mean().item()
        mean_err = (quantized - mu).abs().mean().item()
        
        pbar.set_postfix(
            bpp=f"{bpp:.4f}",
            sigma=f"{sigma_val:.4f}",
            mean_err=f"{mean_err:.4f}",
            loss=f"{loss.item():.4f}"
        )

    scheduler.step()
    avg_nll = total_nll / num_batches
    avg_bpp = avg_nll / np.log(2)
    print(f"Epoch {epoch}: Avg NLL = {avg_nll:.6f}  →  {avg_bpp:.4f} bpp (✅ should be >0)")

    if epoch % 10 == 0 or epoch == EPOCHS:
        torch.save(entropy_model.state_dict(), f'{checkpoint_dir}/entropy_epoch{epoch}.pt')

torch.save(entropy_model.state_dict(), f'{checkpoint_dir}/entropy_final.pt')
print("✅ Entropy model training complete (DISCRETE GAUSSIAN LIKELIHOOD).")
