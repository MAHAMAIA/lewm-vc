#!/usr/bin/env python3
import os
import sys
import glob
import torch
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------- Model definitions (same as training) ----------
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = torch.nn.InstanceNorm2d(channels)
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = torch.nn.InstanceNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        residual = x
        x = torch.nn.functional.gelu(self.norm1(x))
        x = self.conv1(x)
        x = torch.nn.functional.gelu(self.norm2(x))
        x = self.conv2(x)
        return x + residual

class LeWMDecoder(torch.nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512):
        super().__init__()
        self.proj = torch.nn.Conv2d(latent_dim, hidden_dim, 1)
        self.up1 = torch.nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4,2,1)
        self.res1 = ResidualBlock(hidden_dim//2)
        self.up2 = torch.nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4,2,1)
        self.res2 = ResidualBlock(hidden_dim//4)
        self.up3 = torch.nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4,2,1)
        self.res3 = ResidualBlock(hidden_dim//8)
        self.up4 = torch.nn.ConvTranspose2d(hidden_dim//8, hidden_dim//16, 4,2,1)
        self.res4 = ResidualBlock(hidden_dim//16)
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_dim//16, hidden_dim//32, 3,1,1),
            torch.nn.InstanceNorm2d(hidden_dim//32),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden_dim//32, 3, 3,1,1),
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

class AffineNormalization(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = torch.nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x):
        return x * self.scale + self.shift

class VideoAutoencoder(torch.nn.Module):
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

class CheckerboardContext(torch.nn.Module):
    def __init__(self, channels, hidden_dim=128):
        super().__init__()
        self.mask_conv = torch.nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.refine = torch.nn.Sequential(
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden_dim, channels, 3, padding=1),
        )
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

class ContextualEntropyModel(torch.nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=512, context_hidden=128):
        super().__init__()
        self.down = torch.nn.Sequential(
            torch.nn.Conv2d(latent_dim, hyper_channels, 5, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2, stride=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2),
            torch.nn.GELU(),
        )
        self.up = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2),
            torch.nn.GELU(),
        )
        self.skip_proj = torch.nn.Conv2d(latent_dim, hyper_channels, 1)
        self.head = torch.nn.Conv2d(hyper_channels, latent_dim * 2, 1)
        self.context = CheckerboardContext(latent_dim, context_hidden)
        self.refine_mu = torch.nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        self.refine_scale = torch.nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
    def forward(self, x):
        x_down = self.down(x)
        x_up = self.up(x_down)
        x_skip = torch.nn.functional.interpolate(x, size=x_up.shape[2:], mode='bilinear', align_corners=False)
        x_skip = self.skip_proj(x_skip)
        features = x_up + x_skip
        base_params = self.head(features)
        mu_base = base_params[:, :192, :, :]
        log_scale_base = base_params[:, 192:, :, :]
        ctx = self.context(x)
        mu_offset = self.refine_mu(ctx)
        scale_offset = self.refine_scale(ctx)
        mu = mu_base + mu_offset
        log_scale = log_scale_base + scale_offset
        return mu, log_scale

# Load test video
test_video = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')[0]
cap = cv2.VideoCapture(test_video)
frames = []
target_size = (256,256)
while len(frames) < 150:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, target_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()
print(f"Loaded {len(frames)} frames")

quantizer = Quantizer(num_levels=256, mode='inference').to(device)
QUANT_STEP = 2.0 / 255

def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.003, epsilon=1e-9):
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    laplace = torch.distributions.laplace.Laplace(mu, scale)
    cdf_upper = laplace.cdf(y + 0.5 * step)
    cdf_lower = laplace.cdf(y - 0.5 * step)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    nll = -torch.log(pmf)
    return nll.mean()

# Use the best checkpoint for λ=0.05
lam = 0.05
ae_ckpt = f'/root/le-maia/checkpoints_joint_phase0/ae_lambda_{lam}_best.pt'
ent_ckpt = f'/root/le-maia/checkpoints_joint_phase0/entropy_lambda_{lam}_best.pt'

autoencoder = VideoAutoencoder().to(device)
autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device, weights_only=False), strict=False)
autoencoder.eval()
entropy_model = ContextualEntropyModel().to(device)
state = torch.load(ent_ckpt, map_location=device, weights_only=False)
for key in list(state.keys()):
    if 'mask' in key:
        del state[key]
entropy_model.load_state_dict(state, strict=False)
entropy_model.eval()

total_bits = 0
total_mse = 0
for frame in tqdm(frames, desc="Validating"):
    frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
    frame_t = frame_t.to(device)
    with torch.no_grad():
        latent_norm = autoencoder.encode(frame_t)
        quantized = quantizer(latent_norm)
        mu, log_scale = entropy_model(quantized)
        nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP, sigma_floor=0.003)
        bits = nll * quantized.numel() / np.log(2)
        total_bits += bits.item()
        recon = autoencoder.decode(quantized, target_size=target_size)
        mse = torch.nn.functional.mse_loss(recon, frame_t).item()
        total_mse += mse

bpp = total_bits / (len(frames) * target_size[0] * target_size[1])
psnr = 20 * np.log10(1.0 / np.sqrt(total_mse / len(frames)))
print(f"λ={lam} (best): bpp={bpp:.4f}, PSNR={psnr:.2f} dB")
