#!/usr/bin/env python3
"""
Generate demo videos using joint-trained checkpoint (λ=0.05) on the training/validation clips.
"""

import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.distributions.laplace import Laplace

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.quant import Quantizer

# ---------- Model definitions (same as training) ----------
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

class CheckerboardContext(nn.Module):
    def __init__(self, channels, hidden_dim=128):
        super().__init__()
        self.mask_conv = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.refine = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, 3, padding=1),
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

class ContextualEntropyModel(nn.Module):
    def __init__(self, latent_dim=192, hyper_channels=512, context_hidden=128):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2, stride=2),
            nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2),
            nn.GELU(),
        )
        self.skip_proj = nn.Conv2d(latent_dim, hyper_channels, 1)
        self.head = nn.Conv2d(hyper_channels, latent_dim * 2, 1)
        self.context = CheckerboardContext(latent_dim, context_hidden)
        self.refine_mu = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        self.refine_scale = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
    def forward(self, x):
        x_down = self.down(x)
        x_up = self.up(x_down)
        x_skip = nn.functional.interpolate(x, size=x_up.shape[2:], mode='bilinear', align_corners=False)
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

# Load joint-trained checkpoint (λ=0.05)
lam = 0.05
ae_ckpt = f'/root/le-maia/checkpoints_joint_phase0/ae_lambda_{lam}_final.pt'
ent_ckpt = f'/root/le-maia/checkpoints_joint_phase0/entropy_lambda_{lam}_final.pt'

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

quantizer = Quantizer(num_levels=256, mode='inference').to(device)
QUANT_STEP = 2.0 / 255

def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.003, epsilon=1e-9):
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    laplace = Laplace(mu, scale)
    cdf_upper = laplace.cdf(y + 0.5 * step)
    cdf_lower = laplace.cdf(y - 0.5 * step)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    nll = -torch.log(pmf)
    return nll.mean()

def encode_lewm_frames(frames):
    decoded = []
    bitmaps = []
    for frame in tqdm(frames, desc="LeWM encoding"):
        frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
        frame_t = frame_t.to(device)
        with torch.no_grad():
            latent_norm = autoencoder.encode(frame_t)
            quantized = quantizer(latent_norm)
            mu, log_scale = entropy_model(quantized)
            # Per-patch bits
            scale = torch.nn.functional.softplus(log_scale) + 0.003
            laplace_dist = Laplace(mu, scale)
            nll_per_element = -laplace_dist.log_prob(quantized)
            bits_per_patch = nll_per_element.mean(dim=1).cpu().numpy()[0]  # [H/16, W/16]
            bitmaps.append(bits_per_patch)
            recon = autoencoder.decode(quantized, target_size=(frame.shape[0], frame.shape[1]))
            recon_np = (recon.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            decoded.append(recon_np)
    return decoded, bitmaps

def overlay_heatmap(img, heatmap, alpha=0.6, color_map=cv2.COLORMAP_JET):
    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_NEAREST)
    heatmap_norm = (heatmap_resized / (heatmap_resized.max() + 1e-8) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_norm, color_map)
    blended = cv2.addWeighted(img, 1-alpha, colored, alpha, 0)
    return blended

def main():
    # Use the first video from the dataset (or iterate over all)
    dataset_dir = '/root/le-maia/datasets/pevid-hd'
    video_paths = glob.glob(os.path.join(dataset_dir, '*.mpg'))
    output_dir = '/root/le-maia/demo_videos_joint'
    os.makedirs(output_dir, exist_ok=True)
    target_size = (256,256)
    for vp in video_paths[:1]:  # just one for test, remove [:1] for all
        basename = os.path.splitext(os.path.basename(vp))[0]
        out_path = os.path.join(output_dir, f'{basename}_demo.mp4')
        print(f"Processing {basename}...")
        # Read frames
        cap = cv2.VideoCapture(vp)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        # Encode with LeWM-VC
        lewm_frames, bitmaps = encode_lewm_frames(frames)
        # Encode with x265 at CRF 36 for comparison
        import subprocess
        temp_x265 = '/tmp/x265_out.mp4'
        cmd = ['ffmpeg', '-y', '-i', vp, '-vf', f'scale={target_size[0]}:{target_size[1]}', '-c:v', 'libx265', '-crf', '36', '-preset', 'medium', temp_x265]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cap_x265 = cv2.VideoCapture(temp_x265)
        x265_frames = []
        while True:
            ret, frame = cap_x265.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x265_frames.append(frame)
        cap_x265.release()
        min_len = min(len(frames), len(x265_frames), len(lewm_frames))
        frames = frames[:min_len]
        x265_frames = x265_frames[:min_len]
        lewm_frames = lewm_frames[:min_len]
        bitmaps = bitmaps[:min_len]
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (target_size[1]*2, target_size[0]))
        for i, (orig, x265, lewm, bm) in enumerate(zip(frames, x265_frames, lewm_frames, bitmaps)):
            left = cv2.cvtColor(x265, cv2.COLOR_RGB2BGR)
            cv2.putText(left, 'x265 (CRF 36)', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            right = cv2.cvtColor(lewm, cv2.COLOR_RGB2BGR)
            right = overlay_heatmap(right, bm, alpha=0.6)
            cv2.putText(right, f'LeWM-VC (λ={lam})', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(right, 'Heatmap: bits/patch (red=high)', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            combined = np.hstack((left, right))
            out.write(combined)
        out.release()
        os.remove(temp_x265)
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
