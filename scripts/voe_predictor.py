#!/usr/bin/env python3
"""
VoE (Video Outlier/Anomaly) Predictor with joint surprise training.
Trains jointly with the autoencoder to detect anomalous frames via prediction error.
"""

import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

FRAME_SIZE = (256, 256)
FRAMES_PER_CLIP = 8


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
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1)
        self.res1 = ResidualBlock(hidden_dim // 2)
        self.up2 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1)
        self.res2 = ResidualBlock(hidden_dim // 4)
        self.up3 = nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, 4, 2, 1)
        self.res3 = ResidualBlock(hidden_dim // 8)
        self.up4 = nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, 4, 2, 1)
        self.res4 = ResidualBlock(hidden_dim // 16)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim // 16, hidden_dim // 32, 3, 1, 1),
            nn.InstanceNorm2d(hidden_dim // 32),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 32, 3, 3, 1, 1),
        )

    def forward(self, latent, target_size=None):
        x = self.proj(latent)
        x = self.up1(x)
        x = self.res1(x)
        x = self.up2(x)
        x = self.res2(x)
        x = self.up3(x)
        x = self.res3(x)
        x = self.up4(x)
        x = self.res4(x)
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


class SurprisePredictor(nn.Module):
    """Predicts next frame latent to detect anomalies via prediction error."""
    def __init__(self, latent_dim=192, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, latent_dim, 3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.conv1(x))
        x = torch.nn.functional.gelu(self.conv2(x))
        x = self.conv3(x)
        return x


class VideoAutoencoderWithPredictor(nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.affine = AffineNormalization(latent_dim)
        self.predictor = SurprisePredictor(latent_dim)

    def forward(self, x, prev_latent=None):
        b, t, c, h, w = x.shape
        x_flat = x.view(b * t, c, h, w)
        latent = self.encoder(x_flat, return_surprise=False)
        latent_norm = self.affine(latent)
        recon = self.decoder(latent_norm, target_size=(h, w))
        recon = recon.view(b, t, c, h, w)

        latent_reshaped = latent_norm.view(b, t, -1, h, w)
        surprises = []
        for i in range(t):
            if prev_latent is not None:
                pred = self.predictor(prev_latent)
                surprise = torch.mean((latent_reshaped[:, i] - pred) ** 2, dim=[1, 2, 3])
                surprises.append(surprise)
            else:
                surprises.append(torch.zeros(b, device=latent.device))
            prev_latent = latent_reshaped[:, i].detach()

        return recon, torch.stack(surprises)

    def encode(self, x):
        latent = self.encoder(x, return_surprise=False)
        return self.affine(latent)

    def decode(self, latent_norm, target_size):
        return self.decoder(latent_norm, target_size=target_size)


class VideoDataset(Dataset):
    def __init__(self, video_paths, frame_size=(256,256), frames_per_clip=8, is_anomaly=False):
        self.videos = video_paths
        self.frame_size = frame_size
        self.frames_per_clip = frames_per_clip
        self.is_anomaly = is_anomaly

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
        return torch.from_numpy(np.stack(frames)).float(), self.is_anomaly


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
    def __init__(self, latent_dim=192, hyper_channels=640, context_hidden=256):
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


class VideoDatasetWithLabels(Dataset):
    def __init__(self, normal_paths, anomaly_paths, frame_size=(256,256), frames_per_clip=8):
        self.normal_paths = normal_paths
        self.anomaly_paths = anomaly_paths
        self.frame_size = frame_size
        self.frames_per_clip = frames_per_clip
        self.all_paths = normal_paths + anomaly_paths
        self.labels = [0] * len(normal_paths) + [1] * len(anomaly_paths)

    def __len__(self):
        return len(self.all_paths) * 200

    def __getitem__(self, idx):
        path_idx = idx % len(self.all_paths)
        is_anomaly = self.labels[path_idx]
        video_path = self.all_paths[path_idx]
        cap = cv2.VideoCapture(video_path)
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
        return torch.from_numpy(np.stack(frames)).float(), is_anomaly


def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.05, epsilon=1e-9):
    from torch.distributions.laplace import Laplace
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    scale = torch.clamp(scale, min=sigma_floor, max=5.0)
    laplace = Laplace(mu, scale)
    cdf_upper = laplace.cdf(y + 0.5 * step)
    cdf_lower = laplace.cdf(y - 0.5 * step)
    pmf = cdf_upper - cdf_lower
    pmf = torch.clamp(pmf, min=epsilon, max=1.0)
    nll = -torch.log(pmf)
    return nll.mean()


def quantize_with_temp(x, step, temp):
    x_quant = torch.round(x / step) * step
    return x_quant + (x - x_quant.detach()) * temp


def train_voe_predictor():
    normal_videos = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')
    anomaly_videos = glob.glob('/root/le-maia/datasets/pevid-anomaly/*.mpg')

    if not normal_videos:
        raise FileNotFoundError("No normal videos found")
    if not anomaly_videos:
        print("WARNING: No anomaly videos found, using normal only")
        dataset = VideoDataset(normal_videos, frame_size=FRAME_SIZE, frames_per_clip=FRAMES_PER_CLIP, is_anomaly=False)
    else:
        dataset = VideoDatasetWithLabels(normal_videos, anomaly_videos, frame_size=FRAME_SIZE, frames_per_clip=FRAMES_PER_CLIP)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    val_size = max(1, int(len(dataset) * 0.1))
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    autoencoder = VideoAutoencoderWithPredictor(latent_dim=192).to(device)
    entropy_model = ContextualEntropyModel(latent_dim=192, hyper_channels=640, context_hidden=256).to(device)
    quantizer = Quantizer(num_levels=256, mode='training').to(device)

    criterion_mse = nn.MSELoss()
    QUANT_STEP = 2.0 / 255

    optimizer = optim.AdamW([
        {'params': autoencoder.parameters(), 'lr': 1e-4},
        {'params': entropy_model.parameters(), 'lr': 5e-5},
    ], weight_decay=1e-6)

    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5)
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[5])

    scaler = GradScaler('cuda')

    EPOCHS = 100
    LAMBDA = 0.1
    LAMBDA_SURPRISE = 0.5

    best_val_loss = float('inf')
    best_model_path = '/root/le-maia/checkpoints_rd_scratch/voe_best.pt'
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        temp = max(0.1, 1.0 - epoch * 0.02)
        autoencoder.train()
        entropy_model.train()

        total_loss = 0
        total_surprise_normal = 0
        total_surprise_anomaly = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch, labels in pbar:
            batch = batch.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                recon, surprises = autoencoder(batch)
                b, t, c, h, w = batch.shape
                recon_flat = recon.view(b * t, c, h, w)
                batch_flat = batch.view(b * t, c, h, w)

                quantized = quantize_with_temp(
                    autoencoder.affine(autoencoder.encoder(batch_flat, return_surprise=False)),
                    QUANT_STEP, temp
                )

                mu, log_scale = entropy_model(quantized)
                nll = laplace_likelihood_discrete(quantized, mu, log_scale, step=QUANT_STEP)
                mse = criterion_mse(recon_flat, batch_flat)

                rate_per_pixel = nll / np.log(2)
                distortion = mse
                loss = LAMBDA * rate_per_pixel + distortion

                surprise_anomaly = surprises[labels == 1].mean() if (labels == 1).sum() > 0 else 0
                surprise_normal = surprises[labels == 0].mean() if (labels == 0).sum() > 0 else 0
                surprise_loss = -LAMBDA_SURPRISE * (surprise_anomaly - surprise_normal)

                total_loss_val = loss + surprise_loss

            scaler.scale(total_loss_val).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(autoencoder.parameters()) + list(entropy_model.parameters()),
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if (labels == 1).sum() > 0:
                total_surprise_anomaly += surprise_anomaly.item()
            if (labels == 0).sum() > 0:
                total_surprise_normal += surprise_normal.item()
            num_batches += 1

            pbar.set_postfix(
                loss=loss.item(),
                sup_n=surprise_normal.item() if (labels == 0).sum() > 0 else 0,
                sup_a=surprise_anomaly.item() if (labels == 1).sum() > 0 else 0
            )

        scheduler.step()

        avg_surprise_normal = total_surprise_normal / max(1, num_batches)
        avg_surprise_anomaly = total_surprise_anomaly / max(1, num_batches)
        ratio = avg_surprise_anomaly / (avg_surprise_normal + 1e-8)

        print(f"Epoch {epoch}: Loss={total_loss/num_batches:.4f}, "
              f"Surprise Normal={avg_surprise_normal:.4f}, Anomaly={avg_surprise_anomaly:.4f}, "
              f"Ratio={ratio:.2f}x")

        if epoch % 10 == 0:
            autoencoder.eval()
            v_surprise_normal = 0
            v_surprise_anomaly = 0
            v_batches = 0
            with torch.no_grad():
                for val_batch, val_labels in val_loader:
                    val_batch = val_batch.to(device)
                    val_labels = val_labels.to(device)
                    with autocast('cuda'):
                        _, val_surprises = autoencoder(val_batch)
                        if (val_labels == 0).sum() > 0:
                            v_surprise_normal += val_surprises[val_labels == 0].mean().item()
                        if (val_labels == 1).sum() > 0:
                            v_surprise_anomaly += val_surprises[val_labels == 1].mean().item()
                        v_batches += 1

            v_surprise_normal /= max(1, v_batches)
            v_surprise_anomaly /= max(1, v_batches)
            v_ratio = v_surprise_anomaly / (v_surprise_normal + 1e-8)
            print(f"  Val: Normal={v_surprise_normal:.4f}, Anomaly={v_surprise_anomaly:.4f}, Ratio={v_ratio:.2f}x")

            if v_ratio > 1.5 and total_loss / num_batches < best_val_loss:
                best_val_loss = total_loss / num_batches
                torch.save({
                    'autoencoder': autoencoder.state_dict(),
                    'entropy_model': entropy_model.state_dict(),
                }, best_model_path)
                print(f"  -> Saved best model (ratio={v_ratio:.2f}x)")

    print(f"\nTraining complete. Best model: {best_model_path}")
    return best_model_path


def evaluate_voe(model_path, test_videos):
    print(f"\nEvaluating VoE predictor from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    autoencoder = VideoAutoencoderWithPredictor(latent_dim=192).to(device)
    autoencoder.load_state_dict(checkpoint['autoencoder'], strict=False)
    autoencoder.eval()

    quantizer = Quantizer(num_levels=256, mode='inference').to(device)
    QUANT_STEP = 2.0 / 255

    results = []
    for video_path in tqdm(test_videos, desc="Evaluating"):
        cap = cv2.VideoCapture(video_path)
        frames = []
        target_size = FRAME_SIZE
        while len(frames) < 150:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        frames = frames[:150]

        if not frames:
            continue

        surprises = []
        prev_latent = None
        with torch.no_grad():
            for frame in frames:
                frame_t = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                frame_t = frame_t.to(device)
                latent = autoencoder.encode(frame_t)
                quantized = quantizer(latent)

                if prev_latent is not None:
                    pred = autoencoder.predictor(prev_latent)
                    surprise = torch.mean((latent - pred) ** 2).item()
                    surprises.append(surprise)
                else:
                    surprises.append(0.0)

                prev_latent = latent

        avg_surprise = np.mean(surprises)
        max_surprise = np.max(surprises)
        results.append({
            'video': os.path.basename(video_path),
            'avg_surprise': avg_surprise,
            'max_surprise': max_surprise,
            'surprises': surprises
        })

    normal_results = [r for r in results if 'normal' in r['video'].lower()]
    anomaly_results = [r for r in results if 'anomaly' in r['video'].lower()]

    if normal_results and anomaly_results:
        avg_normal = np.mean([r['avg_surprise'] for r in normal_results])
        avg_anomaly = np.mean([r['avg_surprise'] for r in anomaly_results])
        ratio = avg_anomaly / (avg_normal + 1e-8)
        print(f"\nVoE Results:")
        print(f"  Normal avg surprise: {avg_normal:.4f}")
        print(f"  Anomaly avg surprise: {avg_anomaly:.4f}")
        print(f"  Ratio: {ratio:.2f}x")

        if ratio >= 1.5:
            print(f"  ✅ VoE detection PASSED (ratio >= 1.5x)")
        else:
            print(f"  ❌ VoE detection FAILED (ratio < 1.5x)")

    output_path = '/root/le-maia/benchmark_results/voe_predictor_results.txt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"VoE Predictor Results\n")
        f.write(f"=" * 50 + "\n")
        for r in results:
            f.write(f"{r['video']}: avg={r['avg_surprise']:.4f}, max={r['max_surprise']:.4f}\n")
        if normal_results and anomaly_results:
            f.write(f"\nNormal avg: {avg_normal:.4f}\n")
            f.write(f"Anomaly avg: {avg_anomaly:.4f}\n")
            f.write(f"Ratio: {ratio:.2f}x\n")

    print(f"Results saved to {output_path}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'train':
        train_voe_predictor()
    else:
        if not args.model:
            print("ERROR: --model required for eval mode")
            sys.exit(1)
        test_videos = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')
        if not test_videos:
            test_videos = glob.glob('/root/le-maia/datasets/pevid-anomaly/*.mpg')
        evaluate_voe(args.model, test_videos)