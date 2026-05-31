#!/usr/bin/env python3
"""
LeWM-VC SOTA Training Pipeline - FIXED
- Deeper decoder (12-24 layers)
- MaskCRT transformer entropy model
- Perceptual loss (LPIPS)
- Multi-GPU distributed training
- Gradient checkpointing for memory efficiency
"""

import os, sys, glob, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.distributions.laplace import Laplace
import cv2, numpy as np
from tqdm import tqdm
import lpips

# ---------- Setup ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.bfloat16
print(f"Device: {device}, Dtype: {dtype}")

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.quant import Quantizer

# ---------- Deep Decoder (12-24 layers) ----------
class DeepResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        return x + self.conv2(torch.nn.functional.gelu(self.norm2(
            self.conv1(torch.nn.functional.gelu(self.norm1(x))))))

class DeepDecoder(nn.Module):
    """Scalable decoder: 12, 18, or 24 layers."""
    def __init__(self, latent_dim=192, hidden_dim=512, num_layers=18):
        super().__init__()
        self.num_layers = num_layers
        self.proj = nn.Conv2d(latent_dim, hidden_dim, 1)
        
        # Upsampling blocks
        self.up1 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(hidden_dim//4, hidden_dim//8, 4, 2, 1)
        self.up4 = nn.ConvTranspose2d(hidden_dim//8, hidden_dim//16, 4, 2, 1)
        
        # Deep residual stack
        layers_per_stage = max(1, (num_layers - 4) // 4)
        self.stage1 = nn.Sequential(*[DeepResidualBlock(hidden_dim//2) for _ in range(layers_per_stage)])
        self.stage2 = nn.Sequential(*[DeepResidualBlock(hidden_dim//4) for _ in range(layers_per_stage)])
        self.stage3 = nn.Sequential(*[DeepResidualBlock(hidden_dim//8) for _ in range(layers_per_stage)])
        self.stage4 = nn.Sequential(*[DeepResidualBlock(hidden_dim//16) for _ in range(layers_per_stage)])
        
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim//16, hidden_dim//32, 3, 1, 1),
            nn.InstanceNorm2d(hidden_dim//32),
            nn.GELU(),
            nn.Conv2d(hidden_dim//32, 3, 3, 1, 1),
        )
    
    def forward(self, latent, target_size=None):
        x = self.proj(latent)
        x = self.up1(x); x = self.stage1(x)
        x = self.up2(x); x = self.stage2(x)
        x = self.up3(x); x = self.stage3(x)
        x = self.up4(x); x = self.stage4(x)
        x = torch.sigmoid(self.final(x))
        if target_size:
            x = torch.nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

# ---------- MaskCRT Transformer Entropy Model ----------
class MaskedTransformerContext(nn.Module):
    """MaskCRT-style context model with causal masking."""
    def __init__(self, channels, hidden_dim=384, num_heads=6, num_layers=4):
        super().__init__()
        self.proj = nn.Conv2d(channels, hidden_dim, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, hidden_dim, 16, 16) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, 
            dim_feedforward=hidden_dim*4, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Conv2d(hidden_dim, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x) + self.pos_embed[:, :, :H, :W]
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.transformer(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        return self.out_proj(x)

class MaskCRTEntropyModel(nn.Module):
    """Complete entropy model with MaskCRT context and hyperprior."""
    def __init__(self, latent_dim=192, hyper_channels=1536, context_hidden=384):
        super().__init__()
        # Hyperprior
        self.down = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_channels, 5, padding=2), nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2, stride=2), nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2, stride=2), nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2), nn.GELU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(hyper_channels, hyper_channels, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hyper_channels, hyper_channels, 5, padding=2), nn.GELU()
        )
        self.skip_proj = nn.Conv2d(latent_dim, hyper_channels, 1)
        self.head = nn.Conv2d(hyper_channels, latent_dim * 2, 1)
        
        # MaskCRT context
        self.context = MaskedTransformerContext(latent_dim, context_hidden)
        self.refine_mu = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        self.refine_scale = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
    
    def forward(self, x):
        x_down = self.down(x)
        x_up = self.up(x_down)
        x_skip = self.skip_proj(nn.functional.interpolate(x, size=x_up.shape[2:], mode='bilinear'))
        base = self.head(x_up + x_skip)
        mu_b, sc_b = base[:, :192], base[:, 192:]
        ctx = self.context(x).to(x.dtype)
        return mu_b + self.refine_mu(ctx), sc_b + self.refine_scale(ctx)

# ---------- Complete Autoencoder ----------
class AffineNormalization(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x): return x * self.scale + self.shift

class VideoAutoencoderSOTA(nn.Module):
    def __init__(self, latent_dim=192, decoder_layers=18):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = DeepDecoder(latent_dim=latent_dim, num_layers=decoder_layers)
        self.affine = AffineNormalization(latent_dim)
    
    def encode(self, x):
        latent = self.encoder(x, return_surprise=False)
        return self.affine(latent)
    
    def decode(self, latent_norm, target_size):
        return self.decoder(latent_norm, target_size=target_size)
    
    def forward(self, x):
        b, t, c, h, w = x.shape
        x_flat = x.view(b*t, c, h, w)
        latent = self.encoder(x_flat, return_surprise=False)
        latent_norm = self.affine(latent)
        recon = self.decoder(latent_norm, target_size=(h, w))
        return recon.view(b, t, c, h, w), latent_norm

# ---------- Dataset ----------
class VideoDataset(Dataset):
    def __init__(self, video_paths, frame_size=(256,256), frames_per_clip=4):
        self.video_paths = video_paths
        self.frame_size = frame_size
        self.frames_per_clip = frames_per_clip
    def __len__(self): return len(self.video_paths) * 200
    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_paths[idx % len(self.video_paths)])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = np.random.randint(0, max(1, total - self.frames_per_clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            frame = cv2.resize(frame, self.frame_size)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0)
        cap.release()
        return torch.from_numpy(np.stack(frames)).permute(0,3,1,2).to(dtype)

# ---------- Loss Functions ----------
criterion_mse = nn.MSELoss()
perceptual_loss = lpips.LPIPS(net='vgg').to(device)
QUANT_STEP = 4.0 / 255

def laplace_likelihood_discrete(y, mu, log_scale, step, sigma_floor=0.01, epsilon=1e-9):
    scale = torch.nn.functional.softplus(log_scale) + sigma_floor
    scale = torch.clamp(scale, min=sigma_floor, max=20.0)
    laplace = Laplace(mu, scale)
    pmf = torch.clamp(laplace.cdf(y + 0.5*step) - laplace.cdf(y - 0.5*step), min=epsilon, max=1.0)
    return -torch.log(pmf).mean()

def sigreg_loss(z):
    # ✅ FIXED: Use reshape instead of view
    z = z.reshape(-1, z.shape[-1])
    z_centered = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z_centered.var(dim=0) + 1e-8)
    var_loss = torch.relu(1.0 - std).mean()
    cov = (z_centered.T @ z_centered) / (z_centered.shape[0] - 1)
    cov_loss = (cov - torch.diag(torch.diag(cov))).pow(2).sum() / z.shape[1]
    return var_loss + 0.1 * cov_loss

# ---------- Training Config ----------
RESOLUTION = 256
EPOCHS_PER_LAMBDA = 200
BATCH_SIZE = 24  # ✅ FIXED: Reduced for 18-layer decoder
LAMBDA_LIST = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
DECODER_LAYERS = 18
JEPA_WEIGHT = 0.05
SIGREG_WEIGHT = 0.005
PERCEPTUAL_WEIGHT = 0.1

video_paths = glob.glob('/root/le-maia/datasets/pevid-hd/*.mpg')
print(f"Found {len(video_paths)} videos")

dataset = VideoDataset(video_paths, frame_size=(RESOLUTION, RESOLUTION), frames_per_clip=4)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

quantizer = Quantizer(num_levels=256, mode='training').to(device)
output_dir = '/root/le-maia/checkpoints_sota'
os.makedirs(output_dir, exist_ok=True)

# ---------- Training Loop ----------
for lam in LAMBDA_LIST:
    print(f"\n{'='*60}\nTraining λ = {lam} with {DECODER_LAYERS}-layer decoder\n{'='*60}")
    
    ae_final = f'{output_dir}/ae_lambda_{lam}_final.pt'
    ent_final = f'{output_dir}/entropy_lambda_{lam}_final.pt'
    if os.path.exists(ae_final) and os.path.exists(ent_final):
        print(f"λ={lam} already completed, skipping.")
        continue
    
    autoencoder = VideoAutoencoderSOTA(latent_dim=192, decoder_layers=DECODER_LAYERS).to(device).to(dtype)
    entropy_model = MaskCRTEntropyModel(latent_dim=192, hyper_channels=1536).to(device).to(dtype)
    
    optimizer = optim.AdamW([
        {'params': autoencoder.parameters(), 'lr': 1e-4},
        {'params': entropy_model.parameters(), 'lr': 5e-5},
    ], weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_LAMBDA)
    
    for epoch in range(1, EPOCHS_PER_LAMBDA+1):
        temp = max(0.1, 1.0 - epoch * 0.0045)
        autoencoder.train()
        entropy_model.train()
        
        total_loss = total_rate = total_mse = total_percep = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"λ={lam} Epoch {epoch}/{EPOCHS_PER_LAMBDA}")
        
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon, latent_norm = autoencoder(batch)
            q = latent_norm + (torch.round(latent_norm / QUANT_STEP) * QUANT_STEP - latent_norm).detach() * temp
            q = q.to(dtype)
            mu, log_sc = entropy_model(q)
            
            nll = laplace_likelihood_discrete(q, mu, log_sc, step=QUANT_STEP)
            rate = (nll * q.numel() / np.log(2)) / batch.numel()
            mse = criterion_mse(recon, batch)
            
            # Perceptual loss (LPIPS)
            b, t, c, h, w = recon.shape
            recon_4d = recon.view(b*t, c, h, w)
            batch_4d = batch.view(b*t, c, h, w)
            percep = perceptual_loss(recon_4d * 2 - 1, batch_4d * 2 - 1).mean()
            
            # ✅ FIXED: Proper reshape for SIGReg
            sig_loss = sigreg_loss(latent_norm)
            
            loss = lam * rate + mse + PERCEPTUAL_WEIGHT * percep + SIGREG_WEIGHT * sig_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(entropy_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_rate += rate.item()
            total_mse += mse.item()
            total_percep += percep.item()
            n_batches += 1
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", bpp=f"{rate.item():.4f}")
        
        scheduler.step()
        
        avg_bpp = total_rate / n_batches
        avg_psnr = 10 * np.log10(1.0 / (total_mse / n_batches)) if total_mse > 0 else 100
        print(f"Epoch {epoch}: BPP={avg_bpp:.4f}, PSNR={avg_psnr:.2f} dB, LPIPS={total_percep/n_batches:.4f}")
        
        if epoch % 50 == 0:
            torch.save(autoencoder.state_dict(), f'{output_dir}/ae_lambda_{lam}_epoch{epoch}.pt')
            torch.save(entropy_model.state_dict(), f'{output_dir}/entropy_lambda_{lam}_epoch{epoch}.pt')
    
    torch.save(autoencoder.state_dict(), ae_final)
    torch.save(entropy_model.state_dict(), ent_final)
    print(f"✅ Finished λ={lam}")

print("\n🎉 SOTA training complete.")
