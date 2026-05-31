# debug_affine.py
import torch
import cv2
import numpy as np
import sys
sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load affine model
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

class VideoAutoencoder(torch.nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)

class AffineNormalization(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.shift = torch.nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x):
        return x * self.scale + self.shift
    def inverse(self, y):
        return (y - self.shift) / self.scale

class VideoAutoencoderWithAffine(torch.nn.Module):
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

model = VideoAutoencoderWithAffine().to(device)
checkpoint_path = '/root/le-maia/checkpoints_affine/autoencoder_affine_final.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
model.eval()

# Load a frame
cap = cv2.VideoCapture('/root/le-maia/datasets/pevid-hd/stealing_night_outdoor_1_2.mpg')
ret, frame = cap.read()
cap.release()
frame = cv2.resize(frame, (256,256))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
frame_t = frame_t.to(device)

with torch.no_grad():
    latent_norm = model.encode(frame_t)
    print(f"Latent norm mean: {latent_norm.mean().item():.4f}, std: {latent_norm.std().item():.4f}")
    # Quantize
    std = latent_norm.std().item()
    step_size = max(0.01, std / 4.0)
    print(f"Step size: {step_size:.6f}")
    quantizer = Quantizer(num_levels=256, mode='inference').to(device)
    quantizer.step_size = torch.tensor(step_size).to(device)
    quantized = quantizer(latent_norm)
    print(f"Quantized unique values: {len(torch.unique(quantized))} (out of {quantized.numel()})")
    # Entropy model
    entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
    phase1_ckpt = '/root/le-maia/checkpoints/phase1_lambda_0.1/final.pt'
    state_ent = torch.load(phase1_ckpt, map_location=device)
    if 'entropy_model' in state_ent:
        entropy_model.load_state_dict(state_ent['entropy_model'])
    else:
        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.entropy_model = entropy_model
        dummy = Dummy()
        dummy.load_state_dict(state_ent, strict=False)
    entropy_model.eval()
    rate_nats, _ = entropy_model(quantized)
    bits = rate_nats.sum().item() * np.log2(np.e)
    print(f"Rate bits: {bits:.4f}")
    # Decode
    recon = model.decode(quantized, target_size=(256,256))
    recon_np = (recon.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    psnr = 20*np.log10(255/np.sqrt(np.mean((frame.astype(float)-recon_np.astype(float))**2)))
    print(f"PSNR: {psnr:.2f} dB")
