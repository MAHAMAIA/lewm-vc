import torch
import cv2
import numpy as np
import sys
sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Affine autoencoder (same as before)
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

class VideoAutoencoderWithAffine(torch.nn.Module):
    def __init__(self, latent_dim=192):
        super().__init__()
        self.encoder = LeWMEncoder(latent_dim=latent_dim, semantic_surprise=True)
        self.decoder = LeWMDecoder(latent_dim=latent_dim)
        self.affine = AffineNormalization(latent_dim)
    def encode(self, x):
        latent = self.encoder(x, return_surprise=False)
        return self.affine(latent)

# Load autoencoder
autoencoder = VideoAutoencoderWithAffine(latent_dim=192).to(device)
checkpoint_affine = '/root/le-maia/checkpoints_affine/autoencoder_affine_final.pt'
autoencoder.load_state_dict(torch.load(checkpoint_affine, map_location=device), strict=False)
autoencoder.eval()

# Load a frame
cap = cv2.VideoCapture('/root/le-maia/datasets/pevid-hd/stealing_night_outdoor_1_2.mpg')
ret, frame = cap.read()
cap.release()
frame = cv2.resize(frame, (256,256))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_t = torch.from_numpy(frame).float().permute(2,0,1).unsqueeze(0) / 255.0
frame_t = frame_t.to(device)

with torch.no_grad():
    latent_norm = autoencoder.encode(frame_t)
    print(f"Latent shape: {latent_norm.shape}")
    print(f"Mean: {latent_norm.mean().item():.4f}, Std: {latent_norm.std().item():.4f}")
    print(f"Min: {latent_norm.min().item():.4f}, Max: {latent_norm.max().item():.4f}")
    # Quantize with inference mode
    quantizer = Quantizer(num_levels=256, mode='inference').to(device)
    quantized = quantizer(latent_norm)
    print(f"Quantized unique values: {torch.unique(quantized).numel()}")
    print(f"Quantized min: {quantized.min().item():.4f}, max: {quantized.max().item():.4f}")

# Load entropy model (NLL version) and compute rate
from lewm_vc.entropy import HyperpriorEntropy
entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
entropy_ckpt = '/root/le-maia/checkpoints_entropy_affine_v3/entropy_final.pt'
entropy_model.load_state_dict(torch.load(entropy_ckpt, map_location=device))
entropy_model.eval()
with torch.no_grad():
    rate_nats, _ = entropy_model(quantized)
    bits = rate_nats.sum().item() * np.log2(np.e)
    print(f"Rate bits: {bits:.4f}")
