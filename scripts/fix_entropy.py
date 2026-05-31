# fix_entropy.py
import torch
import sys
sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define FullCodec class (same as in training)
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

class FullCodec(torch.nn.Module):
    def __init__(self, autoencoder, entropy_model, quantizer):
        super().__init__()
        self.autoencoder = autoencoder
        self.entropy_model = entropy_model
        self.quantizer = quantizer

# Load Phase 1 checkpoint
phase1_ckpt = '/root/le-maia/checkpoints/phase1_lambda_0.1/final.pt'
state = torch.load(phase1_ckpt, map_location=device)
print("Keys in Phase1 checkpoint:", state.keys())

# Create a FullCodec model with dummy components to load the state dict
autoencoder = VideoAutoencoder().to(device)
entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
quantizer = Quantizer(num_levels=256, mode='inference').to(device)
full_model = FullCodec(autoencoder, entropy_model, quantizer).to(device)
full_model.load_state_dict(state, strict=False)

# Now extract the entropy model (which now has trained weights)
trained_entropy = full_model.entropy_model

# Test it on a random latent
test_latent = torch.randn(1, 192, 16, 16).to(device)
test_quant = quantizer(test_latent)
with torch.no_grad():
    rate_nats, _ = trained_entropy(test_quant)
    bits = rate_nats.sum().item() * 1.4427  # log2(e)
    print(f"Test rate (bits): {bits:.4f}")

print("Entropy model parameters (first layer):")
for name, param in trained_entropy.named_parameters():
    if 'hyperprior_cnn.0.weight' in name:
        print(f"{name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
        break
