import os
import sys
import torch
import numpy as np

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer
from lewm_vc.encoder import LeWMEncoder
from lewm_vc.working_decoder import LeWMDecoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load a sample lambda checkpoint (use λ=0.1)
ckpt_path = '/root/le-maia/checkpoints/phase1_lambda_0.1/final.pt'
print(f"Loading checkpoint from {ckpt_path}")
state_dict = torch.load(ckpt_path, map_location=device)
print("Keys in checkpoint:", state_dict.keys())

# Create entropy model and quantizer
entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
quantizer = Quantizer(num_levels=256, mode='inference').to(device)

# Load only the entropy_model part
if 'entropy_model' in state_dict:
    entropy_model.load_state_dict(state_dict['entropy_model'])
    print("Loaded entropy_model from checkpoint")
else:
    # Try to load the whole state dict into a dummy FullCodec and extract
    from lewm_vc.encoder import LeWMEncoder
    from lewm_vc.working_decoder import LeWMDecoder
    class FullCodec(torch.nn.Module):
        def __init__(self, autoencoder, entropy_model, quantizer):
            super().__init__()
            self.autoencoder = autoencoder
            self.entropy_model = entropy_model
            self.quantizer = quantizer
    autoencoder = torch.nn.Module()
    model = FullCodec(autoencoder, entropy_model, quantizer)
    model.load_state_dict(state_dict, strict=False)
    print("Loaded state dict into FullCodec, entropy_model now has parameters")

entropy_model.eval()
quantizer.eval()

# Create a dummy latent (simulate a quantized latent)
dummy_latent = torch.randn(1, 192, 16, 16).to(device)
quantized = quantizer(dummy_latent)

# Run entropy model
with torch.no_grad():
    rate_nats, params = entropy_model(quantized)
    mu = params['mu']
    sigma = params['sigma']

print(f"\nRate (nats) shape: {rate_nats.shape}")
print(f"Rate (nats) sum: {rate_nats.sum().item():.6f}")
print(f"Rate (bits) sum: {rate_nats.sum().item() * np.log2(np.e):.6f}")
print(f"mu: mean={mu.mean().item():.4f}, std={mu.std().item():.4f}")
print(f"sigma: mean={sigma.mean().item():.4f}, std={sigma.std().item():.4f}")
print(f"sigma min: {sigma.min().item():.6f}, sigma max: {sigma.max().item():.6f}")

# Check if any mu or sigma is NaN or Inf
print(f"mu NaN: {torch.isnan(mu).any().item()}, mu Inf: {torch.isinf(mu).any().item()}")
print(f"sigma NaN: {torch.isnan(sigma).any().item()}, sigma Inf: {torch.isinf(sigma).any().item()}")

# Manually compute KL for a few elements to verify formula
kl_elements = 0.5 * (mu**2 + sigma**2 - torch.log(sigma**2) - 1)
print(f"KL element mean: {kl_elements.mean().item():.6f}")
print(f"KL element sum: {kl_elements.sum().item():.6f}")

# Check if the rate_nats matches kl_elements
diff = (rate_nats - kl_elements).abs().max()
print(f"Max diff between rate_nats and computed KL: {diff.item():.6f}")

print("Diagnostic complete.")
