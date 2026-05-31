# test_entropy.py
import torch
import sys
sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.entropy import HyperpriorEntropy
from lewm_vc.quant import Quantizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
entropy_ckpt = '/root/le-maia/checkpoints_entropy_affine/entropy_final.pt'
entropy_model.load_state_dict(torch.load(entropy_ckpt, map_location=device))
entropy_model.eval()
quantizer = Quantizer(num_levels=256, mode='inference').to(device)

# Generate a random latent with similar statistics to affine output (mean ~0.05, std ~0.45)
latent = torch.randn(1, 192, 16, 16).to(device) * 0.45 + 0.05
quantized = quantizer(latent)
with torch.no_grad():
    rate_nats, _ = entropy_model(quantized)
    bits = rate_nats.sum().item() * 1.4427
    print(f"Rate bits: {bits:.4f}")
    print(f"Quantized unique values: {torch.unique(quantized).numel()}")
