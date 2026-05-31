import sys
import torch
from train_joint_phase0 import *

# Load checkpoint for λ=0.05
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

# Get one batch from val_loader
val_batch = next(iter(val_loader))
val_batch = val_batch.to(device)
B, T, C, H, W = val_batch.shape
val_flat = val_batch.view(B*T, C, H, W)

with torch.no_grad():
    latent_val = autoencoder.encode(val_flat)
    # Use quantize_with_temp with temp=0.0 (hard rounding) as in training validation
    quant_val = quantize_with_temp(latent_val, QUANT_STEP, temp=0.0)
    mu_val, log_scale_val = entropy_model(quant_val)
    nll_val = laplace_likelihood_discrete(quant_val, mu_val, log_scale_val, step=QUANT_STEP, sigma_floor=0.003)
    print(f"nll mean: {nll_val.item():.6f}")
    print(f"log_scale mean: {log_scale_val.mean().item():.4f}")
    rate_val = (nll_val * quant_val.numel() / np.log(2)) / val_flat.numel()
    print(f"rate per pixel: {rate_val.item():.6f}")
