#!/usr/bin/env python3
import sys
import torch
from train_joint_phase0 import *  # import all classes, dataset, val_loader, etc.

# Load final checkpoint for λ=0.05
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

# Run validation exactly as in training (using val_loader)
val_mse = 0
val_rate = 0
val_batches = 0
with torch.no_grad():
    for val_batch in val_loader:
        val_batch = val_batch.to(device)
        Bv, Tv, Cv, Hv, Wv = val_batch.shape
        val_flat = val_batch.view(Bv*Tv, Cv, Hv, Wv)
        latent_val = autoencoder.encode(val_flat)
        quant_val = quantize_with_temp(latent_val, QUANT_STEP, temp=0.0)   # same as training validation
        mu_val, log_scale_val = entropy_model(quant_val)
        nll_val = laplace_likelihood_discrete(quant_val, mu_val, log_scale_val, step=QUANT_STEP)
        rate_val = (nll_val * quant_val.numel() / np.log(2)) / val_flat.numel()
        recon_val = autoencoder.decode(quant_val, target_size=(Hv, Wv))
        recon_val = recon_val.view(Bv, Tv, Cv, Hv, Wv)
        val_mse += criterion_mse(recon_val, val_batch).item()
        val_rate += rate_val.item()
        val_batches += 1
val_psnr = 10 * np.log10(1.0 / (val_mse / val_batches))
val_bpp = val_rate / val_batches
print(f"Validation on training split: PSNR={val_psnr:.2f} dB, BPP={val_bpp:.4f}")
