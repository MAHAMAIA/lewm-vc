#!/usr/bin/env python3
"""Lambda sweep for trained GMM model - WITH LOGGING"""

import os
import sys
import torch
import numpy as np
from datetime import datetime
from train_gmm import VideoAutoencoder, GMMEntropyModel, gmm_likelihood_discrete, val_loader, device, QUANT_STEP

# Create log directory
log_dir = '/root/le-maia/gmm_sweep_logs'
os.makedirs(log_dir, exist_ok=True)

# Timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'gmm_sweep_{timestamp}.txt')

lambdas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
results = []

# Load best model
print("Loading best GMM model from checkpoints_gmm/")
autoencoder = VideoAutoencoder().to(device)
entropy_model = GMMEntropyModel().to(device)

ae_path = 'checkpoints_gmm/ae_best.pt'
ent_path = 'checkpoints_gmm/entropy_best.pt'

if not os.path.exists(ae_path) or not os.path.exists(ent_path):
    print(f"ERROR: Checkpoints not found in checkpoints_gmm/")
    print(f"  - {ae_path}: {os.path.exists(ae_path)}")
    print(f"  - {ent_path}: {os.path.exists(ent_path)}")
    sys.exit(1)

autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
entropy_model.load_state_dict(torch.load(ent_path, map_location=device))
autoencoder.eval()
entropy_model.eval()

print(f"Model loaded. Validation set size: {len(val_loader)} batches")
print(f"Logging to: {log_file}")

# Write header to log file
with open(log_file, 'w') as f:
    f.write(f"GMM Lambda Sweep Results\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Model: checkpoints_gmm/ae_best.pt\n")
    f.write(f"Lambda values: {lambdas}\n")
    f.write(f"Validation batches: {len(val_loader)}\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"{'λ':<10} {'bpp':<15} {'PSNR (dB)':<12} {'MSE':<12}\n")
    f.write(f"{'-'*50}\n")

print("\n" + "="*60)
print("Running λ sweep...")
print(f"{'λ':<10} {'bpp':<15} {'PSNR (dB)':<12}")
print("-"*40)

for lam in lambdas:
    total_rate = 0
    total_mse = 0
    total_psnr = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = batch.to(device)
            B, T, C, H, W = batch.shape
            flat = batch.view(B*T, C, H, W)
            
            # Encode and quantize
            latent = autoencoder.encode(flat)
            quantized = torch.round(latent / QUANT_STEP) * QUANT_STEP
            
            # Get GMM parameters
            mu, scale, weight = entropy_model(quantized)
            
            # Compute NLL (nats per latent element)
            nll = gmm_likelihood_discrete(quantized, mu, scale, weight, step=QUANT_STEP)
            
            # Convert to bits per original pixel
            bits_per_latent = nll / np.log(2)
            latent_elements = quantized.numel()
            original_pixels = flat.numel()
            rate_bpp = bits_per_latent * (latent_elements / original_pixels)
            
            # Decode and compute MSE
            recon = autoencoder.decode(quantized, target_size=(H, W))
            recon = recon.view(B, T, C, H, W)
            mse = torch.nn.functional.mse_loss(recon, batch).item()
            
            total_rate += rate_bpp.item()
            total_mse += mse
            num_batches += 1
    
    avg_rate = total_rate / num_batches
    avg_mse = total_mse / num_batches
    psnr = 10 * np.log10(1.0 / avg_mse) if avg_mse > 0 else 0
    
    results.append((lam, avg_rate, psnr, avg_mse))
    
    # Print to console
    print(f"{lam:<10} {avg_rate:<15.4f} {psnr:<12.2f}")
    
    # Write to log file
    with open(log_file, 'a') as f:
        f.write(f"{lam:<10} {avg_rate:<15.6f} {psnr:<12.2f} {avg_mse:<12.6f}\n")

# Summary
print("\n" + "="*60)
print("RD Curve Summary:")
print("λ\tbpp\t\tPSNR(dB)")
for r in results:
    print(f"{r[0]}\t{r[1]:.6f}\t{r[2]:.2f}")

# Write summary to log
with open(log_file, 'a') as f:
    f.write(f"\n{'='*60}\n")
    f.write(f"Summary:\n")
    f.write(f"Best PSNR: {max(r[2] for r in results):.2f} dB\n")
    f.write(f"Lowest bitrate: {min(r[1] for r in results):.4f} bpp\n")
    f.write(f"Average bitrate: {sum(r[1] for r in results)/len(results):.4f} bpp\n")

print(f"\nLog saved to: {log_file}")

# Also save as CSV for easy import
csv_file = os.path.join(log_dir, f'gmm_sweep_{timestamp}.csv')
with open(csv_file, 'w') as f:
    f.write("lambda,bpp,psnr,mse\n")
    for r in results:
        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
print(f"CSV saved to: {csv_file}")

# Print final status
print("\n" + "="*60)
print("Verification checks:")
print(f"  - All bitrates positive? {all(r[1] > 0 for r in results)}")
print(f"  - Bitrate varies with λ? {len(set(round(r[1], 3) for r in results)) > 1}")
print(f"  - PSNR range: {min(r[2] for r in results):.2f} - {max(r[2] for r in results):.2f} dB")
