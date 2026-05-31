import sys
import torch
from train_joint_phase0 import *

lambda_list = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
checkpoint_dir = '/root/le-maia/checkpoints_joint_phase0'

results = []
for lam in lambda_list:
    ae_ckpt = os.path.join(checkpoint_dir, f'ae_lambda_{lam}_best.pt')
    ent_ckpt = os.path.join(checkpoint_dir, f'entropy_lambda_{lam}_best.pt')
    if not os.path.exists(ae_ckpt) or not os.path.exists(ent_ckpt):
        print(f"Missing best checkpoints for λ={lam}, skipping")
        continue
    print(f"Loading λ={lam} best...")
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
    val_mse = 0
    val_rate = 0
    val_batches = 0
    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = val_batch.to(device)
            Bv, Tv, Cv, Hv, Wv = val_batch.shape
            val_flat = val_batch.view(Bv*Tv, Cv, Hv, Wv)
            latent_val = autoencoder.encode(val_flat)
            quant_val = quantize_with_temp(latent_val, QUANT_STEP, temp=0.0)
            mu_val, log_scale_val = entropy_model(quant_val)
            nll_val = laplace_likelihood_discrete(quant_val, mu_val, log_scale_val, step=QUANT_STEP, sigma_floor=0.003)
            rate_val = (nll_val * quant_val.numel() / np.log(2)) / val_flat.numel()
            recon_val = autoencoder.decode(quant_val, target_size=(Hv, Wv))
            recon_val = recon_val.view(Bv, Tv, Cv, Hv, Wv)
            val_mse += criterion_mse(recon_val, val_batch).item()
            val_rate += rate_val.item()
            val_batches += 1
    val_psnr = 10 * np.log10(1.0 / (val_mse / val_batches))
    val_bpp = val_rate / val_batches
    results.append((lam, val_bpp, val_psnr))
    print(f"λ={lam}: bpp={val_bpp:.4f}, PSNR={val_psnr:.2f} dB")

import csv
os.makedirs('/root/le-maia/benchmark_results', exist_ok=True)
with open('/root/le-maia/benchmark_results/rd_curve_correct.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['λ', 'bpp', 'PSNR'])
    for lam, bpp, psnr in results:
        writer.writerow([lam, bpp, psnr])
print("Results saved to /root/le-maia/benchmark_results/rd_curve_correct.csv")
