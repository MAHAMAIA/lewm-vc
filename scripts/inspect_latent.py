# inspect_latent.py
import torch
import cv2
import numpy as np
import sys
sys.path.insert(0, '/root/le-maia/src')
from final_benchmark_affine_entropy import VideoAutoencoderWithAffine, device

# Load affine autoencoder
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
    print(f"Latent norm: mean={latent_norm.mean().item():.4f}, std={latent_norm.std().item():.4f}")
    # Quantize with the same quantizer as benchmark
    from lewm_vc.quant import Quantizer
    quantizer = Quantizer(num_levels=256, mode='inference').to(device)
    quantized = quantizer(latent_norm)
    print(f"Quantized: unique values={torch.unique(quantized).numel()}, min={quantized.min().item()}, max={quantized.max().item()}")
    # Load entropy model and compute rate
    from lewm_vc.entropy import HyperpriorEntropy
    entropy_model = HyperpriorEntropy(latent_dim=192, hyper_channels=320).to(device)
    entropy_ckpt = '/root/le-maia/checkpoints_entropy_affine/entropy_final.pt'
    entropy_model.load_state_dict(torch.load(entropy_ckpt, map_location=device))
    entropy_model.eval()
    rate_nats, _ = entropy_model(quantized)
    bits = rate_nats.sum().item() * 1.4427
    print(f"Rate bits: {bits:.4f}")
