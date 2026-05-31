#!/usr/bin/env python3
"""Benchmark decode latency on MI300X."""
import torch, time, sys
import numpy as np

sys.path.insert(0, '/root/le-maia/src')
from lewm_vc.working_decoder import LeWMDecoder

device = torch.device('cuda')
dtype = torch.bfloat16

# ✅ FIXED: Remove num_layers argument
decoder = LeWMDecoder(latent_dim=192, hidden_dim=512).to(device).to(dtype)
decoder.eval()

# Dummy latent (batch=1, 192 channels, 16x16)
latent = torch.randn(1, 192, 16, 16).to(device).to(dtype)

# Warmup
print("Warming up...")
for _ in range(50):
    _ = decoder(latent, target_size=(256,256))
torch.cuda.synchronize()

# Benchmark
print("Benchmarking...")
times = []
for _ in range(200):
    start = time.time()
    _ = decoder(latent, target_size=(256,256))
    torch.cuda.synchronize()
    times.append(time.time() - start)

fps = 1.0 / np.mean(times)
ms = np.mean(times) * 1000
print(f"\n✅ Decode latency: {ms:.2f} ms/frame ({fps:.2f} FPS)")

# Save result
with open('/root/le-maia/benchmark_results/latency.txt', 'w') as f:
    f.write(f"Decode latency: {ms:.2f} ms/frame\n")
    f.write(f"Decode FPS: {fps:.2f}\n")
    f.write(f"Hardware: {torch.cuda.get_device_name(0)}\n")
print("✅ Saved to benchmark_results/latency.txt")
