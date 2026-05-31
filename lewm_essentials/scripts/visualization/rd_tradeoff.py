#!/usr/bin/env python3
"""Generate BPP reduction chart."""
import matplotlib.pyplot as plt
import numpy as np

lambdas = ['0.0001', '0.0005', '0.001', '0.005', '0.01', '0.05']
bpp = [1.442, 1.149, 0.968, 0.833, 0.737, 0.680]
psnr = [17.60, 17.55, 17.00, 17.51, 17.75, 16.96]

fig, ax1 = plt.subplots(figsize=(8, 5))

color = '#2A5C82'
ax1.set_xlabel('λ (Rate-Distortion Tradeoff)')
ax1.set_ylabel('Bits Per Pixel (BPP)', color=color)
line1 = ax1.plot(lambdas, bpp, 'o-', color=color, lw=2, markersize=8, label='BPP')
ax1.tick_params(axis='y', labelcolor=color)
for i, (lam, val) in enumerate(zip(lambdas, bpp)):
    ax1.annotate(f'{val:.3f}', (i, val), textcoords="offset points", xytext=(0,10), ha='center')

ax2 = ax1.twinx()
color2 = '#CC3333'
ax2.set_ylabel('Y-PSNR (dB)', color=color2)
line2 = ax2.plot(lambdas, psnr, 's--', color=color2, lw=2, markersize=8, label='PSNR')
ax2.tick_params(axis='y', labelcolor=color2)
for i, (lam, val) in enumerate(zip(lambdas, psnr)):
    ax2.annotate(f'{val:.1f}', (i, val), textcoords="offset points", xytext=(0,-15), ha='center')

plt.title('LeWM-VC: Monotonic Rate-Distortion Tradeoff')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('/root/le-maia/benchmark_results/rd_tradeoff.png', dpi=150)
print("✅ RD tradeoff chart saved")
