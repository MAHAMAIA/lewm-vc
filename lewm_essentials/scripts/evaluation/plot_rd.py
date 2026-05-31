#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

lewm_bpp = [1.442, 1.149, 0.968, 0.833, 0.737, 0.680]
lewm_psnr = [17.60, 17.55, 17.00, 17.51, 17.75, 16.96]

x265_bpp = [0.3356, 0.1029, 0.0479, 0.0312, 0.0234, 0.0193]
x265_psnr = [34.82, 34.40, 33.84, 33.07, 32.16, 31.08]

plt.figure(figsize=(6, 4))
plt.plot(lewm_bpp, lewm_psnr, 'o-', color='#2A5C82', lw=2, label='LeWM-VC')
plt.plot(x265_bpp, x265_psnr, 's--', color='#CCCCCC', lw=2, label='x265 (veryslow)')
plt.xlabel('Bits Per Pixel (BPP)')
plt.ylabel('Y-PSNR (dB)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/root/le-maia/benchmark_results/rd_curve.png', dpi=150)
print("✅ RD curve saved")
