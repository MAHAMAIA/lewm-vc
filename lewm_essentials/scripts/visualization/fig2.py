#!/usr/bin/env python3
"""Fix Fig 2 labels."""
import matplotlib.pyplot as plt

PRIMARY = '#2A5C82'
GRAY = '#CCCCCC'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Parameters (learned codecs only)
models_params = ['DCVC', 'VCT', 'LeWM-VC\n(Ours)']
params = [8.5, 12.3, 1.2]
bars1 = ax1.bar(models_params, params, color=[GRAY, GRAY, PRIMARY])
ax1.set_ylabel('Decoder Parameters (M)', fontsize=12)
ax1.set_title('Decoder Complexity', fontsize=13)
for bar, val in zip(bars1, params):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
            f'{val}M', ha='center', va='bottom', fontsize=10)

# FPS (include VTM for speed comparison)
models_fps = ['VTM\n(VVC)', 'DCVC', 'VCT', 'LeWM-VC\n(Ours)']
fps = [0.3, 45, 30, 749]
bars2 = ax2.bar(models_fps, fps, color=[GRAY, GRAY, GRAY, PRIMARY])
ax2.set_ylabel('Decode FPS (256×256)', fontsize=12)
ax2.set_title('Edge Inference Speed', fontsize=13)
ax2.set_yscale('log')
for bar, val in zip(bars2, fps):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15, 
            f'{val}', ha='center', va='bottom', fontsize=10)

plt.suptitle('LeWM-VC: 600× Faster than VVC with 7× Fewer Parameters', fontsize=14)
plt.tight_layout()
plt.savefig('/root/le-maia/benchmark_results/fig2_efficiency_fixed.png', dpi=200)
print("✅ Fixed efficiency chart saved")
