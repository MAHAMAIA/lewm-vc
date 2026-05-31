#!/usr/bin/env python3
import csv
import numpy as np
from scipy.interpolate import interp1d

def bd_rate(rate1, psnr1, rate2, psnr2):
    # rate in bpp, psnr in dB
    # Sort by PSNR
    idx1 = np.argsort(psnr1)
    idx2 = np.argsort(psnr2)
    psnr1_s = np.array(psnr1)[idx1]
    psnr2_s = np.array(psnr2)[idx2]
    rate1_s = np.array(rate1)[idx1]
    rate2_s = np.array(rate2)[idx2]

    # Common PSNR range
    psnr_min = max(psnr1_s.min(), psnr2_s.min())
    psnr_max = min(psnr1_s.max(), psnr2_s.max())
    if psnr_min >= psnr_max:
        return float('nan')
    psnr_interp = np.linspace(psnr_min, psnr_max, 100)
    rate1_interp = np.interp(psnr_interp, psnr1_s, np.log(rate1_s))
    rate2_interp = np.interp(psnr_interp, psnr2_s, np.log(rate2_s))
    avg_diff = np.mean(rate2_interp - rate1_interp)
    return (np.exp(avg_diff) - 1) * 100

# Read CSV
csv_path = '/root/le-maia/benchmark_results/rd_curve_final.csv'
lewm_bpp = []
lewm_psnr = []
x265_bpp = []
x265_psnr = []
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if not row:
            continue
        if row[0] == 'x265 CRF':
            break
        try:
            lam = float(row[0])
            lewm_bpp.append(float(row[1]))
            lewm_psnr.append(float(row[2]))
        except:
            pass
    for row in reader:
        if not row:
            continue
        if row[0].startswith('x265'):
            x265_bpp.append(float(row[1]))
            x265_psnr.append(float(row[2]))
        else:
            break

bd = bd_rate(lewm_bpp, lewm_psnr, x265_bpp, x265_psnr)
print(f"BD-rate (LeWM-VC vs x265): {bd:.2f}% (negative = better)")
