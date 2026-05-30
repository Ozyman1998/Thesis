#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:57:57 2026
@author: nacho
"""
import numpy as np
import matplotlib.pyplot as plt
from pybaselines import Baseline

# ── 1. Load data ──────────────────────────────────────────────────────────────
path = "/home/nacho/Escritorio/THESIS_USB/Z/ATR-FTIR/zflx_thf_2B.CSV"  # <- change path
f = open(path, 'r')
lines = []
for line in f:
    lines.append(line.split(","))
f.close()
l = len(lines)
w, T = [], []
for i in range(0, l - 1):
    w.append(float(lines[i][0]))
    T.append(float(lines[i][1]))
w = np.array(w)
A = 2 - np.log(T)

# ── 2. Baseline correction & normalization ────────────────────────────────────
baseline_fitter = Baseline(x_data=w)
bkg_2, params_2 = baseline_fitter.derpsalsa(A, lam=1e10)
sub = A - bkg_2

norm_wave = 1410
peak_index = np.abs(w - norm_wave).argmin()
norm_factor = sub[peak_index]
if norm_factor != 0:
    sub = sub / norm_factor

# ── 3. Band annotations ───────────────────────────────────────────────────────
bands = [
    dict(wn=815.0,  label="815 cm$^{-1}$\nMDI para-subst.",       color="#6A1B9A", h=0),
    dict(wn=1215.0, label="1215 cm$^{-1}$\nC-N stretch\n(urethane)", color="#00008B", h=1),
    dict(wn=1596.0, label="1596 cm$^{-1}$\nAmide II / C=C MDI",   color="#228B22", h=0),
    dict(wn=1698.0, label="1698 cm$^{-1}$\nC=O H-bonded\n(ordered)", color="#B8860B", h=1),
    dict(wn=2932.0, label="2932 cm$^{-1}$\nCH$_2$ str.\n(soft seg.)", color="#CC5500", h=0),
    dict(wn=3310.0, label="3310 cm$^{-1}$\nN-H stretch\n(urethane)", color="#8B0000", h=1),
]
for b in bands:
    idx = np.argmin(np.abs(w - b['wn']))
    b['abs'] = sub[idx]
    b['wn_actual'] = w[idx]

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(w, sub, color='#1f77b4', lw=1.4)
ax.set_xlim([3500, 500])
ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=27)
ax.set_ylabel('Normalized absorbance (a.u.)', fontsize=27)
ax.set_title('Taglus (hTPU)', fontsize=30, fontweight='bold')
ax.tick_params(axis='both', labelsize=25)

y_max = sub[(w >= 500) & (w <= 3500)].max()
y_min = sub[(w >= 500) & (w <= 3500)].min()
y_range = y_max - y_min

label_heights = [y_max + y_range * 0.18,
                 y_max + y_range * 0.38]

for b in bands:
    x0    = b['wn_actual']
    y0    = b['abs']
    ytext = label_heights[b['h']]
    ax.axvline(x0, color=b['color'], lw=0.9, ls='--', alpha=0.6)
    ax.scatter(x0, y0, color=b['color'], s=35, zorder=5,
               edgecolors='black', linewidths=0.4)
    ax.annotate(
        b['label'],
        xy=(x0, y0), xytext=(x0, ytext),
        fontsize=11, color=b['color'], ha='center', va='bottom',
        fontweight='bold', rotation=90,
        arrowprops=dict(arrowstyle='-', color=b['color'], lw=0.7),
        annotation_clip=False, linespacing=1.4,
    )

ax.set_ylim([y_min - y_range * 0.05, y_max + y_range * 0.75])

plt.tight_layout()
plt.show()