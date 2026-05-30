#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:57:57 2026
@author: nacho
"""
import numpy as np 
import matplotlib.pyplot as plt
from pybaselines import Baseline
from scipy.integrate import trapezoid

path = "/home/nacho/Escritorio/THESIS_USB/Z/ATR-FTIR/zflx_thf_1.CSV"
f = open(path, 'r+')
lines = []
w = []
T = []
for line in f: 
    lines.append(line.split(","))
l = len(lines)
for i in range(0, l - 1):
    w.append(float(lines[i][0]))
    T.append(float(lines[i][1]))
w = np.array(w)
A = 2 - np.log(T)

baseline_fitter = Baseline(x_data=w)
bkg_2, params_2 = baseline_fitter.derpsalsa(A, lam=1E9)
sub = A - bkg_2

norm_wave = 725
peak_index = np.abs(w - norm_wave).argmin()
norm_factor = sub[peak_index]
if norm_factor != 0:
    sub = sub / norm_factor

# ── Band annotations ──────────────────────────────────────────────────────────
bands = [
    dict(wn=722.7,  label="722 cm$^{-1}$\nC-H arom. o.o.p.",  color="#8B0000", h=0),
    dict(wn=1014.4, label="1014 cm$^{-1}$\nCHDM cis",          color="#DC143C", h=1),
    dict(wn=1039.9, label="1040 cm$^{-1}$\nCHDM trans",        color="#228B22", h=0),
    dict(wn=1240.0, label="1240 cm$^{-1}$\nC-O-C asym.",       color="#00008B", h=1),
    dict(wn=1407.3, label="1407 cm$^{-1}$\nAromatic ring",     color="#4B0082", h=0),
    dict(wn=1709.1, label="1709 cm$^{-1}$\nC=O ester",         color="#B8860B", h=1),
]
for b in bands:
    idx = np.argmin(np.abs(w - b['wn']))
    b['abs'] = sub[idx]
    b['wn_actual'] = w[idx]

# ── Cis/Trans analysis ────────────────────────────────────────────────────────
def peak_area(wn, A, center, half_width=10):
    mask = (wn >= center - half_width) & (wn <= center + half_width)
    return trapezoid(A[mask], wn[mask])

abs_cis   = next(b['abs'] for b in bands if 'cis'   in b['label'] and 'trans' not in b['label'])
abs_trans = next(b['abs'] for b in bands if 'trans' in b['label'])
area_cis   = peak_area(w, sub, 1014.4)
area_trans = peak_area(w, sub, 1039.9)
ratio_abs  = abs_trans / abs_cis
ratio_area = area_trans / area_cis

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(w, sub, color='#1f77b4', lw=1.4)
ax.set_xlim([3500, 500])
ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=27)
ax.set_ylabel('Normalized absorbance (a.u.)', fontsize=27)
ax.set_title('Outer layer of Zendura FLX after THF dissolution', fontsize=30, fontweight='bold')
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
        fontsize=9, color=b['color'], ha='center', va='bottom',
        fontweight='bold', rotation=90,
        arrowprops=dict(arrowstyle='-', color=b['color'], lw=0.7),
        annotation_clip=False, linespacing=1.4,
    )

ax.set_ylim([y_min - y_range * 0.05, y_max + y_range * 0.75])

# Cis/trans result box


plt.tight_layout()
plt.show()