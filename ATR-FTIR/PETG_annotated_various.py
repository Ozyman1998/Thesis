#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pybaselines import Baseline

# ── CONFIGURACIÓN ─────────────────────────────────────────────
spectra_config = [
    {
        'path':  '/home/nacho/Escritorio/THESIS_USB/Z/ATR-FTIR/zflx_thf_2B.CSV',
        'label': 'As received',
        'color': '#1f77b4',
    },
    {
        'path':  '/ruta/al/segundo/espectro.CSV',
        'label': 'Water immersion for 14 days',
        'color': '#ff7f0e',
    },
]

TITLE   = 'Taglus (hTPU)'
NORM_WN = 1410

bands = [
    dict(wn=815.0,  label="815 cm$^{-1}$\nMDI para-subst.",         color="#6A1B9A", h=0),
    dict(wn=1215.0, label="1215 cm$^{-1}$\nC-N stretch\n(urethane)", color="#00008B", h=1),
    dict(wn=1596.0, label="1596 cm$^{-1}$\nAmide II / C=C MDI",     color="#228B22", h=0),
    dict(wn=1698.0, label="1698 cm$^{-1}$\nC=O H-bonded\n(ordered)", color="#B8860B", h=1),
    dict(wn=2932.0, label="2932 cm$^{-1}$\nCH$_2$ str.\n(soft seg.)", color="#CC5500", h=0),
    dict(wn=3310.0, label="3310 cm$^{-1}$\nN-H stretch\n(urethane)", color="#8B0000", h=1),
]

# ── FUNCIÓN DE CARGA Y PROCESADO ──────────────────────────────
def load_and_process(path, norm_wn=NORM_WN):
    w, T = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                w.append(float(parts[0]))
                T.append(float(parts[1]))
            except ValueError:
                continue
    w = np.array(w)
    T = np.array(T)
    A = 2 - np.log(T)
    baseline_fitter = Baseline(x_data=w)
    bkg, _ = baseline_fitter.derpsalsa(A, lam=1e10)
    sub = A - bkg
    peak_index = np.abs(w - norm_wn).argmin()
    norm_factor = sub[peak_index]
    if norm_factor != 0:
        sub = sub / norm_factor
    return w, sub

# ── CARGA DE ESPECTROS ────────────────────────────────────────
loaded = []
for cfg in spectra_config:
    w, sub = load_and_process(cfg['path'])
    loaded.append({'w': w, 'sub': sub,
                   'label': cfg['label'],
                   'color': cfg['color']})

# ── RANGO Y ALTURAS DE ETIQUETAS (primer espectro como ref.) ──
w_ref   = loaded[0]['w']
sub_ref = loaded[0]['sub']

mask_plot = (w_ref >= 500) & (w_ref <= 3500)
y_max   = sub_ref[mask_plot].max()
y_min   = sub_ref[mask_plot].min()
y_range = y_max - y_min

label_heights = [
    y_max + y_range * 0.18,
    y_max + y_range * 0.38,
]

for b in bands:
    idx = np.argmin(np.abs(w_ref - b['wn']))
    b['abs']       = sub_ref[idx]
    b['wn_actual'] = w_ref[idx]

# ── PLOT ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for s in loaded:
    ax.plot(s['w'], s['sub'],
            color=s['color'],
            lw=1.4,
            label=s['label'])

ax.set_xlim([3500, 500])
ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=27)
ax.set_ylabel('Normalized absorbance (a.u.)', fontsize=27)
ax.set_title(TITLE, fontsize=30, fontweight='bold')
ax.tick_params(axis='both', labelsize=25)
ax.set_ylim([y_min - y_range * 0.05,
             y_max + y_range * 0.75])

for b in bands:
    x0    = b['wn_actual']
    y0    = b['abs']
    ytext = label_heights[b['h']]
    ax.axvline(x0, color=b['color'], lw=0.9,
               ls='--', alpha=0.6)
    ax.scatter(x0, y0, color=b['color'], s=35,
               zorder=5, edgecolors='black', linewidths=0.4)
    ax.annotate(
        b['label'],
        xy=(x0, y0), xytext=(x0, ytext),
        fontsize=11, color=b['color'],
        ha='center', va='bottom',
        fontweight='bold', rotation=90,
        arrowprops=dict(arrowstyle='-',
                        color=b['color'], lw=0.7),
        annotation_clip=False, linespacing=1.4,
    )

ax.legend(fontsize=13, loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.savefig('ftir_tpu_comparison.png',
            dpi=300, bbox_inches='tight',
            facecolor='white')
plt.show()
print("Saved.")
