#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:46:34 2026

@author: ozymandias
"""

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ── CONFIGURACIÓN ─────────────────────────────────────────────
files = {
    '0d': '/media/ozymandias/489F-4801/Z/indirecto_FTIR/secon_try/medio.CSV',
    '1d': '/media/ozymandias/489F-4801/Z/indirecto_FTIR/secon_try/MP100_1d.CSV',
    '3d': '/media/ozymandias/489F-4801/Z/indirecto_FTIR/secon_try/MP100_3d.CSV',
    '7d': '/media/ozymandias/489F-4801/Z/indirecto_FTIR/secon_try/MP100_7d.CSV',
}

colors = {
    '0d': '#1f77b4',
    '1d': '#ff7f0e',
    '3d': '#2ca02c',
    '7d': '#d62728',
}

TITLE    = 'ATR-FTIR spectra of culture media\nexposed to MP100 copolyester for 1, 3 and 7 days'
SMOOTH   = True
NORM_WN  = 1650   # ← cambia aquí el número de onda de normalización
                   # opciones típicas para medio de cultivo:
                   # 1650 = Amide I (proteínas)
                   # 3300 = O-H/N-H stretching (agua)
                   # 2900 = C-H stretching

# ── FUNCIONES ─────────────────────────────────────────────────
def load_csv(path):
    w, A = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                w.append(float(parts[0]))
                A.append(float(parts[1]))
            except ValueError:
                continue
    w = np.array(w)
    A = np.array(A)
    order = np.argsort(w)
    return w[order], A[order]

def normalize(w, A, norm_wn):
    """Normaliza al valor de absorbancia en norm_wn."""
    idx = np.abs(w - norm_wn).argmin()
    nf  = A[idx]
    if nf != 0:
        A = A / nf
    return A

# ── PLOT ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

for label, path in files.items():
    w, A = load_csv(path)
    if SMOOTH:
        A = savgol_filter(A, window_length=21, polyorder=3)
    A = normalize(w, A, NORM_WN)
    ax.plot(w, A,
            color=colors[label],
            lw=1.2,
            label=label)

ax.set_xlim([3500, 500])
ax.set_xlabel('wavenumber (cm$^{-1}$)', fontsize=22)
ax.set_ylabel('Normalized absorbance (a.u.)', fontsize=22)
#ax.set_title(TITLE, fontsize=13)
ax.tick_params(axis='both', labelsize=20)
ax.legend(fontsize=20, loc='upper center', framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
#plt.savefig('FTIR_medium_MP100_normalized.png',
#            dpi=300, bbox_inches='tight',
#            facecolor='white')
plt.show()
print("Saved.")
