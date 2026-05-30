#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:06:05 2026

@author: nacho
"""

"""
FTIR ATR — Clasificador automático de TPU
==========================================
  · Lee archivos .txt (o .CSV) de OMNIC / FTIR con dos columnas:
      wavenumber (cm⁻¹)  |  %Transmitancia
  · Corrección de línea base (pybaselines - derpsalsa)
  · Normalización espectral a la banda C=O/Amida (1500-1760 cm⁻¹)
  · Diagnóstico THF residual (muestras drop-cast desde THF)
  · Clasificación segmento blando:  POLIÉSTER  /  POLIÉTER
  · Clasificación diisocianato:     MDI / TDI / HDI
  · Deconvolución carbonilo C=O → DPS, DOR
  · Espectro anotado (estilo publicación)
  · Panel diagnóstico 4 subplots

Uso:
    python ftir_tpu_classifier.py spectrum.txt
    python ftir_tpu_classifier.py          # usa la ruta por defecto

Dependencias:
    pip install numpy matplotlib scipy lmfit pybaselines
"""

# ── CRÍTICO: fijar backend ANTES de cualquier otro import de matplotlib
import matplotlib
matplotlib.use('Qt5Agg')   # Si falla, prueba: 'TkAgg'

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter, find_peaks
from lmfit.models import GaussianModel, ConstantModel

warnings.filterwarnings("ignore")

try:
    from pybaselines import Baseline
    _HAS_PYBASELINES = True
except ImportError:
    _HAS_PYBASELINES = False
    print("  [AVISO] pybaselines no instalado — se usará corrección lineal simple.\n"
          "          Instala con: pip install pybaselines\n")

# =============================================================================
# CONSTANTE
# =============================================================================

K_MOLAR = 1.2          # ratio extinción H-bonded / free C=O  (Ernzen et al., 2022)

# =============================================================================
# 1. UTILIDADES ESPECTRALES
# =============================================================================

def band_max(wn, A, center, half_width=20):
    mask = (wn >= center - half_width) & (wn <= center + half_width)
    return float(A[mask].max()) if mask.any() else 0.0


def band_area(wn, A, lo, hi):
    mask = (wn >= lo) & (wn <= hi)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(A[mask], wn[mask]))


def has_real_peak(wn, A, center, hw=25, min_prom=0.04):
    mask = (wn >= center - hw) & (wn <= center + hw)
    if mask.sum() < 5:
        return False, 0.0, 0.0
    seg  = savgol_filter(A[mask], window_length=min(11, mask.sum() | 1), polyorder=2)
    pksi, props = find_peaks(seg, prominence=min_prom)
    if len(pksi) == 0:
        return False, 0.0, 0.0
    best = pksi[np.argmax(seg[pksi])]
    return True, float(wn[mask][best]), float(seg[best])


# =============================================================================
# 2. CARGA Y PREPROCESADO
# =============================================================================

def load_and_preprocess(filepath):
    data = None
    for sep in [',', ';', '\t', ' ']:
        try:
            d = np.loadtxt(filepath, delimiter=sep, comments=['#', '"'])
            if d.ndim == 2 and d.shape[1] >= 2:
                data = d
                break
        except Exception:
            continue
    if data is None:
        import pandas as pd
        df = pd.read_csv(filepath, sep=None, engine='python', comment='#', header='infer')
        data = df.values[:, :2].astype(float)

    wn  = data[:, 0]
    raw = data[:, 1]

    idx = np.argsort(wn)
    wn  = wn[idx]
    raw = raw[idx]

    T      = np.clip(raw, 0.01, 99.99)
    Ab     = -np.log10(T / 100.0)

    if _HAS_PYBASELINES:
        fitter  = Baseline(x_data=wn)
        bkg, _  = fitter.derpsalsa(Ab, lam=1e5, k=2.0, p=0.01)
        A_corr  = np.clip(Ab - bkg, 0, None)
    else:
        bkg    = np.linspace(Ab[0], Ab[-1], len(Ab))
        A_corr = np.clip(Ab - bkg, 0, None)

    mask_norm = (wn >= 1500) & (wn <= 1760)
    norm_val  = A_corr[mask_norm].max() if mask_norm.any() else 1.0
    if norm_val < 1e-6:
        norm_val = 1.0
    A_norm = A_corr / norm_val

    idx_desc = np.argsort(wn)[::-1]
    return wn[idx_desc], A_norm[idx_desc], A_corr[idx_desc], norm_val


# =============================================================================
# 3. DIAGNÓSTICO DE THF RESIDUAL
# =============================================================================

def check_thf_contamination(wn, A):
    A_913  = band_max(wn, A, 913, 15)
    A_815  = band_max(wn, A, 815, 15)
    A_1072 = band_max(wn, A, 1072, 20)
    ratio  = A_913 / A_815 if A_815 > 0 else 0.0
    peak_913, _, _ = has_real_peak(wn, A, 913, hw=20, min_prom=0.04)

    if ratio < 0.30 and not peak_913:
        level = 'clean'
        msg   = f"Sin THF detectable (A913/A815 = {ratio:.3f})"
    elif ratio < 0.80:
        level = 'moderate'
        msg   = (f"THF MODERADO (A913/A815 = {ratio:.3f}). "
                 f"Banda 1072 cm-1 (A={A_1072:.3f}) puede solapar con eter. "
                 f"Criterios C3/C4 marcados como inciertos.")
    else:
        level = 'severe'
        msg   = (f"THF SEVERO (A913/A815 = {ratio:.3f}). "
                 f"Clasificacion segmento blando NO FIABLE. "
                 f"Secar a vacio 40-50 C, 48 h antes de reanálisis.")

    return {'A_913': round(A_913, 4), 'A_815': round(A_815, 4),
            'A_1072': round(A_1072, 4), 'ratio': round(ratio, 4),
            'peak_913': peak_913, 'level': level, 'message': msg}


# =============================================================================
# 4. CLASIFICACIÓN SEGMENTO BLANDO
# =============================================================================

def classify_soft_segment(wn, A, thf_level='clean'):
    score_ester = 0
    score_ether = 0
    evidence    = {}

    ester_peak, wn_peak, _ = has_real_peak(wn, A, 1730, hw=25, min_prom=0.08)
    mask_co  = (wn >= 1680) & (wn <= 1760)
    Amax_co  = A[mask_co].max() if mask_co.any() else 1.0
    A1730    = band_max(wn, A, 1730, 15)
    A1700    = band_max(wn, A, 1700, 15)
    A1240    = band_max(wn, A, 1240, 20)
    A1170    = band_max(wn, A, 1170, 20)
    A1110    = band_max(wn, A, 1110, 20)
    asymmetry       = A1730 / Amax_co if Amax_co > 0 else 0
    ratio_1240_1110 = A1240 / A1110   if A1110 > 0.01 else 0
    ratio_1170_1110 = A1170 / A1110   if A1110 > 0.01 else 0
    ratio_1730_1700 = A1730 / A1700   if A1700 > 0.01 else 0

    if ester_peak:
        score_ester += 3
        evidence['C1 Pico C=O ester ~1730'] = f"SI a {wn_peak:.1f} cm-1 -> ESTER (+3)"
    else:
        score_ether += 1
        evidence['C1 Pico C=O ester ~1730'] = "NO resuelto -> no concluyente (+1 eter)"

    if asymmetry > 0.65:
        score_ester += 3
        evidence['C2 Asimetria A(1730)/A(max C=O)'] = f"{asymmetry:.3f} > 0.65 -> ESTER (+3)"
    elif asymmetry > 0.50:
        score_ester += 1
        evidence['C2 Asimetria A(1730)/A(max C=O)'] = f"{asymmetry:.3f} -> moderada (+1 ester)"
    else:
        score_ether += 2
        evidence['C2 Asimetria A(1730)/A(max C=O)'] = f"{asymmetry:.3f} < 0.50 -> ETER (+2)"

    if thf_level != 'clean':
        evidence['C3 Ratio 1240/1110 (C-O-C)'] = f"{ratio_1240_1110:.3f} -- IGNORADO (THF)"
    elif ratio_1240_1110 > 1.5:
        score_ester += 3
        evidence['C3 Ratio 1240/1110 (C-O-C)'] = f"{ratio_1240_1110:.3f} > 1.5 -> ESTER (+3)"
    elif ratio_1240_1110 < 0.8:
        score_ether += 3
        evidence['C3 Ratio 1240/1110 (C-O-C)'] = f"{ratio_1240_1110:.3f} < 0.8 -> ETER (+3)"
    else:
        evidence['C3 Ratio 1240/1110 (C-O-C)'] = f"{ratio_1240_1110:.3f} -> zona ambigua"

    if thf_level != 'clean':
        evidence['C4 Ratio 1170/1110 (C-O-C)'] = f"{ratio_1170_1110:.3f} -- IGNORADO (THF)"
    elif ratio_1170_1110 > 1.3:
        score_ester += 2
        evidence['C4 Ratio 1170/1110 (C-O-C)'] = f"{ratio_1170_1110:.3f} > 1.3 -> ESTER alifatico (+2)"
    elif ratio_1170_1110 < 0.75:
        score_ether += 2
        evidence['C4 Ratio 1170/1110 (C-O-C)'] = f"{ratio_1170_1110:.3f} < 0.75 -> ETER (+2)"
    else:
        evidence['C4 Ratio 1170/1110 (C-O-C)'] = f"{ratio_1170_1110:.3f} -> zona ambigua"

    if ratio_1730_1700 > 0.35:
        score_ester += 1
        evidence['C5 Ratio A(1730)/A(1700)'] = f"{ratio_1730_1700:.3f} > 0.35 -> ESTER (+1)"
    else:
        evidence['C5 Ratio A(1730)/A(1700)'] = f"{ratio_1730_1700:.3f} -> no concluyente"

    total   = score_ester + score_ether
    ss_type = 'POLIESTER' if score_ester >= score_ether else 'POLIOTER'
    conf_v  = score_ester if ss_type == 'POLIESTER' else score_ether
    conf    = conf_v / total if total > 0 else 0.5

    return ss_type, conf, evidence, {
        'score_ester': score_ester, 'score_ether': score_ether,
        'asymmetry': round(asymmetry, 3),
        'ratio_1240_1110': round(ratio_1240_1110, 3),
        'ratio_1170_1110': round(ratio_1170_1110, 3),
    }


# =============================================================================
# 5. CLASIFICACIÓN DIISOCIANATO
# =============================================================================

def classify_hard_segment(wn, A):
    A815  = band_max(wn, A,  815, 15)
    A1596 = band_max(wn, A, 1596, 12)
    A1614 = band_max(wn, A, 1614, 10)
    A1510 = band_max(wn, A, 1510, 15)
    A2850 = band_max(wn, A, 2850, 15)
    A730  = band_max(wn, A,  730, 15)

    ratio_aa = A2850 / A815 if A815 > 0 else 99.0
    scores   = {'MDI': 0, 'TDI': 0, 'HDI': 0}
    evidence = {}

    if A815 > 0.12:
        scores['MDI'] += 2; scores['TDI'] += 1
        evidence['C-H aromatico 815 cm-1'] = f"A={A815:.3f} > 0.12 -> aromatico (MDI o TDI)"
    else:
        scores['HDI'] += 3
        evidence['C-H aromatico 815 cm-1'] = f"A={A815:.3f} <= 0.12 -> ausente -> HDI"

    if A1596 > 0.08:
        scores['MDI'] += 2
        evidence['C=C aromatico 1596 cm-1'] = f"A={A1596:.3f} -> MDI"
    else:
        evidence['C=C aromatico 1596 cm-1'] = f"A={A1596:.3f} -> debil"

    if A815 > 0.12:
        if A1596 > A1614:
            scores['MDI'] += 2
            evidence['1596 vs 1614 cm-1'] = f"{A1596:.3f} > {A1614:.3f} -> MDI (4,4-MDI)"
        else:
            scores['TDI'] += 2
            evidence['1596 vs 1614 cm-1'] = f"{A1596:.3f} < {A1614:.3f} -> TDI"
    else:
        evidence['1596 vs 1614 cm-1'] = "N/A (sin aromatico)"

    if A730 > 0.15 and A815 < 0.12:
        scores['HDI'] += 2
        evidence['CH2 rocking 730 cm-1'] = f"A={A730:.3f} -> HDI cadena larga"
    else:
        evidence['CH2 rocking 730 cm-1'] = f"A={A730:.3f}"

    if ratio_aa > 3:
        scores['HDI'] += 1
        evidence['Ratio CH2/arom (2850/815)'] = f"{ratio_aa:.2f} > 3 -> HDI"
    else:
        evidence['Ratio CH2/arom (2850/815)'] = f"{ratio_aa:.2f} <= 3 -> aromatico"

    hard_type  = max(scores, key=scores.get)
    total      = sum(scores.values())
    confidence = scores[hard_type] / total if total > 0 else 0.0

    return hard_type, confidence, evidence, scores, {
        'A815': A815, 'A1596': A1596, 'A1614': A1614,
        'A1510': A1510, 'A2850': A2850, 'A730': A730,
    }


# =============================================================================
# 6. DECONVOLUCIÓN CARBONILO
# =============================================================================

def deconvolve_carbonyl(wn, A, title=''):
    mask = (wn >= 1650) & (wn <= 1760)
    if mask.sum() < 10:
        return False, None, None, None, None, None, None

    wn_r = wn[mask]
    A_r  = A[mask]

    m1    = GaussianModel(prefix='g1_')
    m2    = GaussianModel(prefix='g2_')
    m3    = GaussianModel(prefix='g3_')
    cst   = ConstantModel()
    model = m1 + m2 + m3 + cst

    p = model.make_params()
    p['g1_center'].set(value=1698, min=1688, max=1710)
    p['g1_sigma'].set(value=10,    min=3,    max=25)
    p['g1_amplitude'].set(value=0.3, min=0)
    p['g2_center'].set(value=1715, min=1708, max=1724)
    p['g2_sigma'].set(value=10,    min=3,    max=25)
    p['g2_amplitude'].set(value=0.3, min=0)
    p['g3_center'].set(value=1730, min=1722, max=1745)
    p['g3_sigma'].set(value=10,    min=3,    max=25)
    p['g3_amplitude'].set(value=0.3, min=0)
    p['c'].set(value=0, min=-0.05, max=0.1)

    try:
        result = model.fit(A_r, p, x=wn_r, method='least_squares')
    except Exception as e:
        print(f"  [WARN] Deconvolucion fallida: {e}")
        return False, None, None, None, None, None, None

    pv = result.params
    def gauss_area(pref):
        return float(pv[f'{pref}amplitude'].value *
                     pv[f'{pref}sigma'].value * np.sqrt(2 * np.pi))

    Ag1 = gauss_area('g1_')
    Ag2 = gauss_area('g2_')
    Ag3 = gauss_area('g3_')

    denom = Ag1 + Ag2 + K_MOLAR * Ag3
    if denom <= 0:
        return False, None, None, None, None, None, None

    DPS = (Ag1 + Ag2) / denom * 100
    DOR = Ag1         / denom * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wn_r, A_r, 'k-', lw=1.5, label='Espectro')
    ax.plot(wn_r, result.best_fit, 'r--', lw=1.5, label='Ajuste total')
    colors    = ['#2196F3', '#FF9800', '#4CAF50']
    labels_dc = ['C=O H-bond ord. (1698)', 'C=O H-bond dis. (1715)', 'C=O libre (1730)']
    for pref, c, lb in zip(['g1_', 'g2_', 'g3_'], colors, labels_dc):
        comp = result.eval_components(x=wn_r)[pref]
        ax.fill_between(wn_r, 0, comp, alpha=0.35, color=c, label=lb)
    ax.set_xlim(1760, 1650)
    ax.set_xlabel('Numero de onda (cm-1)')
    ax.set_ylabel('Absorbancia norm.')
    ax.set_title(f'Deconvolucion C=O -- {title}\nDPS = {DPS:.1f}%   DOR = {DOR:.1f}%')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return True, DPS, DOR, Ag1, Ag2, Ag3, fig


# =============================================================================
# 7. ESPECTRO ANOTADO
# =============================================================================

_BANDS_COMMON = [
    (3320, 'N-H str.',   'top'),
    (2920, 'CH2 asim.',  'top'),
    (2850, 'CH2 sim.',   'top'),
    (1700, 'C=O HB',     'top'),
    (1530, 'Amida II',   'top'),
    (1220, 'C-N ur.',    'bottom'),
]
_BANDS_ESTER = [
    (1730, 'C=O ester', 'top'),
    (1170, 'C-O-C est', 'bottom'),
]
_BANDS_ETHER = [
    (1110, 'C-O-C eter (PTMG)', 'bottom'),
    (1080, 'C-O-C eter',        'bottom'),
]
_BANDS_MDI = [
    (815,  'MDI 815',  'bottom'),
    (1596, 'MDI 1596', 'bottom'),
]
_BANDS_TDI = [
    (815,  'TDI 815',  'bottom'),
    (1614, 'TDI 1614', 'bottom'),
]
_BANDS_HDI = [
    (730,  'HDI CH2 rock.', 'bottom'),
    (1465, 'CH2 bend.',     'bottom'),
]


def plot_annotated_spectrum(wn, A, soft_type, hard_type, DPS, DOR, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wn, A, color='#1a1a2e', lw=1.2)

    bands = list(_BANDS_COMMON)
    bands += _BANDS_ESTER  if 'ESTER' in soft_type else _BANDS_ETHER
    bands += (_BANDS_MDI if hard_type == 'MDI' else
              _BANDS_TDI if hard_type == 'TDI' else _BANDS_HDI)

    wn_min, wn_max = wn.min(), wn.max()
    Amax = A.max()
    offset_top    = Amax * 0.08
    offset_bottom = Amax * 0.05

    for center, label, pos in bands:
        if not (wn_min <= center <= wn_max):
            continue
        mask_b = (wn >= center - 15) & (wn <= center + 15)
        if not mask_b.any():
            continue
        y_peak = A[mask_b].max()
        ax.axvline(center, color='#888888', lw=0.5, ls='--', alpha=0.6)

        if pos == 'top':
            y_text = y_peak + offset_top
            va = 'bottom'
        else:
            y_text = -offset_bottom
            va = 'top'

        ax.annotate(f'{center}',
                    xy=(center, y_peak), xytext=(center, y_text),
                    ha='center', va=va, fontsize=7,
                    arrowprops=dict(arrowstyle='-', color='#888888', lw=0.6))
        ax.text(center,
                y_text + (offset_top * 0.5 if pos == 'top' else -offset_bottom * 0.5),
                label, ha='center', va=va, fontsize=6.5, color='#444444',
                rotation=90 if len(label) > 8 else 0)

    dps_str = f'DPS = {DPS:.1f}%\nDOR = {DOR:.1f}%' if DPS is not None else 'Deconvolucion no disponible'
    textstr = f'Soft segment: {soft_type}\nDiisocianato: {hard_type}\n{dps_str}'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    ax.set_xlim(wn_max + 50, wn_min - 50)
    ax.set_ylim(-0.15, Amax * 1.35)
    ax.set_xlabel('Numero de onda (cm-1)', fontsize=11)
    ax.set_ylabel('Absorbancia (norm.)', fontsize=11)
    ax.set_title(f'ATR-FTIR -- {title}', fontsize=12, fontweight='bold')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which='both', alpha=0.2)
    fig.tight_layout()
    return fig


# =============================================================================
# 8. PANEL DIAGNÓSTICO
# =============================================================================

def plot_diagnostic_panel(wn, A, dc_result, title):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'Panel diagnostico ATR-FTIR -- {title}', fontsize=13, fontweight='bold')

    def sub_plot(ax, lo, hi, title_ax, mark_bands):
        mask = (wn >= lo) & (wn <= hi)
        if not mask.any():
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center', transform=ax.transAxes)
            return
        ax.plot(wn[mask], A[mask], 'k-', lw=1.2)
        ax.set_xlim(hi + 5, lo - 5)
        ymax = A[mask].max()
        for c, lb, clr in mark_bands:
            if lo <= c <= hi:
                ax.axvline(c, color=clr, lw=0.8, ls='--', alpha=0.8)
                ax.text(c, ymax * 0.95, lb, ha='center', va='top',
                        fontsize=7, color=clr, rotation=90)
        ax.set_xlabel('Numero de onda (cm-1)', fontsize=9)
        ax.set_ylabel('Absorbancia norm.', fontsize=9)
        ax.set_title(title_ax, fontsize=10)
        ax.grid(True, alpha=0.25)

    sub_plot(axes[0, 0], 3050, 3600, 'Region N-H', [
        (3320, 'N-H str.',  '#1565C0'),
        (3030, '=C-H arom.','#E65100'),
    ])

    mask_co = (wn >= 1640) & (wn <= 1800)
    axes[0, 1].plot(wn[mask_co], A[mask_co], 'k-', lw=1.5, label='Espectro')
    if dc_result[0]:
        for c, lb, clr in [(1698, '1698 (ord)', '#2196F3'),
                            (1715, '1715 (dis)', '#FF9800'),
                            (1730, '1730 (libre)', '#4CAF50')]:
            axes[0, 1].axvline(c, color=clr, lw=1, ls=':', label=lb)
    axes[0, 1].set_xlim(1800, 1640)
    axes[0, 1].set_xlabel('Numero de onda (cm-1)', fontsize=9)
    axes[0, 1].set_ylabel('Absorbancia norm.', fontsize=9)
    axes[0, 1].set_title('Region C=O (1640-1800 cm-1)', fontsize=10)
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.25)

    sub_plot(axes[1, 0], 1050, 1300, 'Fingerprint segmento blando', [
        (1240, '1240 (ester)', '#C62828'),
        (1170, '1170 (ester)', '#E53935'),
        (1110, '1110 (eter)',  '#1565C0'),
        (1080, '1080 (eter)',  '#1976D2'),
    ])

    sub_plot(axes[1, 1], 650, 900, 'Fingerprint diisocianato', [
        (815, '815 (MDI/TDI)', '#6A1B9A'),
        (730, '730 (HDI)',     '#2E7D32'),
    ])

    fig.tight_layout()
    return fig


# =============================================================================
# 9. FUNCIÓN PRINCIPAL
# =============================================================================

def classify_tpu(filepath, verbose=True):
    """
    Pipeline completo. Muestra resultados en terminal y graficos en pantalla.
    No guarda nada en disco.
    """
    sample = os.path.basename(filepath)
    title  = os.path.splitext(sample)[0]

    wn, A, A_raw, norm_val = load_and_preprocess(filepath)

    thf_data                          = check_thf_contamination(wn, A)
    soft_type, soft_conf, soft_ev, sd = classify_soft_segment(wn, A, thf_level=thf_data['level'])
    hard_type, hard_conf, hard_ev, hard_scores, hd = classify_hard_segment(wn, A)
    dc_result                         = deconvolve_carbonyl(wn, A, title=title)
    success_dc, DPS, DOR, Ag1, Ag2, Ag3, fig_dc = dc_result

    # ── Terminal
    if verbose:
        sep = "=" * 68
        print(f"\n{sep}")
        print(f"  CLASIFICACION ATR-FTIR: {sample}")
        print(sep)
        print(f"  Factor normalizacion (max 1500-1760 cm-1): {norm_val:.4f}")
        print(f"\n  THF RESIDUAL : {thf_data['level'].upper()}")
        print(f"    {thf_data['message']}")
        print(f"\n  SEGMENTO BLANDO : {soft_type}  (confianza {soft_conf*100:.0f}%,"
              f"  ester={sd['score_ester']}, eter={sd['score_ether']})")
        for k, v in soft_ev.items():
            print(f"    · {k}: {v}")
        print(f"\n  DIISOCIANATO : {hard_type}  (confianza {hard_conf*100:.0f}%)")
        print(f"    Puntuaciones -- MDI={hard_scores['MDI']}"
              f"  TDI={hard_scores['TDI']}  HDI={hard_scores['HDI']}")
        for k, v in hard_ev.items():
            print(f"    · {k}: {v}")
        if success_dc:
            print(f"\n  DECONVOLUCION C=O:")
            print(f"    A(1698) H-bond ord. = {Ag1:.4f}")
            print(f"    A(1715) H-bond dis. = {Ag2:.4f}")
            print(f"    A(1730) C=O libre   = {Ag3:.4f}   (k = {K_MOLAR})")
            print(f"    DPS = {DPS:.1f}%")
            print(f"    DOR = {DOR:.1f}%")
        else:
            print("\n  DECONVOLUCION C=O: no convergio")
        print(f"\n  {'─'*50}")
        print(f"  RESULTADO: {soft_type} / {hard_type}")
        print(f"{sep}\n")
        sys.stdout.flush()

    # ── Graficos (solo pantalla)
    plot_annotated_spectrum(wn, A, soft_type, hard_type, DPS, DOR, title)
    plot_diagnostic_panel(wn, A, dc_result, title)
    if success_dc and fig_dc is not None:
        pass  # fig_dc ya fue creada dentro de deconvolve_carbonyl

    plt.show(block=True)

    return {
        'sample':          sample,
        'soft':            soft_type,
        'hard':            hard_type,
        'soft_confidence': round(soft_conf * 100, 1),
        'hard_confidence': round(hard_conf * 100, 1),
        'hard_scores':     hard_scores,
        'DPS':             round(DPS, 2) if DPS is not None else None,
        'DOR':             round(DOR, 2) if DOR is not None else None,
        'thf_level':       thf_data['level'],
        'norm_factor':     round(norm_val, 4),
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = '/home/nacho/Escritorio/THESIS_USB/ATR-FTIR/tagup.CSV'

    if not os.path.exists(target):
        print(f"Archivo no encontrado: {target}")
        print("Uso: python ftir_tpu_classifier.py archivo.txt")
        sys.exit(1)

    classify_tpu(target, verbose=True)