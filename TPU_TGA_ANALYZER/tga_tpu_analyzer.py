#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 08:34:29 2026

@author: ozymandias
"""

"""
tga_tpu_analyzer.py
====================
Lee archivos TGA nativos de TA Instruments (TGA Q50, formato binario)
y realiza deconvolución gaussiana de la curva DTG para estimar el porcentaje
de segmento blando (soft segment, SS) y segmento duro (hard segment, HS) en TPUs.

Autor: Generado con Claude / Ozy – IMDEA Materials / UC3M
Fecha: 2026

Uso:
    python tga_tpu_analyzer.py archivo_tga [polyether|polyester]

O desde Python:
    from tga_tpu_analyzer import analyze_tpu_tga
    results = analyze_tpu_tga('muestra.tga', tpu_type='polyester', n_peaks=3,
                              peak_centers=[330, 400, 457])
"""

import struct
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid


# =============================================================================
# LECTURA DE ARCHIVOS TA INSTRUMENTS
# =============================================================================

def read_ta_file(filepath):
    """
    Lee un archivo TGA nativo de TA Instruments (TGA Q50, formato binario).

    Parameters
    ----------
    filepath : str
        Ruta al archivo TGA.

    Returns
    -------
    dict con:
        'temperature' : np.ndarray  – Temperatura (°C)
        'weight'      : np.ndarray  – Peso (mg)
        'time'        : np.ndarray  – Tiempo (min)
        'metadata'    : dict        – Metadatos de la cabecera
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    # Buscar marcador de fin de cabecera (form feed 0x0C en UTF-16 LE)
    form_feed_pos = data.find(b'\x0c\x00')
    if form_feed_pos == -1:
        raise ValueError(f"No se encontró el marcador de datos en '{filepath}'.")

    # Extraer metadatos
    header_text = data[:form_feed_pos].decode('utf-16-le', errors='ignore')
    metadata = _parse_ta_header(header_text)

    # Los datos empiezan 7 bytes después del form feed
    data_start = form_feed_pos + 7
    record_size = 20  # 5 floats de 4 bytes cada uno

    data_length = len(data) - data_start
    num_records = data_length // record_size

    temperature, weight, time_arr = [], [], []

    for i in range(num_records):
        offset = data_start + i * record_size
        try:
            vals = struct.unpack('<5f', data[offset:offset + record_size])
            # vals: [temperatura, peso_mg, ?, ?, tiempo]
            if 0 <= vals[0] <= 1200 and vals[1] >= 0 and vals[4] >= 0:
                temperature.append(vals[0])
                weight.append(vals[1])
                time_arr.append(vals[4])
        except struct.error:
            break

    return {
        'temperature': np.array(temperature),
        'weight': np.array(weight),
        'time': np.array(time_arr),
        'metadata': metadata
    }


def _parse_ta_header(header_text):
    """Extrae metadatos de la cabecera de texto UTF-16 LE del archivo TA."""
    metadata = {}
    for line in header_text.split('\r\n'):
        line = line.strip()
        if line:
            parts = line.split(None, 1)
            if len(parts) == 2:
                metadata[parts[0]] = parts[1]
    return metadata


# =============================================================================
# PROCESAMIENTO DE LA CURVA DTG
# =============================================================================

def calculate_dtg(temperature, weight, smooth_window=15, smooth_order=3):
    """
    Calcula la curva DTG (−dW/dT) y la devuelve suavizada con Savitzky-Golay.

    Returns
    -------
    temp_dtg : np.ndarray  – Temperatura (°C), sin el primer punto
    dtg      : np.ndarray  – DTG (mg/°C), valores positivos
    """
    dw = np.diff(weight)
    dt = np.diff(temperature)

    # Evitar división por cero
    dt[dt == 0] = 1e-6

    dtg_raw = -dw / dt          # negativo → pérdida de masa es positiva
    temp_dtg = temperature[1:]

    # Suavizado
    if smooth_window < len(dtg_raw):
        dtg = savgol_filter(dtg_raw, smooth_window, smooth_order)
    else:
        dtg = dtg_raw.copy()

    dtg = np.maximum(dtg, 0)    # eliminar artefactos negativos
    return temp_dtg, dtg


# =============================================================================
# FUNCIONES DE AJUSTE GAUSSIANO
# =============================================================================

def gaussian(x, amplitude, center, width):
    """Gaussiana: A · exp(−(x − μ)² / (2σ²))"""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))


def multi_gaussian(x, *params):
    """Suma de N gaussianas. params = [A1,μ1,σ1, A2,μ2,σ2, ...]"""
    n_peaks = len(params) // 3
    result = np.zeros_like(x, dtype=float)
    for i in range(n_peaks):
        result += gaussian(x, params[3 * i], params[3 * i + 1], params[3 * i + 2])
    return result


# =============================================================================
# DETECCIÓN AUTOMÁTICA DE PICOS
# =============================================================================

def _find_dtg_peaks(temp, dtg, min_distance_deg=30):
    """
    Detección automática de picos en la curva DTG.

    Returns
    -------
    peaks : np.ndarray – índices de los picos detectados
    """
    max_val = np.max(dtg)
    avg_spacing = np.mean(np.diff(temp))
    min_dist_pts = max(1, int(min_distance_deg / avg_spacing))

    peaks, _ = find_peaks(
        dtg,
        height=max_val * 0.05,
        prominence=max_val * 0.05,
        distance=min_dist_pts
    )
    return peaks


# =============================================================================
# DECONVOLUCIÓN
# =============================================================================

def deconvolute_dtg(temp, dtg, n_peaks=None, peak_centers=None,
                    temp_range=(250, 500)):
    """
    Deconvolución gaussiana de la curva DTG.

    Parameters
    ----------
    temp        : array – Temperatura (°C)
    dtg         : array – Curva DTG
    n_peaks     : int, optional – Número de picos (None = automático)
    peak_centers: list, optional – Centros iniciales en °C (recomendado cuando
                  se conocen las temperaturas aproximadas de los picos)
    temp_range  : tuple – Rango de temperatura para el ajuste

    Returns
    -------
    dict con:
        'peaks'      : lista de dicts por pico (amplitude, center, width,
                       area, area_percent, curve)
        'temp_fit'   : array de temperatura en el rango
        'dtg_fit'    : DTG original en el rango
        'dtg_smooth' : DTG suavizada usada para el ajuste
        'dtg_fitted' : curva ajustada total
        'n_peaks'    : int
        'total_area' : float
    """
    mask = (temp >= temp_range[0]) & (temp <= temp_range[1])
    temp_fit = temp[mask]
    dtg_raw  = dtg[mask]
    dtg_fit  = np.maximum(dtg_raw, 0)

    # Suavizado adicional para el ajuste
    win = min(21, len(dtg_fit) - 1 if len(dtg_fit) % 2 == 0 else len(dtg_fit))
    win = win if win % 2 == 1 else win - 1
    dtg_smooth = savgol_filter(dtg_fit, win, 3) if win >= 5 else dtg_fit.copy()
    dtg_smooth = np.maximum(dtg_smooth, 0)

    # Determinar número de picos
    if peak_centers is not None:
        n_peaks = len(peak_centers)
    elif n_peaks is None:
        auto_peaks = _find_dtg_peaks(temp_fit, dtg_smooth)
        n_peaks = max(len(auto_peaks), 2)

    # Estimaciones iniciales
    if peak_centers is not None:
        p0 = []
        for c in peak_centers:
            idx = np.argmin(np.abs(temp_fit - c))
            amp = dtg_smooth[idx]
            p0.extend([amp, c, 20.0])
    else:
        auto_peaks = _find_dtg_peaks(temp_fit, dtg_smooth)
        p0 = []
        if len(auto_peaks) >= n_peaks:
            for i in range(n_peaks):
                idx = auto_peaks[i]
                p0.extend([dtg_smooth[idx], temp_fit[idx], 20.0])
        else:
            # Distribuir uniformemente si no hay suficientes picos detectados
            centers = np.linspace(temp_range[0] + 30, temp_range[1] - 30, n_peaks)
            max_amp = np.max(dtg_smooth)
            for c in centers:
                p0.extend([max_amp / n_peaks, c, 25.0])

    # Bounds: amplitud ≥ 0, centro dentro del rango, ancho entre 5 y 100 °C
    lower = [0, temp_range[0], 5.0] * n_peaks
    upper = [np.max(dtg_smooth) * 1.5, temp_range[1], 100.0] * n_peaks

    try:
        popt, _ = curve_fit(
            multi_gaussian, temp_fit, dtg_smooth,
            p0=p0, bounds=(lower, upper),
            maxfev=10000
        )
    except RuntimeError as e:
        print(f"[AVISO] El ajuste no convergió completamente: {e}")
        popt = p0

    # Calcular áreas
    results = {
        'n_peaks': n_peaks,
        'peaks': [],
        'temp_fit': temp_fit,
        'dtg_fit': dtg_fit,
        'dtg_smooth': dtg_smooth,
        'dtg_fitted': multi_gaussian(temp_fit, *popt)
    }

    total_area = 0.0
    for i in range(n_peaks):
        amp    = popt[3 * i]
        center = popt[3 * i + 1]
        width  = popt[3 * i + 2]
        curve  = gaussian(temp_fit, amp, center, width)
        area   = trapezoid(curve, temp_fit)
        total_area += area
        results['peaks'].append({
            'amplitude': amp,
            'center': center,
            'width': width,
            'area': area,
            'curve': curve
        })

    for peak in results['peaks']:
        peak['area_percent'] = (peak['area'] / total_area * 100
                                if total_area > 0 else 0.0)

    results['total_area'] = total_area
    return results


# =============================================================================
# CUANTIFICACIÓN SS / HS
# =============================================================================

def calculate_segment_content(deconv_results, tpu_type='polyether',
                               cutoff_temp=None):
    """
    Estima el contenido de segmento blando (SS) y duro (HS).

    Criterio:
    - Picos a T < cutoff_temp  → soft segment
    - Picos a T ≥ cutoff_temp → hard segment

    Si cutoff_temp es None, se usa el mínimo entre el primer y segundo pico
    como punto de corte natural.

    Returns
    -------
    dict con 'soft_segment_percent', 'hard_segment_percent', 'cutoff_temp'
    """
    peaks = deconv_results['peaks']
    centers = sorted([p['center'] for p in peaks])

    if cutoff_temp is None:
        # Punto de corte = promedio entre el primer y segundo pico más altos
        if len(centers) >= 2:
            cutoff_temp = (centers[0] + centers[1]) / 2
        else:
            cutoff_temp = centers[0] + 30

    ss_area = sum(p['area'] for p in peaks if p['center'] < cutoff_temp)
    hs_area = sum(p['area'] for p in peaks if p['center'] >= cutoff_temp)
    total   = ss_area + hs_area

    return {
        'soft_segment_percent': ss_area / total * 100 if total > 0 else 0.0,
        'hard_segment_percent': hs_area / total * 100 if total > 0 else 0.0,
        'cutoff_temp': cutoff_temp
    }


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']


def plot_tga_analysis(tga_data, dtg_data, deconv_results, segment_results,
                      sample_name='Sample', save_path=None):
    """
    Genera una figura con 3 subplots:
      1) Curva TGA (peso vs temperatura)
      2) DTG con deconvolución gaussiana
      3) Tabla de resultados
    """
    temp_dtg, dtg = dtg_data

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'TGA Analysis – {sample_name}', fontsize=14, fontweight='bold')

    # --- Panel 1: TGA ---
    ax1 = axes[0]
    ax1.plot(tga_data['temperature'], tga_data['weight'], 'k-', linewidth=1.5)
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Weight (mg)')
    ax1.set_title('TGA curve')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: DTG + deconvolución ---
    ax2 = axes[1]
    temp_fit   = deconv_results['temp_fit']
    dtg_smooth = deconv_results['dtg_smooth']
    dtg_fitted = deconv_results['dtg_fitted']
    peaks      = deconv_results['peaks']

    ax2.plot(temp_dtg, dtg, color='lightgray', linewidth=1, label='DTG (raw)')
    ax2.plot(temp_fit, dtg_smooth, 'k-', linewidth=1.5, label='DTG (smoothed)')
    ax2.plot(temp_fit, dtg_fitted, 'r--', linewidth=1.5, label='Fitted sum')

    for i, peak in enumerate(peaks):
        color = COLORS[i % len(COLORS)]
        label = (f'Peak {i+1}: {peak["center"]:.0f} °C '
                 f'({peak["area_percent"]:.1f}%)')
        ax2.fill_between(temp_fit, peak['curve'], alpha=0.3, color=color)
        ax2.plot(temp_fit, peak['curve'], color=color, linewidth=1.2,
                 label=label)

    ax2.axvline(segment_results['cutoff_temp'], color='gray', linestyle=':',
                alpha=0.7, label=f'Cut-off: {segment_results["cutoff_temp"]:.0f} °C')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('−dW/dT  (mg/°C)')
    ax2.set_title('DTG deconvolution')
    ax2.legend(fontsize=7.5)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Tabla de resultados ---
    ax3 = axes[2]
    ax3.axis('off')

    table_data = [
        ['Parameter', 'Value'],
        ['Sample', sample_name],
        ['Soft Segment (SS)', f'{segment_results["soft_segment_percent"]:.1f}%'],
        ['Hard Segment (HS)', f'{segment_results["hard_segment_percent"]:.1f}%'],
        ['Cut-off temperature', f'{segment_results["cutoff_temp"]:.0f} °C'],
        ['', ''],
    ]
    for i, peak in enumerate(peaks):
        tag = 'SS' if peak['center'] < segment_results['cutoff_temp'] else 'HS'
        table_data.append([
            f'Peak {i+1} center ({tag})',
            f'{peak["center"]:.1f} °C'
        ])
        table_data.append([
            f'Peak {i+1} area',
            f'{peak["area_percent"]:.1f}%'
        ])

    table = ax3.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    ax3.set_title('Results summary')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Figura guardada en: {save_path}")

    return fig


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def analyze_tpu_tga(filepath, tpu_type='polyether', n_peaks=None,
                    peak_centers=None, temp_range=(250, 500),
                    cutoff_temp=None, method='deconvolution',
                    save_figure=False, show_figure=True):
    """
    Pipeline completo: lectura → DTG → deconvolución → cuantificación → figura.

    Parameters
    ----------
    filepath     : str   – Ruta al archivo TGA (formato binario TA Instruments)
    tpu_type     : str   – 'polyether' o 'polyester'
    n_peaks      : int   – Número de picos (None = automático)
    peak_centers : list  – Centros iniciales en °C, p.ej. [330, 400, 457]
                           (recomendado cuando se conocen las temperaturas)
    temp_range   : tuple – Rango de temperatura para el ajuste (°C)
    cutoff_temp  : float – Temperatura de corte SS/HS (None = automático)
    method       : str   – 'deconvolution' (único método actual)
    save_figure  : bool  – Guardar figura en el mismo directorio del archivo
    show_figure  : bool  – Mostrar figura interactiva

    Returns
    -------
    dict con: 'tga_data', 'dtg', 'deconvolution', 'segments',
              'sample_name', 'figure'
    """
    sample_name = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n{'='*55}")
    print(f"  TGA analysis: {sample_name}")
    print(f"  TPU type    : {tpu_type}")
    print(f"{'='*55}")

    # 1. Leer archivo
    tga_data = read_ta_file(filepath)
    print(f"  Records loaded: {len(tga_data['temperature'])}")

    # 2. Calcular DTG
    temp_dtg, dtg = calculate_dtg(tga_data['temperature'], tga_data['weight'])

    # 3. Deconvolución
    deconv_results = deconvolute_dtg(
        temp_dtg, dtg,
        n_peaks=n_peaks,
        peak_centers=peak_centers,
        temp_range=temp_range
    )

    # 4. Cuantificación
    segment_results = calculate_segment_content(
        deconv_results, tpu_type=tpu_type, cutoff_temp=cutoff_temp
    )

    # 5. Resultados en consola
    print(f"\n  Peaks found: {deconv_results['n_peaks']}")
    for i, p in enumerate(deconv_results['peaks']):
        tag = 'SS' if p['center'] < segment_results['cutoff_temp'] else 'HS'
        print(f"    Peak {i+1} [{tag}]: {p['center']:.1f} °C  –  "
              f"{p['area_percent']:.1f}%")
    print(f"\n  Soft Segment : {segment_results['soft_segment_percent']:.1f}%")
    print(f"  Hard Segment : {segment_results['hard_segment_percent']:.1f}%")
    print(f"{'='*55}\n")

    # 6. Figura
    save_path = None
    if save_figure:
        out_dir = os.path.dirname(os.path.abspath(filepath))
        save_path = os.path.join(out_dir, f'{sample_name}_tga_analysis.png')

    fig = plot_tga_analysis(
        tga_data, (temp_dtg, dtg), deconv_results, segment_results,
        sample_name=sample_name, save_path=save_path
    )

    if show_figure:
        plt.show()

    return {
        'tga_data': tga_data,
        'dtg': (temp_dtg, dtg),
        'deconvolution': deconv_results,
        'segments': segment_results,
        'sample_name': sample_name,
        'figure': fig
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    fp       = sys.argv[1]
    tpu_type = sys.argv[2] if len(sys.argv) > 2 else 'polyether'

    # Ejemplo de uso con centros manuales (descomenta si los conoces):
    # results = analyze_tpu_tga(fp, tpu_type=tpu_type,
    #                           peak_centers=[330, 400, 457])

    results = analyze_tpu_tga(fp, tpu_type=tpu_type, save_figure=True)tga_tpu_analyzer.py