#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 2026
@author: nacho

Script to read TA Instruments TGA Q50 binary files.
Extracts temperature, weight, and time arrays and computes DTG,
onset temperature, T5%, T10%, T50%, Tmax, and residue.

File format:
  - UTF-16 LE encoded header
  - Binary data after form feed (0x0C 0x00)
  - 5 float32 signals per record: Temperature, Weight(mg), Weight(%), PurgeFlow, Time
  - Data starts 7 bytes after the form feed marker
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path


# =============================================================================
# FILE READER
# =============================================================================

def parse_ta_header(header_text):
    """Extracts metadata from the TA Instruments file header."""
    metadata = {}
    for line in header_text.split('\r\n'):
        line = line.strip()
        if ' ' in line:
            key   = line.split()[0]
            value = ' '.join(line.split()[1:])
            metadata[key] = value
    return metadata


def read_ta_tga_file(filepath):
    """
    Reads a TA Instruments TGA Q50 binary file.

    Parameters
    ----------
    filepath : str or Path
        Path to the TGA file (no extension required)

    Returns
    -------
    dict with keys:
        'temperature' : numpy array (°C)
        'weight_mg'   : numpy array (mg)
        'weight_pct'  : numpy array (%)
        'time'        : numpy array (min)
        'metadata'    : dict with header information
    """
    with open(filepath, 'rb') as f:
        data = f.read()

    # Find form feed marker
    form_feed_pos = data.find(b'\x0c\x00')
    if form_feed_pos == -1:
        raise ValueError("Data section marker not found. Is this a TGA Q50 file?")

    # Parse header
    header_text = data[:form_feed_pos].decode('utf-16-le', errors='ignore')
    metadata    = parse_ta_header(header_text)

    # Data starts 7 bytes after form feed
    # Record: 5 float32 = 20 bytes
    # Signal mapping: [Temperature, Weight(mg), Weight(%), PurgeFlow, Time]
    data_start  = form_feed_pos + 7
    record_size = 20   # 5 × 4 bytes

    data_length = len(data) - data_start
    num_records = data_length // record_size

    temperature = []
    weight_mg   = []
    weight_pct  = []
    time_arr    = []

    for i in range(num_records):
        offset = data_start + i * record_size
        try:
            vals = struct.unpack('<5f', data[offset:offset + record_size])
            # Filter invalid records
            if 0 <= vals[0] <= 1000 and vals[1] >= 0 and vals[4] >= 0:
                temperature.append(vals[0])
                weight_mg.append(vals[1])
                weight_pct.append(vals[2])
                time_arr.append(vals[4])
        except Exception:
            break

    return {
        'temperature': np.array(temperature),
        'weight_mg':   np.array(weight_mg),
        'weight_pct':  np.array(weight_pct),
        'time':        np.array(time_arr),
        'metadata':    metadata,
    }


# =============================================================================
# DTG AND PARAMETERS
# =============================================================================

def calculate_dtg(temperature, weight_pct, smooth_window=15, smooth_order=3):
    """
    Computes the DTG curve (derivative of weight with respect to temperature).

    Returns
    -------
    dtg : numpy array — %/°C
    """
    weight_smooth = savgol_filter(weight_pct, smooth_window, smooth_order)
    dtg = np.gradient(weight_smooth, temperature)
    return savgol_filter(dtg, smooth_window, smooth_order)


def calculate_tga_parameters(temperature, weight_pct, smooth_window=15, smooth_order=3):
    """
    Computes key TGA parameters.

    Returns
    -------
    dict with: T5, T10, T50, T_onset, T_max, DTG_max, residue, dtg
    """
    weight_smooth = savgol_filter(weight_pct, smooth_window, smooth_order)
    dtg           = calculate_dtg(temperature, weight_pct, smooth_window, smooth_order)

    w0      = weight_smooth[0]
    w_final = weight_smooth[-1]

    def find_T_at_weight(target_pct):
        idx = np.argmin(np.abs(weight_smooth - target_pct))
        return float(temperature[idx])

    T5  = find_T_at_weight(w0 * 0.95)
    T10 = find_T_at_weight(w0 * 0.90)
    T50 = find_T_at_weight(w0 * 0.50)

    # Onset: tangent intersection method
    idx_max_dtg = np.argmin(dtg)   # most negative = fastest loss
    T_max       = float(temperature[idx_max_dtg])
    DTG_max     = float(dtg[idx_max_dtg])

    # Tangent at max DTG
    slope     = DTG_max
    intercept = weight_smooth[idx_max_dtg] - slope * T_max
    T_onset   = (w0 - intercept) / slope if slope != 0 else T_max

    residue = float(w_final)

    return {
        'T5':      T5,
        'T10':     T10,
        'T50':     T50,
        'T_onset': T_onset,
        'T_max':   T_max,
        'DTG_max': DTG_max,
        'residue': residue,
        'dtg':     dtg,
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == '__main__':

    # Change this path to your file
    filepath = "/home/nacho/Escritorio/THESIS_USB/TGA/your_file"

    tga  = read_ta_tga_file(filepath)
    params = calculate_tga_parameters(tga['temperature'], tga['weight_pct'])

    T   = tga['temperature']
    W   = tga['weight_pct']
    dtg = params['dtg']

    # Print summary
    print("=" * 55)
    print(f"  TGA ANALYSIS — {tga['metadata'].get('SampleName', filepath)}")
    print("=" * 55)
    print(f"  Points       : {len(T)}")
    print(f"  Temp range   : {T.min():.1f} - {T.max():.1f} °C")
    print(f"  T5%          : {params['T5']:.1f} °C")
    print(f"  T10%         : {params['T10']:.1f} °C")
    print(f"  T50%         : {params['T50']:.1f} °C")
    print(f"  T onset      : {params['T_onset']:.1f} °C")
    print(f"  T max (DTG)  : {params['T_max']:.1f} °C")
    print(f"  Residue      : {params['residue']:.2f} %")
    print("=" * 55)

    # Plot TGA + DTG
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(T, W,    color='#1f77b4', lw=1.8, label='Weight (%)')
    ax2.plot(T, dtg,  color='#d62728', lw=1.2, ls='--', label='DTG (%/°C)')

    ax1.axvline(params['T_onset'], color='gray', ls=':', lw=1,
                label=f"T onset = {params['T_onset']:.0f} °C")
    ax1.axvline(params['T_max'],   color='orange', ls=':', lw=1,
                label=f"T max = {params['T_max']:.0f} °C")

    ax1.set_xlabel('Temperature (°C)', fontsize=13)
    ax1.set_ylabel('Weight (%)',        fontsize=13, color='#1f77b4')
    ax2.set_ylabel('DTG (%/°C)',        fontsize=13, color='#d62728')
    ax1.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax1.set_title(tga['metadata'].get('SampleName', 'TGA Q50'),
                  fontsize=14, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='best')

    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    plt.show()
