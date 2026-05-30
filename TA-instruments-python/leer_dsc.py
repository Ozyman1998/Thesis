
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 2026
@author: nacho

Script to read TA Instruments DSC Q200 .001 files.
Extracts heat flow and temperature as numpy arrays.

File format:
  - UTF-16 LE encoded header with metadata
  - Binary data section after a form feed character (0x0C 0x00)
  - 4 signals per record as float32 little-endian
  - Data located in the last quarter of the binary section
"""

import struct
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def read_ta_dsc_file(filepath):
    """
    Reads a TA Instruments DSC Q200 .001 file.

    Parameters
    ----------
    filepath : str or Path
        Path to the .001 file

    Returns
    -------
    dict with keys:
        'temperature'  : numpy array (°C)
        'heat_flow'    : numpy array (mW)
        'time'         : numpy array (min)
        'metadata'     : dict with header information
    """
    with open(filepath, 'rb') as f:
        content = f.read()

    # Find form feed marker (end of UTF-16 LE header)
    ff_pos = None
    for i in range(len(content) - 1):
        if content[i] == 0x0C and content[i + 1] == 0x00:
            ff_pos = i
            break

    if ff_pos is None:
        raise ValueError("Data section marker not found. Is this a DSC Q200 file?")

    # Parse header metadata
    header_bytes = content[:ff_pos]
    try:
        header_text = header_bytes.decode('utf-16-le', errors='replace')
    except Exception:
        header_text = ""

    metadata = {}
    for line in header_text.split('\r\n'):
        line = line.strip()
        if ' ' in line:
            key   = line.split()[0]
            value = ' '.join(line.split()[1:])
            metadata[key] = value

    # Binary data starts after form feed (2 bytes in UTF-16)
    data_start  = ff_pos + 2
    binary_data = content[data_start:]
    total_bytes = len(binary_data)

    # Data is in the last quarter of the binary section
    # 4 signals per record (Time, Temperature, Heat Flow, Purge Flow)
    sig4_start = 3 * (total_bytes // 4)
    data_chunk = binary_data[sig4_start:]
    n_floats   = len(data_chunk) // 4
    n_points   = n_floats // 4

    all_floats = struct.unpack(f'<{n_floats}f', data_chunk[:n_floats * 4])

    temperature = []
    heat_flow   = []
    time_arr    = []

    for i in range(n_points):
        t   = all_floats[i * 4 + 0]   # Temperature (°C)  — col 0
        hf  = all_floats[i * 4 + 1]   # Heat Flow (mW)    — col 1
        # col 2: purge flow (~50 mL/min), col 3: secondary T or time

        # Filter invalid values
        if (not math.isnan(t) and not math.isinf(t) and
                not math.isnan(hf) and not math.isinf(hf) and
                -200 < t < 500 and -100 < hf < 100):
            temperature.append(t)
            heat_flow.append(hf)
            time_arr.append(i)   # index as proxy for time if not available

    return {
        'temperature': np.array(temperature),
        'heat_flow':   np.array(heat_flow),
        'time':        np.array(time_arr),
        'metadata':    metadata,
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == '__main__':

    # Change this path to your file
    filepath = "/home/nacho/Escritorio/THESIS_USB/DSC/your_file.001"

    result = read_ta_dsc_file(filepath)

    temperature = result['temperature']
    heat_flow   = result['heat_flow']
    metadata    = result['metadata']

    print(f"Points       : {len(temperature)}")
    print(f"Temperature  : {temperature.min():.1f} - {temperature.max():.1f} °C")
    print(f"Heat Flow    : {heat_flow.min():.4f} - {heat_flow.max():.4f} mW")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(temperature, heat_flow, color='#1f77b4', lw=1.5)
    ax.set_xlabel('Temperature (°C)', fontsize=13)
    ax.set_ylabel('Heat Flow (mW)', fontsize=13)
    ax.set_title(metadata.get('SampleName', 'DSC Q200'), fontsize=14,
                 fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()