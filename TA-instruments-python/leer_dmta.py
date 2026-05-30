
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 15:28:03 2026
@author: nacho

Script to read TA Instruments DMA Q800 binary files.
Extracts data as numpy arrays: E', E'', tan delta, temperature, time.
"""

import struct
import numpy as np


def leer_archivo_dma(filepath):
    """
    Reads a TA Instruments DMA Q800 binary file and extracts the main signals.

    Parameters
    ----------
    filepath : str
        Path to the DMA file (no extension required)

    Returns
    -------
    tiempo      : numpy array — time (min)
    temperatura : numpy array — temperature (°C)
    E_storage   : numpy array — storage modulus E' (MPa)
    E_loss      : numpy array — loss modulus E'' (MPa)
    tan_delta   : numpy array — tan delta
    """
    with open(filepath, 'rb') as f:
        content = f.read()

    # Find start of data section (after UTF-16 LE form feed marker)
    header_byte_end = content.find(b'\x0c\x00')
    if header_byte_end == -1:
        raise ValueError("Data section marker not found. Is this a DMA Q800 file?")

    data_start   = header_byte_end + 2
    data_section = content[data_start:]

    # Each record: 17 signals × 4 bytes (float32) = 68 bytes
    # 1 byte offset at the start of the data section
    record_size  = 68
    n_signals    = 17
    first_offset = 1

    n_records = (len(data_section) - first_offset) // record_size

    data = []
    for i in range(n_records):
        offset = first_offset + i * record_size
        record = struct.unpack(
            f'<{n_signals}f',
            data_section[offset:offset + record_size]
        )
        data.append(record)

    data = np.array(data)

    # Keep only valid records (time >= 0)
    valid = data[:, 0] >= 0
    data  = data[valid]

    # Signal mapping (0-indexed):
    #   0 : Time (min)
    #   1 : Temperature (°C)
    #   2 : Storage Modulus E' (MPa)
    #   3 : Loss Modulus E'' (MPa)
    #   5 : Tan Delta
    tiempo      = data[:, 0]
    temperatura = data[:, 1]
    E_storage   = data[:, 2]
    E_loss      = data[:, 3]
    tan_delta   = data[:, 5]

    return tiempo, temperatura, E_storage, E_loss, tan_delta


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == '__main__':

    # Change this path to your file
    archivo = "/home/nacho/Escritorio/THESIS_USB/DMTA/your_file"

    tiempo, temperatura, E_storage, E_loss, tan_delta = leer_archivo_dma(archivo)

    print(f"Points        : {len(tiempo)}")
    print(f"Time          : {tiempo[0]:.2f} - {tiempo[-1]:.2f} min")
    print(f"Temperature   : {temperatura[0]:.1f} - {temperatura[-1]:.1f} °C")
    print(f"E' (Storage)  : {E_storage.min():.2f} - {E_storage.max():.2f} MPa")
    print(f"E'' (Loss)    : {E_loss.min():.2f} - {E_loss.max():.2f} MPa")
    print(f"Tan delta max : {tan_delta.max():.4f} at {temperatura[tan_delta.argmax()]:.1f} °C")

    # Arrays are ready to use — e.g.:
    # import matplotlib.pyplot as plt
    # plt.plot(temperatura, E_storage, label="E'")
    # plt.plot(temperatura, E_loss,    label="E''")
    # plt.show()