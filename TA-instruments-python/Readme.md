# TA-instruments-python
Python scripts to read native binary files from TA Instruments thermal and mechanical analysis equipment, developed as part of a doctoral thesis. No proprietary software (TRIOS, Universal Analysis) is required.
## Contents
 
| File | Instrument | File format |
|------|-----------|-------------|
| `leer_dma.py` | DMA Q800 | Binary, no extension |
| `leer_dsc.py` | DSC Q200 | Binary, `.001` extension |
| `leer_tga.py` | TGA Q50  | Binary, no extension |
## How the files are structured
 
All three instruments export files in the same general format:
 
1. **Header** — UTF-16 LE encoded text containing metadata (sample name, method, date, operator)
2. **Form feed marker** — `0x0C 0x00` bytes separating header from data
3. **Binary data section** — little-endian float32 values packed in fixed-size records
The differences between instruments are the number of signals per record, the offset after the form feed, and which column maps to which physical quantity.
 
| Instrument | Signals per record | Record size | Data offset after form feed |
|------------|--------------------|-------------|----------------------------|
| DMA Q800   | 17                 | 68 bytes    | +2 bytes, then 1 byte skip |
| DSC Q200   | 4                  | 16 bytes    | +2 bytes, last quarter of binary section |
| TGA Q50    | 5                  | 20 bytes    | +7 bytes |
## What each script does
 
### `leer_dma.py` — DMA Q800
 
Reads a DMA Q800 binary file and returns five numpy arrays:
 
```python
from leer_dma import leer_archivo_dma
 
tiempo, temperatura, E_storage, E_loss, tan_delta = leer_archivo_dma("your_file")
```
 
| Array | Units | Signal |
|-------|-------|--------|
| `tiempo` | min | Time |
| `temperatura` | °C | Temperature |
| `E_storage` | MPa | Storage modulus E' |
| `E_loss` | MPa | Loss modulus E'' |
| `tan_delta` | — | Tan delta |
 
---
 
### `leer_dsc.py` — DSC Q200
 
Reads a DSC Q200 `.001` file and returns a dictionary:
 
```python
from leer_dsc import read_ta_dsc_file
 
result = read_ta_dsc_file("your_file.001")
 
temperature = result['temperature']   # °C
heat_flow   = result['heat_flow']     # mW
time        = result['time']          # min
metadata    = result['metadata']      # dict
```
 
Running the script directly also generates a Heat Flow vs Temperature plot.
 
---
 
### `leer_tga.py` — TGA Q50
 
Reads a TGA Q50 binary file, computes the DTG curve, and calculates key degradation parameters:
 
```python
from leer_tga import read_ta_tga_file, calculate_tga_parameters
 
tga    = read_ta_tga_file("your_file")
params = calculate_tga_parameters(tga['temperature'], tga['weight_pct'])
```
 
| Output | Description |
|--------|-------------|
| `tga['temperature']` | Temperature array (°C) |
| `tga['weight_pct']` | Weight array (%) |
| `tga['weight_mg']` | Weight array (mg) |
| `tga['time']` | Time array (min) |
| `params['T5']` | Temperature at 5% weight loss (°C) |
| `params['T10']` | Temperature at 10% weight loss (°C) |
| `params['T50']` | Temperature at 50% weight loss (°C) |
| `params['T_onset']` | Onset degradation temperature (°C) |
| `params['T_max']` | Temperature at maximum DTG (°C) |
| `params['DTG_max']` | Maximum DTG value (%/°C) |
| `params['residue']` | Final residue (%) |
| `params['dtg']` | DTG curve array (%/°C) |
 
Running the script directly also generates a TGA/DTG dual-axis plot.
 
## Usage
 
1. Clone the repository and navigate to this folder:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd ta-instruments-python
   ```
 
2. Install dependencies:
   ```bash
   pip install numpy matplotlib scipy
   ```
 
3. Edit the `filepath` variable at the bottom of each script:
   ```python
   filepath = "/your/path/to/your_file"
   ```
 
4. Run the script:
   ```bash
   python leer_dma.py
   python leer_dsc.py
   python leer_tga.py
   ```
 
## Dependencies
 
| Package | Version tested | Purpose |
|---------|---------------|---------|
| `numpy` | ≥ 1.24 | Array operations |
| `matplotlib` | ≥ 3.7 | Plotting |
| `scipy` | ≥ 1.10 | Savitzky-Golay smoothing (TGA DTG) |
 
## Notes
 
- Scripts were developed and tested on **Spyder (Anaconda)** under Linux.
- If figures do not appear in Spyder, set the graphics backend to **Qt5** under:
  `Tools → Preferences → IPython console → Graphics → Backend`
- Files are read directly from the native instrument format — no export to `.csv` or `.txt` is needed.
- The DSC Q200 stores data in the **last quarter** of the binary section. If your DSC file covers only a partial temperature range (e.g., second heating cycle only), this is expected behavior from the instrument's data structure.
- These scripts have been validated on files from the following instruments: DMA Q800, DSC Q200, TGA Q50. Other TA Instruments models may use a different binary layout and may require adaptation.
 
