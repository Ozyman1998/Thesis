# uv-vis
 
Python script for UV-VIS transmittance analysis of clear aligner materials developed as part of a doctoral thesis.
## Contents
 
| File | Description |
|------|-------------|
| `UV_VIS_BUENO.py` | Complete UV-VIS transmittance analysis of thermoplastics: UV protection, visible transparency, color balance, and pass/fail assessment |
 
## What the script does
 
Given a single `.txt` spectrum file, the script computes and reports:
 
1. **Load** — reads whitespace-delimited `.txt` files exported from the UV-VIS spectrometer. The sample name is extracted from the first line of the file.
2. **Smoothing** — Savitzky-Golay filter applied to the raw transmittance data (`window_length=11`, `polyorder=3`).
3. **UV protection analysis** — average transmittance and blocking (%) in the UVB (280–315 nm) and UVA (315–400 nm) ranges.
4. **Cut-off wavelengths** — wavelengths at T = 10%, T = 50%, and T = 90%.
5. **Visible transparency** — average transmittance over 400–700 nm, transmittance at 550 nm (critical wavelength for human perception), and per-band values (blue, green, yellow, red).
6. **Color balance** — red-blue transmittance difference, approximate yellowing index (YI), and spectral uniformity (standard deviation over the visible range).
7. **Pass/fail scoring** — five criteria evaluated independently; overall classification as SUITABLE / ACCEPTABLE / BORDERLINE / NOT SUITABLE.
8. **Recommendations** — automatic diagnostic messages flagging any out-of-specification results.
9. **Figures** — 5-panel figure saved as PNG (full spectrum, UV-visible transition zoom, per-band bar chart, visible absorbance, and summary box).
10. **Export** — processed data saved as `.csv` and a plain-text summary report saved as `.txt`.

## Spectral ranges
 
| Region | Range (nm) | Parameter reported |
|--------|-----------|-------------------|
| UVB | 280–315 | Average blocking (%) |
| UVA | 315–400 | Average blocking (%) |
| Visible | 400–700 | Average transmittance (%) |
| Blue | 450–495 | Average transmittance (%) |
| Green | 495–570 | Average transmittance (%) |
| Yellow | 570–590 | Average transmittance (%) |
| Red | 620–700 | Average transmittance (%) |
 
## Pass/fail criteria
 
| Criterion | Threshold | Target |
|-----------|-----------|--------|
| UVB blocking | > 85% | Photoprotection |
| UVA blocking | > 45% | Photoprotection |
| Visible transmittance (400–700 nm) | > 85% | Aesthetic invisibility |
| Transmittance at 550 nm | > 85% | Critical visibility wavelength |
| Red-blue difference | −3 to +5% | Neutral color balance |

A score of 4–5 out of 5 is required for a SUITABLE classification.
 
## Output files
 
For an input file with sample name `{sample}`, the script generates:
 
| File | Description |
|------|-------------|
| `{sample}_analisis_completo.png` | 5-panel figure (300 dpi) |
| `{sample}_datos_procesados.csv` | Wavelength, raw %T, smoothed %T, absorbance |
| `{sample}_resumen.txt` | Plain-text summary of all computed metrics |
## Usage
 
1. Clone the repository and navigate to this folder:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd uv-vis
   ```
 
2. Install dependencies:
   ```bash
   pip install numpy matplotlib scipy
   ```
 
3. Edit the `archivo` variable at the top of the script to point to your `.txt` file:
   ```python
   archivo = "/your/path/to/spectrum.txt"
   ```
 
4. Optionally adjust the smoothing parameters:
   ```python
   SMOOTH_WINDOW    = 11   # must be odd
   SMOOTH_POLYORDER = 3
   ```
 
5. Run the script:
   ```bash
   python UV_VIS_BUENO.py
   ```
 
## Input file format
 
The script expects a plain-text file with the following structure:
 
```
SampleName
Wavelength  Transmittance
280.0       0.12
281.0       0.15
...
```
 
- Line 1: sample name (string, no spaces)
- Line 2: column headers (skipped)
- Lines 3 to end−2: two whitespace-separated columns — wavelength (nm) and transmittance (%)
## Dependencies
 
| Package | Version tested | Purpose |
|---------|---------------|---------|
| `numpy` | ≥ 1.24 | Array operations |
| `matplotlib` | ≥ 3.7 | Plotting |
| `scipy` | ≥ 1.10 | Savitzky-Golay smoothing |
 
## Notes
 
- Scripts were developed and tested on **Spyder (Anaconda)** under Linux.
- If figures do not appear in Spyder, set the graphics backend to **Qt5** under:
  `Tools → Preferences → IPython console → Graphics → Backend`
- Output files are saved in the **current working directory**, not in the same folder as the input file.
 

