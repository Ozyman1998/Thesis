# TPU TGA ANALYZER
Python script for reading native TA Instruments TGA binary files (TGA Q50) and performing Gaussian deconvolution of the DTG curve to quantify soft segment (SS) and hard segment (HS) content in thermoplastic polyurethanes (TPUs).
## Background
The thermal degradation of TPUs typically produces two or more overlapping mass-loss events in the differential thermogravimetric (DTG) curve, corresponding to the sequential decomposition of the soft segment (lower T) and the hard segment (higher T). By fitting the DTG curve to a sum of Gaussian functions and computing the relative peak areas, it is possible to estimate the mass fraction of each phase without requiring prior knowledge of the TPU chemistry or formulation.

This approach has been applied to commercial orthodontic TPU aligner materials (MDI-based, polyether and polyester soft segments) characterised on a TA Instruments TGA Q50 under nitrogen atmosphere.
## Features
- Reads native TA Instruments binary files directly — no CSV export needed
- Parses the UTF-16 LE header and extracts temperature, weight and time arrays
- Computes −dW/dT (DTG) with Savitzky-Golay smoothing
- Automatic peak detection (scipy.signal.find_peaks) or user-supplied initial centres
- Gaussian deconvolution via scipy.optimize.curve_fit with physically constrained bounds
- SS/HS quantification by DTG peak area integration (scipy.integrate.trapezoid)
- Three-panel output figure: TGA curve | DTG deconvolution | results table
- Usable as a command-line script or as an importable module

## Installation
No special installation is required. Clone or copy the script and install the dependencies:
```{python}
pip install numpy matplotlib scipy
```
## Usage
### Command line
```{python}
python tga_tpu_analyzer.py path/to/file [polyether|polyester]
```
The script will display the figure interactively and save a PNG in the same directory as the input file.
### As module
```{python}
from tga_tpu_analyzer import analyze_tpu_tga

# Automatic peak detection
results = analyze_tpu_tga('TPU85A_N2', tpu_type='polyester')

# Manual peak centres (recommended when approximate temperatures are known)
results = analyze_tpu_tga('TPU98A_N2', tpu_type='polyester',
                          peak_centers=[330, 400, 457])

# Fix the number of peaks and the SS/HS cut-off temperature
results = analyze_tpu_tga('TPU52D_N2', tpu_type='polyether',
                          n_peaks=2, cutoff_temp=370,
                          save_figure=True, show_figure=False)
```
### Low level access 
Each processing step is also available independently:
```{python}
from tga_tpu_analyzer import read_ta_file, calculate_dtg, deconvolute_dtg

tga  = read_ta_file('TPU85A_N2')
temp_dtg, dtg = calculate_dtg(tga['temperature'], tga['weight'])

deconv = deconvolute_dtg(temp_dtg, dtg,
                         peak_centers=[330, 400],
                         temp_range=(250, 500))

for i, peak in enumerate(deconv['peaks']):
    print(f"Peak {i+1}: {peak['center']:.1f} °C  –  {peak['area_percent']:.1f}%")
```

## `analyze_tpu_tga` parameters
 
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | — | Path to the TA Instruments binary TGA file |
| `tpu_type` | `str` | `'polyether'` | TPU chemistry: `'polyether'` or `'polyester'` |
| `n_peaks` | `int` | `None` | Number of Gaussian peaks. `None` = automatic detection |
| `peak_centers` | `list` | `None` | Initial peak centres in °C, e.g. `[330, 400, 457]`. Overrides `n_peaks` |
| `temp_range` | `tuple` | `(250, 500)` | Temperature window for deconvolution (°C) |
| `cutoff_temp` | `float` | `None` | SS/HS boundary temperature. `None` = midpoint between first two peaks |
| `save_figure` | `bool` | `False` | Save PNG alongside the input file |
| `show_figure` | `bool` | `True` | Display figure interactively |


### Return value
 
A dict with the following keys:
 
| Key | Content |
|-----|---------|
| `tga_data` | Raw data dict: `temperature`, `weight`, `time`, `metadata` |
| `dtg` | Tuple `(temp_dtg, dtg)` — smoothed DTG arrays |
| `deconvolution` | Fit results: peak parameters, areas, fitted curve |
| `segments` | `soft_segment_percent`, `hard_segment_percent`, `cutoff_temp` |
| `sample_name` | Filename without extension |
| `figure` | `matplotlib.figure.Figure` object |
 
---
# File format notes
 
TA Instruments TGA Q50 files use a proprietary binary format. The reader locates the form-feed byte sequence (`\x0c\x00` in UTF-16 LE) that separates the ASCII/UTF-16 header from the binary data block. Each data record consists of five 32-bit little-endian floats:
 
```
[temperature (°C), weight (mg), ?, ?, time (min)]
```

Records with out-of-range values (temperature < 0 or > 1200 °C, negative weight) are discarded.
 
---
 
## Method notes
 
**Peak detection** uses `scipy.signal.find_peaks` with a minimum height and prominence of 5 % of the maximum DTG value, and a minimum inter-peak distance of 30 °C.
 
**Deconvolution** fits a sum of Gaussians to the Savitzky-Golay-smoothed DTG using `curve_fit` with bounds: amplitude ≥ 0, centre within `temp_range`, width between 5 and 100 °C.
 
**Segment quantification** assigns each Gaussian peak to SS or HS based on whether its centre falls below or above `cutoff_temp`. The phase content is computed as the ratio of the integrated peak area to the total DTG area in `temp_range`.
 
> **Note:** this method yields *mass fractions*, not volume fractions. Conversion to volume fractions requires the densities of the pure SS and HS phases. For MDI-based TPUs: ρ(HS) ≈ 1.30 g cm⁻³, ρ(SS, polyether) ≈ 1.05 g cm⁻³, ρ(SS, polyester) ≈ 1.18 g cm⁻³.



## Dependencies
 
| Package | Version tested | Purpose |
|---------|---------------|---------|
| `numpy` | ≥ 1.24 | Array operations |
| `matplotlib` | ≥ 3.7 | Plotting |
| `scipy` | ≥ 1.10 | Savitzky-Golay smoothing, peak detection, curve fitting, integration |
 
---
 
## Repository context
 
> *Thermoplastic materials for thermoformable clear aligners* — Doctoral thesis, IMDEA Materials Institute / Universidad Carlos III de Madrid.
 
Related scripts in other folder for TA Analysis file types reading (TA-instruments-python): `leer_dsc.py` (DSC Q200), `leer_dma.py` (DMA Q800), `leer_tga.py` (TGA Q50)





