# ATR-FTIR 
Python scripts for ATR-FTIR spectral analysis of clear aligner materials developed as part of a doctoral thesis.
## Contents
 
| File | Description |
|------|-------------|
| `ftir_petg_annotated.py` | Spectral analysis and annotation of PETG and copolyester materials |
| `ftir_tpu_annotated.py` | Spectral analysis and annotation of TPU materials |
| `ftir_tpu_classifier.py` | Automatic classification of TPU type, hard segment, and microphase separation indices |
| `PETG_annotated_various.py` | Multi spectral analysis and annotation of PETG and copolyester materials |
| `TPU_annotated_various.py`| Multi spectral analysis and annotation of TPU materials |
| `Culture_media_ftir.py`| Normalization and representation of spectras from culture medium in contact with plastics |

### `ftir_petg_annotated.py` and `ftir_tpu_annotated.py`
 
Both scripts follow the same pipeline:
 
1. **Load** — reads `.CSV` files exported from OMNIC. Two formats are supported automatically:
   - Format A: 2 columns, comma-separated `(wavenumber, %T)`
   - Format B: 4 columns, semicolon-separated with European decimal notation `(wavenumber×1e6, %T×1e6, ...)`
2. **Baseline correction** — asymmetric least-squares baseline using `derpsalsa` (pybaselines)
3. **Normalization** — spectrum normalized to a stable internal reference band
4. **Smoothing** *(optional)* — Savitzky-Golay filter (`scipy.signal.savgol_filter`, `window_length=15`, `polyorder=3`)
5. **Annotation** — key diagnostic bands marked with vertical labels and arrows
6. **Plot** — publication-quality figure displayed interactively

#### Normalization reference bands
 
| Material | Reference band |
|----------|---------------|
| PETG | 725 cm⁻¹ (C-H aromatic o.o.p.) |
| hTPU | 1410 cm⁻¹ (aromatic ring deformation) |
| culture medium | 3250 cm⁻¹  (O-H Stretching band in water) |

#### Annotated bands
 
**PETG** (`ftir_petg_annotated.py`):
- 722 cm⁻¹ — C-H aromatic out-of-plane deformation
- 1040 cm⁻¹ — CHDM trans conformer
- 1240 cm⁻¹ — C-O-C asymmetric stretch
- 1407 cm⁻¹ — aromatic ring deformation
- 1709 cm⁻¹ — C=O ester stretch
**hTPU** (`ftir_tpu_annotated.py`):
- 815 cm⁻¹ — MDI para-substituted aromatic C-H
- 1215 cm⁻¹ — C-N stretch (urethane)
- 1596 cm⁻¹ — Amide II / C=C MDI
- 1698 cm⁻¹ — C=O H-bonded ordered (hard segment)
- 2932 cm⁻¹ — CH₂ stretch (soft segment)
- 3310 cm⁻¹ — N-H stretch (urethane)
More bands can be included changing the code
### `ftir_tpu_classifier.py`
 
Automatic classification pipeline for TPU materials. Given a single spectrum file, the script returns:
 
- **Soft segment type** — polyester or polyether, with confidence score
- **Hard segment / diisocyanate** — MDI, TDI, or HDI, with confidence score
- **Residual THF contamination** — diagnostic for drop-cast films dissolved in THF (still in testing)
- **Carbonyl deconvolution** — three-Gaussian fit of the C=O region (1650–1760 cm⁻¹)
- **Degree of Phase Separation (DPS)** — fraction of hydrogen-bonded carbonyls relative to total carbonyl content
- **Degree of Ordering (DOR)** — fraction of ordered hydrogen-bonded carbonyls relative to total carbonyl content

#### Classification criteria
 
**Soft segment** (5 criteria, weighted scoring):
 
| Criterion | Band | Ester | Ether |
|-----------|------|-------|-------|
| C1 | Resolved peak ~1730 cm⁻¹ | +3 | +1 |
| C2 | Asymmetry A(1730)/A(max C=O) > 0.65 | +3 | — |
| C3 | Ratio A(1240)/A(1110) > 1.5 | +3 | +3 |
| C4 | Ratio A(1170)/A(1110) > 1.3 | +2 | +2 |
| C5 | Ratio A(1730)/A(1700) > 0.35 | +1 | — |
 
**Hard segment / diisocyanate:**
 
| Band | MDI | TDI | HDI |
|------|-----|-----|-----|
| 815 cm⁻¹ (aromatic C-H) | strong | moderate | absent |
| 1596 cm⁻¹ vs 1614 cm⁻¹ | 1596 > 1614 | 1614 > 1596 | — |
| 730 cm⁻¹ (CH₂ rocking) | — | — | strong |

#### DPS and DOR equations
 
DPS (%) $= \dfrac{A_{1698} + A_{1715}}{A_{1698} + A_{1715} + k \cdot A_{1730}} \times 100$
 
DOR(%) $= \frac{A_{1698}}{A_{1698} + A_{1715} + k \cdot A_{1730}} \times 100$
 
where $k = 1.2$ is the molar absorption coefficient ratio between free and hydrogen-bonded carbonyl groups.
 
## Usage
 
1. Clone the repository and navigate to this folder:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd ftir-atr
   ```
 
2. Install dependencies:
   ```bash
   pip install numpy matplotlib scipy lmfit pybaselines
   ```
 
3. Edit the `path` variable at the top of the annotation scripts to point to your `.CSV` file:
   ```python
   path = "/your/path/to/spectrum.CSV"
   ```
 
4. Run the annotation scripts:
   ```bash
   python ftir_petg_annotated.py
   python ftir_tpu_annotated.py
   ```
By default none of the scripts save images or data, only display graphs and data in terminal. Changes may be made for data and graphs saving
Two files are avialable for testing:
 - vivak_exolon.csv - PETG sheet
 - 98a.csv - TPU 98 shore A
   

