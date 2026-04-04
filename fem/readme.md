# Clear Aligner FEM — Biomechanical Analysis Framework
A finite element simulation framework for the biomechanical analysis of clear dental aligners, implemented entirely in open-source Python. Developed as part of a PhD thesis at IMDEA Materials and Secret Aligner S.L.
## Overview 
This tool computes the displacement, strain, stress, and von Mises stress fields that develop in a thermoplastic clear aligner shell upon insertion onto a patient-specific dental geometry. It models the mechanical interaction between the aligner and the tooth through an incremental node-to-surface contact formulation driven by a geometric activation mechanism.
Three simulation cases are supported:
- **Verification case**: cubic surrogate geometry (controlled numerical validation)
- **Single tooth case**: isolated anatomical crown (upper right first molar)
- **Full arch case**: complete lower dental arch
Results are exported in XDMF format for visualisation in ParaView.
## Dependencies
| Package | Version | Purpose |
|---|---|---|
| FEniCSx (dolfinx) | ≥ 0.7 | FEM framework |
| PETSc (petsc4py) | ≥ 3.18 | Linear solver (MUMPS) |
| MPI (mpi4py) | ≥ 3.1 | Parallel communication |
| NumPy | ≥ 1.24 | Array operations |
| SciPy | ≥ 1.10 | k-d tree contact detection |
| Gmsh  | > 3.0  | mesh visualization (optional) |
| ParaView | ≥ 5.11 | Post-processing (optional) |
## Installation (recommended: conda)
The easiest way to install FEniCSx and its dependencies is via conda:
``` bash
conda create -n fenicsx -c conda-forge fenics-dolfinx mpich petsc4py scipy
conda activate fenicsx
```
Verify the installation:
``` bash
python -c "import dolfinx; print(dolfinx.__version__)"
```
## Mesh Preparation
```
Patient STL (intraoral scan)
        ↓
  Meshmixer (surface cleaning, offset shell generation)
        ↓
  Gmsh (volumetric tetrahedralisation)
        ↓
  XDMF export (compatible with FEniCSx)
```
## Usage 
```bash
conda activate fenicsx
python clear_aligner_fem.py
```
### Changing the material parameters 
Edit the material parameters in the __main__ block:
```python
sim.E_shell  = 1300.0   # MPa
sim.nu_shell = 0.30

# hTPU
sim.E_shell  = 1800.0
sim.nu_shell = 0.45

# Multilayer (effective homogeneous)
sim.E_shell  = 800.0
sim.nu_shell = 0.40

# Polycarbonate (PC) — limiting case
sim.E_shell  = 2300.0
sim.nu_shell = 0.37
```
### Changing the activation distance
```python
sim.activation_mm = 0.25   # mm — standard clinical range: 0.15-0.30 mm
```
### Using custom mesh files 
```python
shell_path  = "path/to/your/aligner.xdmf"
tooth_path  = "path/to/your/tooth.xdmf"
```
## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `E_shell` | 1300.0 MPa | Aligner Young's modulus |
| `nu_shell` | 0.30 | Aligner Poisson's ratio |
| `E_tooth` | 18000.0 MPa | Tooth Young's modulus (enamel) |
| `nu_tooth` | 0.30 | Tooth Poisson's ratio |
| `activation_mm` | 0.20 mm | Radial geometric activation distance |
| `F_per_node` | 0.0004 N | Contact force per node |
| `num_load_steps` | 50 | Incremental load steps |
| `relaxation_factor` | 0.65 | Displacement relaxation per step |
| `contact_threshold` | 0.6 mm | Node-to-surface contact detection radius |

## Output Fields

| Field | Units | Description |
|---|---|---|
| `Displacement` | m | Nodal displacement vector |
| `Strain` | — | Linearised strain tensor |
| `Stress` | MPa | Cauchy stress tensor |
| `VonMises` | MPa | Von Mises equivalent stress (clipped at 2×p99) |
| `DistanceToTooth` | mm | Node-to-surface distance (aligner file only) |
| `DistanceToAligner` | mm | Node-to-surface distance (tooth file only) |
### Recommended ParaView settings
- Colour by VonMises`
- Set colour scale maximum to the **p99 value** reported in the terminal output
- Add units to the color bar Edit → Color Map Editor → Title: Von Mises stress (MPa)`
## Clinical Validation Ranges

| Quantity | Expected range | Reference |
|---|---|---|
| Aligner displacement | 0.05–0.30 mm | Hahn 2009, Salas 2025 |
| Aligner von Mises (p99) | 3–20 MPa | Gomez 2015, Elshazly 2022 |
| Tooth displacement | 1–50 µm | Gomez 2015 |
| Tooth von Mises | 0.01–2 MPa | Gomez 2015, Elshazly 2022 |
## Method Summary 
1. **Mesh loading**: Both meshes imported from XDMF; coordinates converted mm → m.
2. **Geometric activation**: Each aligner node displaced radially outward by `activation_mm` from the XY centroid of the shell, introducing the geometric discrepancy that drives contact force generation.
3. **Contact detection**: At each load step, aligner node coordinates are compared against the tooth surface. A node is in contact when its distance to the nearest tooth node falls below `contact_threshold`
4. **Incremental loading**: 50 load steps scaling forces linearly from 0 to full value. Relaxation factor applied to each incremental displacement.
5. **Direct RHS assembly**: Contact forces inserted directly into the right-hand side vector at contact node DOFs, avoiding L2 projection diffusion.
6. **Stiffness matrix**: Assembled once before the incremental loop and reused across all steps. MUMPS direct LU factorisation.
7. **Degenerate element exclusion**: Tetrahedral elements with volume < 1e-20 m³ excluded from the integration domain via restricted ufl.Measure`
8. **Tooth boundary conditions**: Apical 50% of tooth nodes (by Z height) fixed via `zeroRowsLocal. Isolated nodes from excluded elements fixed by unit diagonal insertion.
9. **Force transfer**: Reaction forces transferred to tooth via Newton's third law, averaged at nodes receiving multiple contributions.
10. **Post processing**: Strain, stress, and von Mises computed in DG-1 spaces. Von Mises clipped at 2×p99. Results interpolated to CG and exported to XDMF.
## Limitations of the code 
- **Only Linear elasticity applied**: Viscoelastic stress relaxation during clinical wear is not modelled. Results correspond to the initial insertion state.
- **No Periodontal ligament/gum/soft tissue**: The tooth is modelled as a near-rigid elastic body.
- **Single layer clear aligners**: Multilayer architectures (e.g. Zendura FLX,SmartTrack...) are approximated by an effective homogeneous modulus.
- **Empirical force calibration**: 'F_per_node' must be adjusted manually for each mesh to produce clinically meaningful force totals.
- **Uniform Activation**:  The geometric activation is radially uniform. Patient-specific activation fields require aligner digitisation and mesh registration.
  

