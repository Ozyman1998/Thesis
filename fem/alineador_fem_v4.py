#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEM Simulator for Clear Aligner Biomechanical Analysis — FEniCSx v4.1
======================================================================
Key features:
  - Radial geometric activation of the aligner shell (generates physical forces)
  - Manual RHS assembly for aligner and tooth (avoids L2 projection diffusion)
  - Exclusion of degenerate cells (vol=0) via restricted integration measure
    (ufl.Measure + meshtags)
  - Tooth boundary conditions via direct diagonal penalisation (zeroRowsLocal)
    (avoids NaN produced by locate_dofs_geometrical on vector spaces)
  - Separate Lame parameters for aligner (lam_s/mu_s) and tooth (lam_t/mu_t)

Dependencies:
  - FEniCSx (dolfinx), PETSc (petsc4py), MPI (mpi4py), UFL
  - NumPy, SciPy
  - ParaView (for post-processing XDMF output)

Input:  Two XDMF mesh files (aligner shell + tooth), generated via
        the STL -> Gmsh -> XDMF pipeline described in the thesis.
Output: Two XDMF result files containing displacement, strain, stress,
        von Mises stress, and node-to-surface distance fields.

References:
  Hahn 2009, Gomez 2015, Salas 2025
"""

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.fem import (Function, dirichletbc, locate_dofs_geometrical,
                          apply_lifting, set_bc)
from dolfinx.fem.petsc import (assemble_matrix, create_vector)
from dolfinx.mesh import meshtags
from ufl import (TrialFunction, TestFunction, Identity, grad, sym,
                 tr, inner, dx, sqrt)
from petsc4py import PETSc
import ufl
from scipy.spatial import cKDTree


class ClearAlignerFEM:
    """
    FEM simulator for clear aligner biomechanical analysis — v4.1

    Key parameters
    --------------
    E_shell          : Aligner Young's modulus (MPa). PETG ~ 1300.
    nu_shell         : Aligner Poisson's ratio.
    E_tooth          : Tooth Young's modulus (MPa). Enamel ~ 18000.
    nu_tooth         : Tooth Poisson's ratio.
    activation_mm    : Radial expansion of the aligner shell (mm).
                       Generates the geometric discrepancy driving contact.
                       Clinical range: 0.15-0.30 mm.
    F_per_node       : Contact force per active contact node (N).
                       F_total ~ F_per_node x n_contact_nodes.
                       Clinical range: 1-10 N total (Hahn 2009).
    num_load_steps   : Number of incremental load steps.
    relaxation_factor: Displacement relaxation factor per step (0.5-0.8).
    contact_threshold: Node-to-surface contact detection threshold (m).
    """

    def __init__(self):
        self.E_shell            = 1300.0   # MPa — PETG reference case
        self.nu_shell           = 0.30
        self.E_tooth            = 18000.0  # MPa — enamel (Gomez 2015)
        self.nu_tooth           = 0.30
        self.activation_mm      = 0.20     # mm — clinical activation distance
        self.F_per_node         = 0.0004   # N/node
        self.num_load_steps     = 50
        self.relaxation_factor  = 0.65
        self.contact_threshold  = 0.6e-3   # m
        self.shell_mesh         = None
        self.tooth_mesh         = None

    # ─────────────────────────────────────────────────────────────────
    # MESH LOADING
    # ─────────────────────────────────────────────────────────────────

    def load_meshes(self, shell_path, tooth_path):
        """Load aligner and tooth meshes from XDMF files.
        Coordinates are converted from mm to m on import."""
        print("=" * 70)
        print("LOADING 3D MESHES")
        print("=" * 70)
        with XDMFFile(MPI.COMM_WORLD, shell_path, "r") as f:
            self.shell_mesh = f.read_mesh(name="Grid")
        with XDMFFile(MPI.COMM_WORLD, tooth_path, "r") as f:
            self.tooth_mesh = f.read_mesh(name="Grid")
        # Convert coordinates from mm to m
        self.shell_mesh.geometry.x[:] *= 1e-3
        self.tooth_mesh.geometry.x[:] *= 1e-3
        sc, tc = self.shell_mesh.geometry.x, self.tooth_mesh.geometry.x
        print(f"  Aligner : {self.shell_mesh.topology.index_map(0).size_local} vertices")
        print(f"  Tooth   : {self.tooth_mesh.topology.index_map(0).size_local} vertices")
        for coords, label in [(sc, "Aligner"), (tc, "Tooth")]:
            print(f"\n  Bounding Box {label}:")
            for ax, lbl in enumerate(["X", "Y", "Z"]):
                print(f"    {lbl}: [{np.min(coords[:,ax])*1e3:.2f},"
                      f" {np.max(coords[:,ax])*1e3:.2f}] mm")

    # ─────────────────────────────────────────────────────────────────
    # GEOMETRIC ACTIVATION
    # ─────────────────────────────────────────────────────────────────

    def apply_activation(self):
        """Expand aligner shell radially outward by activation_mm.
        Each node is displaced along the radial direction from the
        XY centroid of the shell, introducing the geometric discrepancy
        that drives contact force generation."""
        act_m  = self.activation_mm * 1e-3
        coords = self.shell_mesh.geometry.x
        cx, cy = np.mean(coords[:, 0]), np.mean(coords[:, 1])
        n_exp  = 0
        for i in range(len(coords)):
            ddx = coords[i, 0] - cx
            ddy = coords[i, 1] - cy
            r   = np.sqrt(ddx**2 + ddy**2)
            if r > 1e-10:
                coords[i, 0] += (ddx / r) * act_m
                coords[i, 1] += (ddy / r) * act_m
                n_exp += 1
        print(f"  Activation: {self.activation_mm:.2f} mm  ({n_exp} nodes)")
        print(f"  XY centroid: ({cx*1e3:.2f}, {cy*1e3:.2f}) mm")

    # ─────────────────────────────────────────────────────────────────
    # CONSTITUTIVE MODEL
    # ─────────────────────────────────────────────────────────────────

    def lame_parameters(self, E, nu):
        """Compute Lame parameters from Young's modulus and Poisson's ratio."""
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu  = E / (2 * (1 + nu))
        return lam, mu

    def epsilon(self, u):
        """Linearised strain tensor."""
        return sym(grad(u))

    def sigma(self, u, lam, mu):
        """Cauchy stress tensor (linear elastic, isotropic)."""
        d = len(u)
        return lam * tr(self.epsilon(u)) * Identity(d) + 2 * mu * self.epsilon(u)

    def von_mises(self, s):
        """Von Mises equivalent stress from stress tensor."""
        d     = s.ufl_shape[0]
        s_dev = s - (1./3.) * tr(s) * Identity(d)
        return sqrt(3./2. * inner(s_dev, s_dev))

    # ─────────────────────────────────────────────────────────────────
    # CONTACT DETECTION
    # ─────────────────────────────────────────────────────────────────

    def detect_contact(self, shell_coords):
        """Detect aligner nodes in contact with the tooth surface.
        Contact is declared when the node-to-surface distance falls
        below contact_threshold. A fixed force F_per_node is applied
        at each active contact node, directed toward the nearest
        tooth surface node."""
        tc    = self.tooth_mesh.geometry.x
        n     = len(shell_coords)
        flags = np.zeros(n, dtype=bool)
        idx_t = np.zeros(n, dtype=int)
        pens  = []

        for i, p in enumerate(shell_coords):
            dists = np.linalg.norm(tc - p, axis=1)
            d_min = np.min(dists)
            if d_min < self.contact_threshold:
                flags[i] = True
                idx_t[i] = np.argmin(dists)
                pens.append(max(0.0, self.contact_threshold - d_min))

        # Compute force vectors at contact nodes
        forces = np.zeros((n, 3))
        for i in range(n):
            if flags[i]:
                direction = tc[idx_t[i]] - shell_coords[i]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                forces[i] = direction * self.F_per_node

        return flags, forces, pens, idx_t

    # ─────────────────────────────────────────────────────────────────
    # ALIGNER BOUNDARY CONDITIONS
    # ─────────────────────────────────────────────────────────────────

    def aligner_bcs(self, V):
        """Fix the top and bottom 10% of the aligner shell in Z.
        This simulates the constraint imposed by the gingival margin
        and the occlusal border during insertion."""
        z            = self.shell_mesh.geometry.x[:, 2]
        z_min, z_max = np.min(z), np.max(z)
        dz           = z_max - z_min
        z_inf        = z_min + 0.10 * dz
        z_sup        = z_max - 0.10 * dz
        dofs = locate_dofs_geometrical(
            V, lambda x: np.logical_or(x[2] <= z_inf, x[2] >= z_sup)
        )
        if len(dofs) == 0:
            dofs = locate_dofs_geometrical(V, lambda x: x[2] <= z_min + 0.15*dz)
        u0 = Function(V); u0.x.array[:] = 0.0
        print(f"    Aligner BCs: {len(dofs)} DOFs  "
              f"({100*len(dofs)//3/len(z):.1f}% nodes)")
        return [dirichletbc(u0, dofs)]

    # ─────────────────────────────────────────────────────────────────
    # ALIGNER INCREMENTAL ANALYSIS
    # ─────────────────────────────────────────────────────────────────

    def solve_aligner(self):
        """Incremental linear elastic analysis of the aligner shell.
        Uses separate Lame parameters lam_s, mu_s for the aligner.
        Contact forces are inserted directly into the RHS vector
        (direct nodal assembly) to avoid L2 projection diffusion."""
        print(f"\n{'='*70}")
        print(f"INCREMENTAL ANALYSIS — ALIGNER  ({self.num_load_steps} steps)")
        print(f"  E={self.E_shell} MPa  nu={self.nu_shell}  "
              f"F/node={self.F_per_node} N  relax={self.relaxation_factor}  "
              f"activation={self.activation_mm} mm")
        print(f"{'='*70}\n")

        V   = fem.functionspace(self.shell_mesh, ("Lagrange", 1, (3,)))
        u_t = TrialFunction(V)
        v   = TestFunction(V)

        # Lame parameters for the ALIGNER
        lam_s, mu_s = self.lame_parameters(self.E_shell * 1e6, self.nu_shell)

        # Bilinear form over the full aligner domain
        a_form = fem.form(
            inner(self.sigma(u_t, lam_s, mu_s), self.epsilon(v)) * dx
        )
        zero_s = Function(V); zero_s.x.array[:] = 0.0
        L_form = fem.form(inner(zero_s, v) * dx)

        u_total = Function(V); u_total.x.array[:] = 0.0
        coords_orig   = self.shell_mesh.geometry.x.copy()
        coords_actual = coords_orig.copy()

        bcs = self.aligner_bcs(V)

        # Assemble stiffness matrix once — reused across all load steps
        print("  Assembling aligner stiffness matrix...")
        A = assemble_matrix(a_form, bcs=bcs)
        A.assemble()

        # Direct LU factorisation via MUMPS
        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setFromOptions()

        flags_fin = forces_fin = idx_fin = None
        n_div = 0

        for step in range(1, self.num_load_steps + 1):
            flags, forces, pens, idx_t = self.detect_contact(coords_actual)
            flags_fin, forces_fin, idx_fin = (flags.copy(),
                                              forces.copy(),
                                              idx_t.copy())
            n_c     = int(np.sum(flags))
            pen_str = f"pen={np.mean(pens)*1e3:.3f}mm  " if pens else ""

            if n_c == 0:
                print(f"  Step {step:02d}  no contact"); break

            # Scale load linearly from 0 to full value over num_load_steps
            factor = step / self.num_load_steps
            b = create_vector(L_form)
            with b.localForm() as bl: bl.set(0.0)

            # Direct nodal force insertion into RHS (avoids L2 diffusion)
            for i in range(len(self.shell_mesh.geometry.x)):
                if flags[i]:
                    fi = forces[i] * factor
                    b.array[3*i]   += fi[0]
                    b.array[3*i+1] += fi[1]
                    b.array[3*i+2] += fi[2]

            apply_lifting(b, [a_form], [bcs])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bcs)

            u_step = Function(V)
            ksp.solve(b, u_step.x.petsc_vec)
            u_step.x.scatter_forward()

            # Apply relaxation factor to improve convergence
            u_step.x.array[:] *= self.relaxation_factor
            u_total.x.array[:] += u_step.x.array[:]

            # Update deformed coordinates for next contact detection step
            for i in range(len(coords_actual)):
                coords_actual[i] = coords_orig[i] + u_total.x.array[3*i:3*i+3]

            d_tot = np.max(np.sqrt(u_total.x.array[0::3]**2 +
                                   u_total.x.array[1::3]**2 +
                                   u_total.x.array[2::3]**2))
            d_pas = np.max(np.sqrt(u_step.x.array[0::3]**2 +
                                   u_step.x.array[1::3]**2 +
                                   u_step.x.array[2::3]**2))

            print(f"  Step {step:02d}/{self.num_load_steps}  "
                  f"n_c={n_c}({100*n_c/len(coords_actual):.1f}%)  "
                  f"{pen_str}disp={d_tot*1e3:.4f}mm")

            # Divergence check
            if step > 5 and d_pas > 2e-3:
                n_div += 1
                if n_div > 3:
                    print("  DIVERGENCE DETECTED"); break
            else:
                n_div = max(0, n_div - 1)

        ksp.destroy()
        print(f"\n{'='*70}\n")
        return u_total, V, lam_s, mu_s, flags_fin, forces_fin, idx_fin

    # ─────────────────────────────────────────────────────────────────
    # FORCE TRANSFER (Newton's Third Law)
    # ─────────────────────────────────────────────────────────────────

    def transfer_forces(self, shell_flags, shell_forces, tooth_idx):
        """Transfer reaction forces from aligner to tooth via Newton III.
        For tooth nodes receiving contributions from multiple aligner
        contact nodes, forces are averaged to avoid double-counting."""
        print("  Transferring forces to tooth...")
        n_t = len(self.tooth_mesh.geometry.x)
        f_d = np.zeros((n_t, 3))
        cnt = np.zeros(n_t, dtype=int)
        for i_s, in_contact in enumerate(shell_flags):
            if in_contact:
                i_t = tooth_idx[i_s]
                f_d[i_t] -= shell_forces[i_s]  # Newton III: opposite sign
                cnt[i_t] += 1
        for i in range(n_t):
            if cnt[i] > 1:
                f_d[i] /= cnt[i]  # Average if multiple aligner nodes map to same tooth node
        n_c   = int(np.sum(cnt > 0))
        F_tot = float(np.sum(np.linalg.norm(f_d, axis=1)))
        print(f"    Nodes: {n_c}  F_total: {F_tot:.4f} N")
        return f_d

    # ─────────────────────────────────────────────────────────────────
    # TOOTH ANALYSIS
    # ─────────────────────────────────────────────────────────────────

    def solve_tooth(self, tooth_nodal_forces):
        """Linear elastic analysis of the tooth.
        Uses separate Lame parameters lam_t, mu_t for the tooth.

        Key implementation details:
          1. Degenerate elements (vol=0) are excluded via restricted
             integration measure (ufl.Measure + meshtags).
          2. Stiffness matrix assembled without BCs to avoid NaN entries.
          3. BCs enforced via zeroRowsLocal (direct diagonal penalisation).
          4. Isolated nodes from excluded elements fixed by unit diagonal.
        """
        print(f"\n{'='*70}")
        print(f"FEM ANALYSIS — TOOTH  (valid cells + penalisation BCs)")
        print(f"  E={self.E_tooth} MPa  nu={self.nu_tooth}")
        print(f"{'='*70}\n")

        V   = fem.functionspace(self.tooth_mesh, ("Lagrange", 1, (3,)))
        u_t = TrialFunction(V)
        v   = TestFunction(V)

        # Lame parameters for the TOOTH
        lam_t, mu_t = self.lame_parameters(self.E_tooth * 1e6, self.nu_tooth)

        # Step 1: Identify and exclude degenerate tetrahedral elements
        # Degenerate elements have zero volume and produce NaN in the
        # stiffness matrix. They arise from coplanar STL triangulation.
        tdim = self.tooth_mesh.topology.dim
        self.tooth_mesh.topology.create_connectivity(tdim, 0)
        conn_t   = self.tooth_mesh.topology.connectivity(tdim, 0)
        coords_t = self.tooth_mesh.geometry.x
        n_cells  = self.tooth_mesh.topology.index_map(tdim).size_local

        valid_cells, n_degen = [], 0
        for c in range(n_cells):
            nodes = conn_t.links(c)
            pts   = coords_t[nodes]
            if len(pts) == 4:
                vol = abs(np.dot(pts[1]-pts[0],
                                 np.cross(pts[2]-pts[0],
                                          pts[3]-pts[0]))) / 6.0
                if vol > 1e-20:
                    valid_cells.append(c)
                else:
                    n_degen += 1

        valid_cells = np.array(valid_cells, dtype=np.int32)
        print(f"  Valid cells: {len(valid_cells)}/{n_cells}  "
              f"({n_degen} degenerate excluded)")

        # Restricted integration measure over valid cells only
        tags_t   = meshtags(self.tooth_mesh, tdim, valid_cells,
                            np.ones(len(valid_cells), dtype=np.int32))
        dx_tooth = ufl.Measure("dx", domain=self.tooth_mesh,
                               subdomain_data=tags_t, subdomain_id=1)

        # Step 2: Assemble bilinear and linear forms over restricted domain
        a_form = fem.form(
            inner(self.sigma(u_t, lam_t, mu_t), self.epsilon(v)) * dx_tooth
        )
        zero_t = Function(V); zero_t.x.array[:] = 0.0
        L_form = fem.form(inner(zero_t, v) * dx_tooth)

        # Step 3: Assemble stiffness matrix WITHOUT boundary conditions
        # (applying BCs before assembly causes NaN in rows with no element
        # contributions from the restricted domain)
        print("  Assembling tooth stiffness matrix (no BCs)...")
        A = assemble_matrix(a_form)
        A.assemble()

        b = create_vector(L_form)
        with b.localForm() as bl: bl.set(0.0)

        # Step 4: Insert nodal forces into RHS
        bs    = V.dofmap.index_map_bs
        n_ins = 0
        for i in range(len(coords_t)):
            fi = tooth_nodal_forces[i]
            if np.linalg.norm(fi) > 1e-15:
                d0, d1, d2 = bs*i, bs*i+1, bs*i+2
                if d2 < len(b.array):
                    b.array[d0] += fi[0]
                    b.array[d1] += fi[1]
                    b.array[d2] += fi[2]
                    n_ins += 1
        print(f"  Forces: {n_ins} nodes  RHS norm: {b.norm():.4f} N")

        # Step 5: Apply Dirichlet BCs via zeroRowsLocal
        # Fix the apical 50% of tooth nodes (by Z height) to represent
        # the constraint provided by the periodontal ligament and alveolar bone.
        z_arr    = coords_t[:, 2]
        z_thresh = float(np.sort(z_arr)[len(z_arr)//2])

        fixed_dofs = []
        for i in range(len(z_arr)):
            if z_arr[i] <= z_thresh:
                fixed_dofs.extend([bs*i, bs*i+1, bs*i+2])

        fixed_dofs = np.array(fixed_dofs, dtype=np.int32)
        print(f"  BCs zeroRowsLocal: {len(fixed_dofs)//3} nodes fixed")

        # Zero RHS at fixed DOFs and set unit diagonal in A
        b.array[fixed_dofs] = 0.0
        A.zeroRowsLocal(fixed_dofs, diag=1.0)
        A.assemblyBegin(); A.assemblyEnd()

        # Step 6: Fix isolated nodes from excluded degenerate elements
        # These nodes have zero diagonal entries, making A singular.
        # Solution: insert unit diagonal and zero RHS (fixes u=0 at these nodes).
        diag_check = A.getDiagonal().array
        zero_idx   = np.where(np.abs(diag_check) < 1e-10)[0].astype(np.int32)
        if len(zero_idx) > 0:
            print(f"  Fixing {len(zero_idx)} isolated DOFs...")
            A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            for dof in zero_idx:
                A.setValueLocal(int(dof), int(dof), 1.0,
                                PETSc.InsertMode.ADD_VALUES)
                if int(dof) < len(b.array):
                    b.array[int(dof)] = 0.0
            A.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
            A.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)

        # Final diagonal check
        diag2 = A.getDiagonal().array
        n_nan = int(np.sum(~np.isfinite(diag2)))
        n_zer = int(np.sum(np.abs(diag2) < 1e-10))
        print(f"  Final diagonal — nan: {n_nan}  zeros: {n_zer}  "
              f"min: {np.nanmin(np.abs(diag2[diag2!=0])):.2e}  "
              f"max: {np.nanmax(diag2):.2e}")

        # Step 7: Solve via MUMPS direct LU factorisation
        print("  Solving (MUMPS)...")
        u_tooth = Function(V)
        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.setFromOptions()
        ksp.solve(b, u_tooth.x.petsc_vec)
        u_tooth.x.scatter_forward()
        ksp.destroy()

        disp = np.sqrt(u_tooth.x.array[0::3]**2 +
                       u_tooth.x.array[1::3]**2 +
                       u_tooth.x.array[2::3]**2)

        print(f"\n  Tooth results:")
        if np.isfinite(np.max(disp)):
            print(f"    Max displacement:  {np.max(disp)*1e6:.4f} um")
            print(f"    Mean displacement: {np.mean(disp)*1e6:.4f} um")
            print(f"    Solution converged")
        else:
            print(f"    Solution unbounded — check mesh and BCs")
        print(f"\n{'='*70}\n")

        return u_tooth, V, lam_t, mu_t

    # ─────────────────────────────────────────────────────────────────
    # FIELD COMPUTATION (strain, stress, von Mises)
    # ─────────────────────────────────────────────────────────────────

    def compute_fields(self, u_sol, target_mesh, lam, mu, label=""):
        """Compute strain, stress, and von Mises stress fields.
        DG-1 spaces are used to avoid spurious smoothing across element
        boundaries. Von Mises values are clipped at 2x the 99th percentile
        to suppress singularity artefacts at mesh boundaries."""
        print(f"  Computing fields — {label}...")
        V_ten = fem.functionspace(target_mesh, ("DG", 1, (3, 3)))
        V_sca = fem.functionspace(target_mesh, ("DG", 1))

        strain = Function(V_ten)
        strain.interpolate(fem.Expression(
            self.epsilon(u_sol), V_ten.element.interpolation_points()))

        sig_e  = self.sigma(u_sol, lam, mu)
        stress = Function(V_ten)
        stress.interpolate(fem.Expression(
            sig_e, V_ten.element.interpolation_points()))

        vm = Function(V_sca)
        vm.interpolate(fem.Expression(
            self.von_mises(sig_e), V_sca.element.interpolation_points()))

        # Convert Pa to MPa
        stress.x.array[:] *= 1e-6
        vm.x.array[:]      *= 1e-6

        arr = vm.x.array[:]
        # Replace NaN/Inf from degenerate cells with zero before processing
        arr = np.where(np.isfinite(arr), arr, 0.0)
        vm.x.array[:] = arr

        if len(arr) > 0 and np.max(arr) > 0:
            p99 = np.percentile(arr, 99)
            arr[arr > p99 * 2] = p99  # Clip outliers at 2x p99
            vm.x.array[:] = arr
            print(f"    VM max={np.max(arr):.3f} MPa  p99={p99:.3f} MPa")
        else:
            print(f"    VM = 0 (no significant stress)")

        return strain, stress, vm

    def compute_distances(self, mesh_orig, mesh_dest, u_sol=None):
        """Compute node-to-surface distances from mesh_orig to mesh_dest.
        If u_sol is provided, deformed coordinates are used for mesh_orig."""
        coords = mesh_orig.geometry.x.copy()
        if u_sol is not None:
            n_dofs = len(u_sol.x.array) // 3
            for i in range(min(len(coords), n_dofs)):
                coords[i] += u_sol.x.array[3*i:3*i+3]
        dist, _ = cKDTree(mesh_dest.geometry.x).query(coords)
        return dist

    # ─────────────────────────────────────────────────────────────────
    # INSERTION QUALITY VERIFICATION
    # ─────────────────────────────────────────────────────────────────

    def verify_insertion(self, u_shell, flags, forces):
        """Assess aligner seating quality against clinical criteria:
        - Coverage > 70% of aligner nodes in contact
        - Adjustment (gap < 0.1mm) > 80% of nodes
        - Gaps > 0.5mm < 10% of nodes"""
        print("\n" + "="*70)
        print("INSERTION QUALITY CHECK")
        print("="*70)
        n_tot = len(flags)
        n_c   = int(np.sum(flags))
        cov   = 100.0 * n_c / n_tot
        dist  = self.compute_distances(self.shell_mesh, self.tooth_mesh, u_shell)
        n_adj = int(np.sum(dist < 0.1e-3))
        n_gap = int(np.sum(dist > 0.5e-3))
        adj_p = 100.0 * n_adj / n_tot
        gap_p = 100.0 * n_gap / n_tot
        print(f"\n  Coverage:        {n_c}/{n_tot} ({cov:.1f}%)")
        print(f"  Adjustment <0.1mm: {n_adj}/{n_tot} ({adj_p:.1f}%)")
        print(f"  Gaps >0.5mm:     {n_gap}/{n_tot} ({gap_p:.1f}%)")
        print(f"  Dist min/max/mean: {np.min(dist)*1e3:.3f} / "
              f"{np.max(dist)*1e3:.3f} / {np.mean(dist)*1e3:.3f} mm")
        failures = []
        if cov   < 70: failures.append(f"Coverage {cov:.1f}% < 70%")
        if adj_p < 80: failures.append(f"Adjustment {adj_p:.1f}% < 80%")
        if gap_p > 10: failures.append(f"Gaps {gap_p:.1f}% > 10%")
        ok = len(failures) == 0
        print(f"\n  {'INSERTION COMPLETE' if ok else 'INSERTION INCOMPLETE'}")
        for f in failures: print(f"     * {f}")
        print("="*70 + "\n")
        return {'coverage': cov, 'adjustment': adj_p, 'gaps': gap_p,
                'dist_mean': float(np.mean(dist))}, ok, dist

    # ─────────────────────────────────────────────────────────────────
    # XDMF EXPORT
    # ─────────────────────────────────────────────────────────────────

    def _export(self, target_mesh, out_path, u_sol, strain, stress, vm,
                dist_arr, dist_name):
        """Export all result fields to XDMF for ParaView visualisation.
        DG fields are interpolated to CG for compatibility."""
        V_tCG = fem.functionspace(target_mesh, ("Lagrange", 1, (3, 3)))
        V_sCG = fem.functionspace(target_mesh, ("Lagrange", 1))
        def _t(src): f = Function(V_tCG); f.interpolate(src); return f
        def _s(src): f = Function(V_sCG); f.interpolate(src); return f
        sc = _t(strain); sc.name = "Strain"
        ss = _t(stress); ss.name = "Stress"
        sv = _s(vm);     sv.name = "VonMises"
        df = Function(V_sCG)
        df.x.array[:] = dist_arr * 1e3  # Convert m to mm
        df.name = dist_name
        u_sol.name = "Displacement"
        with XDMFFile(MPI.COMM_WORLD, out_path, "w") as xdmf:
            xdmf.write_mesh(target_mesh)
            for fn in [u_sol, sc, ss, sv, df]:
                xdmf.write_function(fn)

    def export_aligner(self, u_sol, strain, stress, vm, dist, out_path):
        print(f"  Exporting ALIGNER: {out_path}")
        self._export(self.shell_mesh, out_path, u_sol, strain, stress, vm,
                     dist, "DistanceToTooth")
        d = np.sqrt(u_sol.x.array[0::3]**2 + u_sol.x.array[1::3]**2 +
                    u_sol.x.array[2::3]**2)
        print(f"    Max disp: {np.max(d)*1e3:.4f} mm  "
              f"VM max: {np.max(vm.x.array):.3f} MPa")

    def export_tooth(self, u_sol, strain, stress, vm, dist, out_path):
        print(f"  Exporting TOOTH: {out_path}")
        self._export(self.tooth_mesh, out_path, u_sol, strain, stress, vm,
                     dist, "DistanceToAligner")
        d = np.sqrt(u_sol.x.array[0::3]**2 + u_sol.x.array[1::3]**2 +
                    u_sol.x.array[2::3]**2)
        if np.isfinite(np.max(d)):
            print(f"    Max disp: {np.max(d)*1e6:.4f} um  "
                  f"VM max: {np.max(vm.x.array):.4f} MPa")
        else:
            print("    Solution unbounded — check mesh and BCs")

    # ─────────────────────────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────────────────────────

    def run_analysis(self, shell_path, tooth_path,
                     out_aligner="results/aligner.xdmf",
                     out_tooth="results/tooth.xdmf"):
        """Run the complete FEM analysis pipeline:
        1. Load meshes
        2. Apply geometric activation to aligner shell
        3. Detect initial contacts
        4. Incremental aligner analysis (num_load_steps steps)
        5. Verify insertion quality
        6. Transfer forces to tooth (Newton III)
        7. Tooth analysis
        8. Compute output fields (strain, stress, von Mises)
        9. Export results to XDMF"""

        print("\n" + "="*70)
        print("FEM ANALYSIS — ALIGNER + TOOTH  (v4.1)")
        print("="*70 + "\n")

        self.load_meshes(shell_path, tooth_path)

        print("\n" + "="*70)
        print("GEOMETRIC ACTIVATION")
        print("="*70)
        self.apply_activation()

        # Check initial contact after activation
        flags_ini, _, _, _ = self.detect_contact(self.shell_mesh.geometry.x)
        n_ini = int(np.sum(flags_ini))
        if n_ini == 0:
            print("No initial contacts detected. Check activation distance and mesh alignment.")
            return None
        print(f"\n  Initial contacts: {n_ini} nodes")

        # Aligner incremental analysis — returns aligner Lame parameters
        (u_shell, V_shell, lam_s, mu_s,
         flags_fin, forces_fin, idx_fin) = self.solve_aligner()

        # Insertion quality verification
        criteria, ok, dist_s2t = self.verify_insertion(
            u_shell, flags_fin, forces_fin)

        # Force transfer and tooth analysis — returns tooth Lame parameters
        f_d = self.transfer_forces(flags_fin, forces_fin, idx_fin)
        u_tooth, V_tooth, lam_t, mu_t = self.solve_tooth(f_d)

        # Compute output fields for both bodies
        print("\n" + "="*70)
        print("COMPUTING OUTPUT FIELDS")
        print("="*70)
        st_s, ss_s, vm_s = self.compute_fields(
            u_shell, self.shell_mesh, lam_s, mu_s, "Aligner")
        st_t, ss_t, vm_t = self.compute_fields(
            u_tooth, self.tooth_mesh, lam_t, mu_t, "Tooth")

        dist_t2s = self.compute_distances(
            self.tooth_mesh, self.shell_mesh, u_shell)

        # Export to XDMF
        print("\n" + "="*70)
        print("EXPORTING RESULTS")
        print("="*70)
        self.export_aligner(u_shell, st_s, ss_s, vm_s, dist_s2t, out_aligner)
        self.export_tooth(u_tooth, st_t, ss_t, vm_t, dist_t2s, out_tooth)

        d_s = np.sqrt(u_shell.x.array[0::3]**2 + u_shell.x.array[1::3]**2 +
                      u_shell.x.array[2::3]**2)
        d_t = np.sqrt(u_tooth.x.array[0::3]**2 + u_tooth.x.array[1::3]**2 +
                      u_tooth.x.array[2::3]**2)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE  (v4.1)")
        print("="*70)
        print(f"\n  ALIGNER: disp={np.max(d_s)*1e3:.4f} mm  "
              f"VM={np.max(vm_s.x.array):.3f} MPa")
        if np.isfinite(np.max(d_t)):
            print(f"  TOOTH:   disp={np.max(d_t)*1e6:.4f} um  "
                  f"VM={np.max(vm_t.x.array):.4f} MPa")
        else:
            print("  TOOTH:   check solution")
        print(f"\n  Output files:")
        print(f"  * {out_aligner}")
        print(f"  * {out_tooth}")
        print("="*70 + "\n")

        return {
            'aligner': {'u': u_shell, 'vm': vm_s,
                        'strain': st_s, 'stress': ss_s},
            'tooth':   {'u': u_tooth, 'vm': vm_t,
                        'strain': st_t, 'stress': ss_t},
            'criteria': criteria, 'insertion_ok': ok,
        }


# ═════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    sim = ClearAlignerFEM()

    # Input mesh files (XDMF format, generated via STL -> Gmsh -> XDMF)
    shell_path  = "meshes/aligner.xdmf"
    tooth_path  = "meshes/tooth.xdmf"
    out_aligner = "results/aligner_results.xdmf"
    out_tooth   = "results/tooth_results.xdmf"

    # --- Material properties ---
    # Uncomment and modify to change aligner material:
    # PETG:       E=1300, nu=0.30
    # hTPU:       E=1800, nu=0.45
    # Multilayer: E=800,  nu=0.40
    # PC:         E=2300, nu=0.37
    sim.E_shell  = 1300.0   # MPa — PETG reference case
    sim.nu_shell = 0.30
    sim.E_tooth  = 18000.0  # MPa — enamel surrogate (Gomez 2015)
    sim.nu_tooth = 0.30

    # --- Loading parameters ---
    sim.activation_mm = 0.20    # mm — clinical activation distance
    sim.F_per_node    = 0.0004  # N/node — calibrated for single tooth case

    # --- Run analysis ---
    results = sim.run_analysis(shell_path, tooth_path, out_aligner, out_tooth)

    print("EXPECTED CLINICAL RANGES:")
    print("  Aligner: disp 0.05-0.30 mm  VM 3-20 MPa")
    print("  Tooth:   disp 1-50 um       VM 0.01-2 MPa")
    print("  Refs: Hahn 2009, Gomez 2015, Salas 2025")
