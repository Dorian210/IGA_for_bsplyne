# IGA_for_bsplyne

<p align="center">
  <img src="https://raw.githubusercontent.com/Dorian210/IGA_for_bsplyne/main/docs/logo.png" width="500" />
</p>

**IGA_for_bsplyne** is a high-performance Python library for 3D Isogeometric Analysis (IGA) applied to linear elasticity. It bridges the gap between Computer-Aided Design (CAD) and Finite Element Analysis (FEA) by using B-Spline basis functions directly as the approximation space for structural mechanics.

This library extends the geometric foundations of [bsplyne](https://github.com/Dorian210/bsplyne) to provide a complete multipatch IGA solver pipeline.

## Installation

### 1. High-Performance Solvers (Optional but Recommended)
For large-scale 3D problems, **IGA_for_bsplyne** leverages `SuiteSparse` (via `scikit-sparse` and `sparseqr`) to achieve significant speedups. 

* **Phase A: System Libraries** Install the `SuiteSparse` headers on your operating system:
    * **macOS**: `brew install suitesparse`
    * **Ubuntu/Debian**: `sudo apt-get install libsuitesparse-dev`
    * **Windows/Conda**: `conda install -c conda-forge suitesparse`

* **Phase B: Python Wrappers** Once the system libraries are installed, you must install the Python wrappers manually:
  ```bash
  pip install scikit-sparse sparseqr
  ```

* **Fallback**: If these packages are not detected, the library automatically falls back to standard `scipy.sparse` solvers and an internal Sparse QR implementation, ensuring full compatibility at the cost of performance.

### 2. Install the Library

#### Standard Installation
Install the latest stable version directly from PyPI:
```bash
pip install IGA_for_bsplyne
```

#### Development Installation
If you wish to modify the code or contribute to the project, install the library in editable mode:
```bash
git clone https://github.com/Dorian210/IGA_for_bsplyne
cd IGA_for_bsplyne
pip install -e .
```

*Note: The core dependency [bsplyne](https://github.com/Dorian210/bsplyne) is handled automatically by pip.*

## Core Architecture

The library is organized into three specialized layers:

### 🛠️ Constraint Management: `Dirichlet`
Handles complex boundary conditions via an affine mapping approach: **u = Cd + k**.
* **Static condensation**: Automatically reduces the system size by removing constrained DOFs.
* **Advanced Relations**: Supports slave-reference dependencies and Rigid Body constraints.
* **Key tools**: `DirichletConstraintHandler` for automated multi-point constraint assembly.

### 🧊 Local Physics: `IGAPatch`
The bridge between B-Spline geometry and continuum mechanics.
* **Operators**: High-performance assembly of Stiffness matrix (**K**) and RHS vector (**f**).
* **Post-Processing**: Integrated calculation of Strains, Stresses, and Von Mises invariants.
* **Digital Twin & Immersed Models**: Supports `IGAPatchDensity` to model heterogeneous materials (e.g., from **CT-scan imagery**) by interpolating volume fractions directly into the B-Spline basis.



### 🌐 Global Solver: `ProblemIGA`
Orchestrates the multipatch environment and global resolution.
* **Connectivity**: Seamlessly "welds" patches together using `MultiPatchBSplineConnectivity`.
* **Solvers**: Choice between Direct (Cholesky) and Iterative (PCG) solvers.
* **Visualization**: Native ParaView (.pvd/.vtu) export.

## Quick Start

```python
from IGA_for_bsplyne.IGAPatch import IGAPatch
from IGA_for_bsplyne.ProblemIGA import ProblemIGA

# Define your patch
patch = IGAPatch(spline, ctrl_pts, E=70e9, nu=0.33)

# Assemble and solve
prob = ProblemIGA([patch], connectivity, dirichlet)
u = prob.solve()

# Export results
prob.save_paraview(u, path="./", name="results")
```

## Tutorials & Examples
A comprehensive hands-on guide is available in `examples/tutorial/`:
1. **Affine Mappings** (01-03)
2. **Local Integration & Post-Processing** (04-06)
3. **Immersed Models & Density** (07)
4. **Multipatch Solving** (08-09)

## Documentation
The full API documentation is available in the `docs/` directory or via the [Online Portal](https://dorian210.github.io/IGA_for_bsplyne/IGA_for_bsplyne.html).

## License
This project is licensed under the **CeCILL License**.