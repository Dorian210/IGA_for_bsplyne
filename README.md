# IGA_for_bsplyne

**IGA_for_bsplyne** is a Python library for 3D isogeometric analysis (IGA) applied to elasticity problems. This library builds upon the previous work from the [bsplyne](https://github.com/Dorian210/bsplyne) library and extends its capabilities for complex elasticity simulations.

> **Note:** This library is not yet available on PyPI. To install, please clone the repository and install it manually.

## Installation

Since **IGA_for_bsplyne** is not yet on PyPI, you can install it locally as follows:

```bash
git clone https://github.com/Dorian210/IGA_for_bsplyne
cd IGA_for_bsplyne
pip install -e .
```

Make sure to also install the required dependency [bsplyne](https://github.com/Dorian210/bsplyne) manually:

```bash
git clone https://github.com/Dorian210/bsplyne
cd bsplyne
pip install -e .
```

Additionally, ensure that [scikit-sparse](https://github.com/scikit-sparse/scikit-sparse) is installed (recommended installation via conda):

```bash
conda install -c conda-forge scikit-sparse
```

## Main Modules

- **Dirichlet**  
  Manages Dirichlet boundary conditions by applying an affine mapping (u = C @ dof + k).  
  *Key functions*: `Dirichlet.eye()`, `Dirichlet.lock_disp_inds()`, `set_u_inds_vals()`, `u_du_ddof()`, `u()`, `dof_lsq()`

- **IGAPatch**  
  Constructs a 3D IGA patch to compute the stiffness matrix, right-hand side vector, and other operators for elasticity problems over a B-spline volume.  
  *Key functions*: `jacobian()`, `grad_N()`, `make_W()`, `stiffness()`, `rhs()`, `epsilon()`, `sigma()`, `sigma_eig()`, `von_mises()`, `save_paraview()`

- **ProblemIGA**  
  Assembles the global system of equations, applies boundary conditions, and solves the elasticity problem across one or more patches.  
  *Key functions*: `assembly_block()`, `lhs_rhs()`, `apply_dirichlet()`, `solve_from_lhs_rhs()`, `solve()`, `save_paraview()`

## Examples

Several example scripts demonstrating the usage of **IGA_for_bsplyne** can be found in the `examples/` directory. These scripts cover different aspects of the library, including setting up boundary conditions, creating IGA patches, and solving elasticity problems.

## Documentation

The full API documentation is available in the `docs/` directory of the project or via the [online documentation portal](https://dorian210.github.io/IGA_for_bsplyne/IGA_for_bsplyne.html).

## Contributing

Contributions are welcome!  
- To report bugs or suggest improvements, please open an issue.
- For direct contributions, feel free to fork the repository and submit pull requests.

## License

This project is licensed under the [CeCILL License](LICENSE.txt).

