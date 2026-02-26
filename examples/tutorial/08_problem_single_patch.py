# %%
import numpy as np
import bsplyne
from bsplyne.multi_patch_b_spline import MultiPatchBSplineConnectivity
from IGA_for_bsplyne.IGAPatch import IGAPatch
from IGA_for_bsplyne.Dirichlet import Dirichlet
from IGA_for_bsplyne.ProblemIGA import ProblemIGA

# %% 1. Geometry Setup (Slender Beam: 5m x 1m x 1m)
p = [2, 2, 2]  # Quadratic degree
knots = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
spline = bsplyne.BSpline(p, knots)

# 3x3x3 Control Point Grid (27 points total)
x, y, z = np.meshgrid(
    np.linspace(0, 5, 3), np.linspace(0, 1, 3), np.linspace(0, 1, 3), indexing="ij"
)
ctrl_pts = np.array([x, y, z])
nb_nodes = x.size  # 27

# %% 2. Global Connectivity
# Defines the global numbering for the Degrees of Freedom (DOFs)
conn = MultiPatchBSplineConnectivity.from_separated_ctrlPts([ctrl_pts])

# %% 3. Boundary Conditions & Material Properties
# --- Dirichlet BC: Cantilever clamp at X=0 ---
# Identify the first 9 nodes (the X=0 plane)
fixed_nodes = np.arange(9)
# DOFs are stored in blocks: all X, then all Y, then all Z
fixed_dofs = np.hstack(
    (
        fixed_nodes + 0 * nb_nodes,  # Ux block
        fixed_nodes + 1 * nb_nodes,  # Uy block
        fixed_nodes + 2 * nb_nodes,  # Uz block
    )
)

dirichlet = Dirichlet.eye(3 * nb_nodes)
dirichlet.set_u_inds_vals(fixed_dofs, np.zeros(fixed_dofs.size))

# --- Material: Aluminum ---
E = 70e9
nu = 0.33

# --- Neumann BC: Tip Load at X=5 ---
# F_N: [Axis (xi=0, eta=1, zeta=2), Side (0=min, 1=max), Component (x,y,z)]
F = np.zeros((3, 2, 3))
F[0, 1, 2] = 0.001 * E  # Z-force applied on the xi=max face

patch = IGAPatch(spline, ctrl_pts, E=E, nu=nu, F_N=F)

# %% 4. Global Assembly and Solving
prob = ProblemIGA([patch], conn, dirichlet)

# Direct Solve (Cholesky Factorization)
u_direct = prob.solve(iterative_solve=False)
print(f"Direct Solve completed. Max displacement: {np.max(np.abs(u_direct)):.2e} m")

# %% 5. Solver Challenge: Direct vs. Iterative
# Iterative Solve (Preconditioned Conjugate Gradient)
u_iter = prob.solve(iterative_solve=True)

rel_diff = np.linalg.norm(u_direct - u_iter) / np.linalg.norm(u_direct)
print(f"\nRelative difference (Direct vs. Iterative): {rel_diff:.2e}")

# %% 6. Post-Processing & Export
# Generate ParaView files (.pvd / .vts)
prob.save_paraview(
    u_direct, path="./08_out", name="cantilever_beam", n_eval_per_elem=10
)
print("\nExport completed to folder './08_out'.")

# %%
