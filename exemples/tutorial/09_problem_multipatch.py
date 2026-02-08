# %%
import numpy as np
import bsplyne
from bsplyne.multi_patch_b_spline import MultiPatchBSplineConnectivity
from IGA_for_bsplyne.IGAPatch import IGAPatch
from IGA_for_bsplyne.Dirichlet import Dirichlet
from IGA_for_bsplyne.ProblemIGA import ProblemIGA

# %% 1. Geometry Setup (Refined Multipatch)
p_init = [1, 1, 1]
knots_init = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
spline = bsplyne.BSpline(p_init, knots_init)

# Patch 1: unit cube
x1, y1, z1 = np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")
ctrl_pts1 = np.array([x1, y1, z1], dtype=float)

# --- Refinement ---
# Elevating to cubic (p=3) and inserting knots for 3x3x3 elements patchs
ctrl_pts1 = spline.orderElevation(ctrl_pts1, [2, 2, 2])
ctrl_pts1 = spline.knotInsertion(ctrl_pts1, [2, 2, 2])

# Patch 2: unit cube translated by (1, 0, 0) (translated copy of Patch 1)
ctrl_pts2 = ctrl_pts1.copy()
ctrl_pts2[0] += 1.0

# %% 2. Global Connectivity (The "Glue")
# Merges coincident nodes at the X=1 interface into shared Global DOFs
separated_ctrl_pts = [ctrl_pts1, ctrl_pts2]
conn = MultiPatchBSplineConnectivity.from_separated_ctrlPts(separated_ctrl_pts)

print(f"Nodes per patch: {spline.getNbFunc()}")
print(f"Total unique nodes: {conn.nb_unique_nodes}")  # Effectively 'welded'

# %% 3. Boundary Conditions & Global Assembly
# A. Get global coordinates for node identification
unique_ctrl_pts = conn.pack(conn.agglomerate(separated_ctrl_pts))

# B. Dirichlet: Clamp the base at X=0
fixed_nodes = np.where(unique_ctrl_pts[0, :] == 0.0)[0]
# DOFs are packed: [All X | All Y | All Z]
fixed_dofs = np.hstack([fixed_nodes + i * conn.nb_unique_nodes for i in range(3)])

dirichlet = Dirichlet.eye(3 * conn.nb_unique_nodes)
dirichlet.set_u_inds_vals(fixed_dofs, np.zeros(fixed_dofs.size))

# C. Initialize Problem (Material: Aluminum)
E, nu = 70e9, 0.33
p1 = IGAPatch(spline, ctrl_pts1, E, nu)
p2 = IGAPatch(spline, ctrl_pts2, E, nu)
prob = ProblemIGA([p1, p2], conn, dirichlet)

# %% 4. Manual RHS Loading
# We assemble the system and inject the force manually
lhs, rhs = prob.lhs_rhs()

# Identify nodes at the tip (X=2)
beam_end_nodes = np.where(unique_ctrl_pts[0, :] == 2.0)[0]

# Apply Z-force (Component index 2)
# Force magnitude is arbitrary for this test (0.0002 * E per node)
rhs[beam_end_nodes + 2 * conn.nb_unique_nodes] = 0.0002 * E

# %% 5. Solve
# Reduce the system based on Dirichlet constraints
lhs_dir, rhs_dir = prob.apply_dirichlet(lhs, rhs)

# Solve for free DOFs
dof_sol = prob.solve_from_lhs_rhs(lhs_dir, rhs_dir)

# Reconstruct the full (3, N) displacement field
u_global = dirichlet.u(dof_sol).reshape((3, -1))

# %% 6. Post-Processing & Export
prob.save_paraview(u_global, path="./09_out", name="multipatch_manual_load")
print("\nExported to ./09_out. Use 'Warp By Vector' in ParaView to see results.")

# %%
