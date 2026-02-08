# %%
import numpy as np
import bsplyne
from IGA_for_bsplyne.IGAPatch import IGAPatch

# %% 1. Setup a Cube Patch
p = [1, 1, 1]
knots = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
spline = bsplyne.BSpline(p, knots)

# Unit cube control points
x, y, z = np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")
ctrl_pts = np.array([x, y, z], dtype=float)

# Material: Steel-like
E = 210e9
nu = 0.3
patch = IGAPatch(spline, ctrl_pts, E, nu)

# %% 2. Challenge: Pure Hydrostatic Stress
# For a hydrostatic state, stress is sigma = [-p, -p, -p, 0, 0, 0]
# This corresponds to a uniform volumetric expansion/contraction.
# Let's create a displacement field: U(x, y, z) = 0.01 * [x, y, z]
# In control points:
U_vec = 0.01 * ctrl_pts.reshape(3, -1)  # Shape (3, 8)

# %% 3. Post-Processing Quantities
# Evaluation grid (parametric coordinates)
XI = spline.linspace(n_eval_per_elem=20)

# A. Compute Strain (epsilon) in Voigt: [exx, eyy, ezz, gxy, gyz, gxz]
eps = patch.epsilon(U_vec, XI)

# B. Compute Stress (sigma) using constitutive law H
sig = patch.sigma(eps)

# C. Principal Stresses and Von Mises
sig_eig = patch.sigma_eig(sig)
vm = patch.von_mises(sig_eig)

# %% 4. Verification
print(f"Mean Strain (Diagonal): {np.mean(eps[:3, :]):.4f}")
print(f"Mean Shear Strain: {np.mean(eps[3:, :]):.4f}")  # Should be 0
print(f"Max Von Mises Stress: {np.max(vm):.2f} Pa")

# In pure hydrostatic stress, Von Mises should be close to 0
if np.allclose(vm, 0, atol=1e-3):
    print("✅ Success: Pure hydrostatic state achieved (Zero Von Mises).")
else:
    print(f"❌ Verification failed: Max VM is {np.max(vm):.2e}")

# %% 5. ParaView Export
# n_eval_per_elem=20 creates a 20x20x20 grid per element for visualization
patch.save_paraview(U_vec, path="./06_out", name="hydrostatic_test", n_eval_per_elem=20)
print("\nFiles 'hydrostatic_test.pvd' generated. Open in ParaView to visualize.")

# %%
