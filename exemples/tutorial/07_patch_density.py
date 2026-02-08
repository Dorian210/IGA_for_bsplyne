# %%
import numpy as np
import bsplyne
from IGA_for_bsplyne.IGAPatch import IGAPatch, IGAPatchDensity

# %% 1. Setup Geometry
p = [1, 1, 1]
knots = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
spline = bsplyne.BSpline(p, knots)

# 2x2x2 grid of control points (8 total)
x, y, z = np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")
ctrl_pts = np.array([x, y, z])

# %% 2. Define a Non-Uniform Density Field
# Density is defined at each control point.
# Let's create a gradient: density = 1.0 at x=0, and 0.0 at x=1
# We reshape it to match the control point grid (2, 2, 2)
d_field = np.array(
    [
        [[1.0, 1.0], [1.0, 1.0]],  # x = 0 plane (Full material)
        [[0.0, 0.0], [0.0, 0.0]],  # x = 1 plane (Void)
    ]
)

# %% 3. Create Patches: Standard vs Density-Weighted
E, nu = 210e9, 0.3
patch_std = IGAPatch(spline, ctrl_pts, E, nu)
patch_den = IGAPatchDensity(spline, ctrl_pts, E, nu, d=d_field)

# %% 4. Comparison: Stiffness Magnitude
K_std = patch_std.stiffness()
K_den = patch_den.stiffness()

ratio = np.linalg.norm(K_den.A) / np.linalg.norm(K_std.A)
print(f"Norm of the standard Stiffness matrix {np.linalg.norm(K_std.A):.3E}")
print(f"Norm of the density weighted Stiffness matrix {np.linalg.norm(K_den.A):.3E}")
# Since half the cube has 0 density, we expect a significant reduction.

# %% 5. Post-processing with Density
# Let's apply a uniform displacement and check stress
U_test = 0.01 * ctrl_pts.reshape(3, -1)
XI = spline.linspace(n_eval_per_elem=1)

# In IGAPatchDensity, stress is scaled by local density: sigma = d * H * eps
eps = patch_den.epsilon(U_test, XI)
den_at_pts = patch_den.density(XI)  # Interpolated density [0, 1]
sig_den = patch_den.sigma(eps, den_at_pts)
sig_eig_den = patch_den.sigma_eig(sig_den)
vm_den = patch_den.von_mises(sig_eig_den)

# %% 6. Challenge: The "Void" Effect
# Check stress at the x=1 face (where density is 0)
# Evaluation points are flattened, x=1 is at the end of the range
vm_at_void = vm_den[-1]
print(f"\nStress in the void zone (x=1): {np.linalg.norm(vm_at_void):.2e} Pa")

if np.allclose(vm_at_void, 0):
    print("✅ Success: Density 0 correctly zeroed out the stress.")
else:
    print("❌ Error: Stress still exists in void zone.")

# %% 7. ParaView Export
patch_den.save_paraview(U_test, path="./07_out", name="density_test")
print("\nExported 'density_test.pvd'. You can now visualize the 'density' field.")

# %%
