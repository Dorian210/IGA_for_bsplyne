# %%
import numpy as np
import bsplyne
from IGA_for_bsplyne.IGAPatch import IGAPatch

# %% 1. Setup a simple 1-element patch
p = [1, 1, 1]  # Linear
knots = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
spline = bsplyne.BSpline(p, knots)

# Define control points for a unit cube
x, y, z = np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")
ctrl_pts = np.array([x, y, z], dtype=float)  # Shape (3, 2, 2, 2)

# Create patch (E and nu don't matter for geometry tests)
patch = IGAPatch(spline, ctrl_pts, E=1.0, nu=0.3)

# %% 2. Inspecting the Jacobian
# We need the parametric derivatives dN/dXI to compute J
# These are usually computed internally, but let's extract them
dN_dXI = spline.DN([patch.xi, patch.eta, patch.zeta], k=1)
J, Jinv, detJ = patch.jacobian(dN_dXI)

print(f"Jacobian determinant at first int point: {detJ[0]}")
# For a unit cube, detJ should be 1.0 everywhere.
# Volume = Integral of detJ over the parametric space
W = patch.make_W(detJ)
print(f"Calculated Volume: {np.sum(W):.2f}")

# %% 3. Physical Gradients (grad_N)
# Transform parametric derivatives to physical space gradients
dN_dX = patch.grad_N(Jinv, dN_dXI)
# dN_dX[0] is a sparse matrix of shape (n_int_pts, n_ctrl_pts)
# representing dN/dx
print(f"Shape of physical gradient dN/dx: {dN_dX[0].shape}")

# %% 4. Challenge: Geometric Distortion & Area
# A. Original Area of the 'top' face (zeta=1)
area_init = patch.area_border(axis=2, front_side=True)
print(f"\nInitial Area (top face): {area_init:.2f}")

# B. Distort the geometry: Stretch the top face by 2x in X and Y
# Control points at zeta=1 are ctrl_pts[:, :, :, 1]
ctrl_pts_distorted = ctrl_pts.copy()
ctrl_pts_distorted[0, :, :, 1] *= 2.0  # Stretch X
ctrl_pts_distorted[1, :, :, 1] *= 2.0  # Stretch Y

# Update the patch geometry
patch_distorted = IGAPatch(spline, ctrl_pts_distorted, E=1.0, nu=0.3)

# C. New Area calculation
area_distorted = patch_distorted.area_border(axis=2, front_side=True)
print(f"Distorted Area (top face): {area_distorted:.2f}")

# Verification: Area of a 2x2 square should be 4.0
if np.isclose(area_distorted, 4.0):
    print("✅ Success: The integration correctly follows the geometry change!")
else:
    print(f"❌ Area mismatch: Expected 4.0, got {area_distorted}")
# %%
