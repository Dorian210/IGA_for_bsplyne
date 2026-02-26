# %%
import numpy as np
import bsplyne
from IGA_for_bsplyne.IGAPatch import IGAPatch

# %% 1. Setup Geometry (A Simple Cube)
# Create a 3D B-spline volume: degree 2 in all directions
p = [2, 2, 2]
# Knot vectors for a unit cube [0, 1]^3
knots = [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
spline = bsplyne.BSpline(p, knots)

# Define 3x3x3 = 27 Control Points (Physical coordinates)
# Here, a simple 1.0 x 1.0 x 1.0 cube
x_grid, y_grid, z_grid = np.meshgrid(
    np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.linspace(0, 1, 3), indexing="ij"
)
ctrl_pts = np.array([x_grid, y_grid, z_grid])  # Shape (3, 3, 3, 3)

# %% 2. Define Material & Forces
E = 210e9  # Young's Modulus (Steel-like)
nu = 0.3  # Poisson's ratio

# Define a Neumann force (Pressure) on a face
# F_N shape: (3 axes, 2 sides, 3 components)
# Let's apply a force of 1000N in the -Z direction on the top face (zeta=1)
F_N = np.zeros((3, 2, 3))
F_N[2, 1, 2] = -1000.0  # Axis 2 (zeta), Side 1 (top), Component 2 (Z)

# %% 3. Create the IGAPatch
patch = IGAPatch(spline, ctrl_pts, E, nu, F_N=F_N)

# %% 4. Compute Operators
K = patch.stiffness()
f = patch.rhs()

print(f"Stiffness Matrix K: {K.shape[0]}x{K.shape[1]}")
print(f"RHS Vector f: {f.shape[0]} elements")

# %% 5. Challenge: Verification
# A. Symmetry check: K should be equal to its transpose
is_symmetric = np.linalg.norm(K.A.T - K.A) < 1e-14 * np.linalg.norm(K.A)
print(f"\nIs K symmetric? {'✅ Yes' if is_symmetric else '❌ No'}")

# B. Magnitude Check
# The sum of forces in the RHS should equal the total applied force
# Area of top face is 1.0 * 1.0 = 1.0. Applied force is -1000.
total_f_z = np.sum(f[2::3])  # Sum every 3rd element (the Z components)
print(f"Total Vertical Force in f: {total_f_z:.2f} N")

if np.isclose(total_f_z, -1000.0):
    print("✅ Force Magnitude Success")
else:
    print("❌ Force Magnitude Error (Check integration)")
# %%
