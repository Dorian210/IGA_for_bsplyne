# %%
import numpy as np
from IGA_for_bsplyne.Dirichlet import Dirichlet

# %% 1. Initial Configuration
# Let's imagine a system with 5 physical DOFs (e.g., 5 control points in 1D)
nb_dofs_init = 5

# Create an identity mapping: u = dof (no constraints)
# C = Eye(5, 5), k = [0, 0, 0, 0, 0]
mapping = Dirichlet.eye(nb_dofs_init)

print(f"Initially: {mapping.C.shape[1]} free DOFs out of {nb_dofs_init}")

# %% 2. Imposing Boundary Conditions
# We lock DOF 0 to 0.0 (homogeneous condition)
# We prescribe a value of 10.0 to DOF 4 (non-homogeneous condition)
locked_indices = np.array([0, 4])
prescribed_values = np.array([0.0, 10.0])

# set_u_inds_vals modifies the object in place
mapping.set_u_inds_vals(locked_indices, prescribed_values)

print(f"After constraints: {mapping.C.shape[1]} free DOFs")
print(f"Particular vector k (offset): {mapping.k}")

# %% 3. Testing the mapping u = C @ dof + k
# We define an arbitrary vector of reduced DOFs (size 3)
d_input = np.array([1.0, 2.0, 3.0])

# Calculate the full displacement vector u
u_physical = mapping.u(d_input)

print(f"\nInput (dof): {d_input}")
print(f"Result (u): {u_physical}")
# Expected output: [0.0, 1.0, 2.0, 3.0, 10.0]

# %% 4. Challenge: Verification of the inversion (Least Squares)
# We verify that dof_lsq(u(d)) == d
d_recovered = mapping.dof_lsq(u_physical)

print(f"\nRecovered (dof_lsq): {d_recovered}")

# Validation test
if np.allclose(d_input, d_recovered):
    print("✅ Success: dof_lsq(u(d)) == d")
else:
    print("❌ Error: The mapping is inconsistent")

# %% 5. Jacobian Extraction
# In an affine mapping, du/ddof is simply the matrix C
u_val, C_matrix = mapping.u_du_ddof(d_input)
print(f"\nIs the C matrix shape correct? {C_matrix.shape == (5, 3)}")

# %%
