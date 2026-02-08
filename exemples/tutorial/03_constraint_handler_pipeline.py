# %%
import numpy as np
from IGA_for_bsplyne.Dirichlet import DirichletConstraintHandler

# %% 1. Initialize the Handler
# 15 physical DOFs (5 nodes with x, y, z components)
nb_physical_dofs = 15
handler = DirichletConstraintHandler(nb_physical_dofs)

# %% 2. Fixed Boundary Conditions
# Lock Node 0 (DOFs 0, 5, 10) to zero
handler.add_eqs_from_inds_vals(np.array([0, 5, 10]), vals=np.zeros(3))

# %% 3. Rigid Body Constraint
# Define a rigid connection for Nodes 1 and 2 around the origin
ref_point = np.array([0.0, 0.0, 0.0])

# Map components to slave indices
slave_inds = np.array(
    [
        [1, 2],  # x-components
        [6, 7],  # y-components
        [11, 12],  # z-components
    ]
)

# Coordinates of slave nodes
slave_pos = np.array(
    [
        [1.0, 0.0],  # x
        [0.0, 1.0],  # y
        [0.0, 0.0],  # z
    ]
)

handler.add_rigid_body_constraint(ref_point, slave_inds, slave_pos)

# %% 4. Fully Constrain Reference DOFs
# Convention: [rx, ry, rz, tx, ty, tz]
# Reference DOFs start at index 15. All must be constrained to
# ensure the particular solution 'k' is well-defined.
ref_indices = np.arange(nb_physical_dofs, nb_physical_dofs + 6)
ref_values = np.array(
    [
        0.0,
        0.0,
        0.0,  # Rotations = 0
        5.0,  # Translation X = 5.0
        0.0,
        0.0,  # Translations Y, Z = 0
    ]
)

handler.add_eqs_from_inds_vals(inds=ref_indices, vals=ref_values)

# %% 5. Finalize the Mapping
# Solves for C (nullspace) and k (particular solution)
dirichlet = handler.create_dirichlet()

print(f"Total DOFs (Phys + Ref): {handler.lhs.shape[1]}")
print(f"Reduced DOFs (Free): {dirichlet.C.shape[1]}")

# %% 6. Verification
# Testing the affine offset 'k' by passing zero reduced DOFs
u_full = dirichlet.u(np.zeros(dirichlet.C.shape[1]))

print(f"\nDisplacement check (Expect 5.0):")
print(f"Slave Node 1 (X-comp): {u_full[1]:.1f}")
print(f"Slave Node 2 (X-comp): {u_full[2]:.1f}")

# %%
