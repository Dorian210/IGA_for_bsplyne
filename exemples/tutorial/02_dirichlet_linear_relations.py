# %%
import numpy as np
from IGA_for_bsplyne.Dirichlet import Dirichlet

# %% 1. Setup
# 6 DOFs system (e.g., 2 nodes with 3 displacement components each)
nb_dofs = 6
mapping = Dirichlet.eye(nb_dofs)

print(f"Initial free DOFs: {mapping.C.shape[1]}")

# %% 2. Slave / Reference Relation
# Let's say we want to link DOF 3 to DOFs 1 and 2.
# Equation: u[3] = 0.5 * u[1] + 0.5 * u[2]  (Average/Interpolation)

slaves = np.array([3])
references = np.array([[1, 2]])
coefficients = np.array([[0.5, 0.5]])

mapping.slave_reference_linear_relation(slaves, references, coefficients)

print(f"After linear relation: {mapping.C.shape[1]} free DOFs")
# DOF 3 is now "gone" from the reduced vector 'dof',
# its value is automatically driven by 1 and 2.

# %% 3. Testing the dependence
# We provide 5 values for the reduced 'dof' vector
d_input = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
u_phys = mapping.u(d_input)

print(f"\nReduced DOFs input: {d_input}")
print(f"Physical DOFs result: {u_phys}")
# Check: u_phys[3] should be (u_phys[1] + u_phys[2]) / 2
# Since 1 and 2 are free, they take values from the input.
expected_val = (u_phys[1] + u_phys[2]) / 2
print(f"Is u[3] ({u_phys[3]}) the average? {np.isclose(u_phys[3], expected_val)}")

# %% 4. Challenge: Plane Symmetry
# To enforce symmetry relative to a plane (e.g., plane X=0),
# the normal displacement (u_x) must be zero at the boundary.
# BUT, if we want to tie two nodes symmetrically: u_x1 = -u_x2

# Let's reset and try: u[5] = -1.0 * u[0]
sym_mapping = Dirichlet.eye(nb_dofs)
slaves_sym = np.array([5])
refs_sym = np.array([[0]])
coefs_sym = np.array([[-1.0]])

sym_mapping.slave_reference_linear_relation(slaves_sym, refs_sym, coefs_sym)

d_sym = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
u_sym = sym_mapping.u(d_sym)
print(f"\nSymmetry Test (u[5] = -u[0]):")
print(f"u[0] = {u_sym[0]}, u[5] = {u_sym[5]}")

# %% 5. Comparison with lock_disp_inds
# lock_disp_inds REMOVES the DOF and sets it to a CONSTANT.
# slave_reference REMOVES the DOF and sets it to a VARIABLE dependent on others.

# %%
