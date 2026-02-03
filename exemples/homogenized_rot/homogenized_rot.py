# %% Mesh geometry
import numpy as np
from bsplyne import new_cube, MultiPatchBSplineConnectivity

spline, ctrl_pts = new_cube([0, 0, 0.5], [0, 0, 1], 1)
# ctrl_pts[2] *= 10
print(f"System defined by the rectangular cuboid:\n[{ctrl_pts[0].min()}, {ctrl_pts[0].max()}]x[{ctrl_pts[1].min()}, {ctrl_pts[1].max()}]x[{ctrl_pts[2].min()}, {ctrl_pts[2].max()}]", end ="\n\n")

ctrl_pts = spline.orderElevation(ctrl_pts, [2, 2, 2])
ctrl_pts = spline.knotInsertion(ctrl_pts, [4, 4, 9])
print(f"System discretized with a B-spline patch of:\n  - {spline.getDegrees().tolist()} degrees\n  - {[b.n - b.p + 1 for b in spline.bases]} elements", end ="\n\n")

connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts([ctrl_pts])

spline.saveParaview(ctrl_pts, "out_homogenized_rot", "B-spline_patch", n_eval_per_elem=2, verbose=False)

# %% Create boundary conditions handler
from IGA_for_bsplyne.Dirichlet import DirichletConstraintHandler

nb_nodes = spline.getNbFunc()
constraints = DirichletConstraintHandler(3*nb_nodes)

# Clamped BC on the bottom boundary
bottom_nodes = np.isclose(ctrl_pts[2], 0).ravel().nonzero()[0]
bottom_inds = np.hstack((bottom_nodes, 
                         bottom_nodes + nb_nodes, 
                         bottom_nodes + 2*nb_nodes))
constraints.add_eqs_from_inds_vals(bottom_inds)

# Create a reference point coupling on the top boundary
top_ref_point = np.array([0, 0, 1]) # Set the center of the top face as the ref point
top_nodes = np.isclose(ctrl_pts[2], ctrl_pts[2].max()).ravel().nonzero()[0]
top_coords = ctrl_pts.reshape((3, -1))[:, top_nodes]
top_inds = np.vstack((top_nodes, top_nodes + nb_nodes, top_nodes + 2*nb_nodes))
constraints.add_rigid_body_constraint(top_ref_point, top_inds, top_coords)

# Save constraints state before imposing the reference point's degrees of freedom
constraints_saved = constraints.copy()

theta = np.array([0, np.pi/16, 0]) # Rotate around e_y
t = np.array([0, 0, 1.]) # translate along e_z
ref_0 = np.hstack((theta, t))
constraints.add_eqs_from_inds_vals(constraints.nb_dofs_init + np.arange(6), ref_0)

# %% Make elasticity problem
from IGA_for_bsplyne.IGAPatch import IGAPatch
from IGA_for_bsplyne.ProblemIGA import ProblemIGA

E, nu = 3e6, 0.3
F = 0

patches = [IGAPatch(spline, ctrl_pts, E, nu)]
dirichlet = constraints.create_dirichlet()
pb = ProblemIGA(patches, connectivity, dirichlet)

# %% Compute operators
K, f = pb.lhs_rhs(verbose=True)
f[-1] = F

# %% Solve the elasticity problem
lhs, rhs = pb.apply_dirichlet(K, f, verbose=True)
dof = pb.solve_from_lhs_rhs(lhs, rhs)
u = pb.dirichlet.u(dof)

# A posteriori extract the lagrange multipliers
lamb = constraints_saved.get_ref_multipliers_from_internal_residual(K@u - f)

moments_top = lamb[0:3]
efforts_top = lamb[3:6]

print("moments_top:", moments_top)
print("efforts_top:", efforts_top)

# %%
pb.save_paraview(u.reshape((3, -1)), "out_homogenized_rot", "IGA_results", n_eval_per_elem=2)
# %%







# %% Make elasticity problem
from IGA_for_bsplyne.IGAPatch import IGAPatch
from IGA_for_bsplyne.ProblemIGA import ProblemIGA
from IGA_for_bsplyne.Dirichlet import Dirichlet

E, nu = 3e6, 0.3
F = 0

patches = [IGAPatch(spline, ctrl_pts, E, nu)]
dirichlet = Dirichlet.eye(3*connectivity.nb_unique_nodes)
pb = ProblemIGA(patches, connectivity, dirichlet)

# %% Compute operators
K, f = pb.lhs_rhs(verbose=True)
f[-1] = F

# %% Solve the elasticity problem
import scipy.sparse as sps

C, k = constraints_saved.make_C_k()
C_ref = C[constraints.nb_dofs_init:, :]
k_ref = k[constraints.nb_dofs_init:]
C_u = C[:constraints.nb_dofs_init, :]
k_u = k[:constraints.nb_dofs_init]

Kr = C_u.T@K@C_u
Fr = C_u.T@(f - K@k_u)

lhs = sps.bmat([[C_u.T@K@C_u, C_ref.T], 
                [C_ref, None]])
rhs = np.hstack((C_u.T@(f - K@k_u), ref_0 - k_ref))

dof_lamb = sps.linalg.spsolve(lhs, rhs)
dof, lamb = dof_lamb[:C.shape[1]], dof_lamb[C.shape[1]:]

u_check = C_u@dof + k_u
ref_check = C_ref@dof + k_ref

print(f"ref==ref_0 : {np.allclose(ref_check, ref_0)}")

moments_top_check = lamb[0:3]
efforts_top_check = lamb[3:6]

print("moments_top:", moments_top_check, "\nSame as before :", np.allclose(moments_top_check, moments_top)) # moments_top: [ 7.58667388e-09 -8.96043194e+04 -1.43105939e-10]
print("efforts_top:", efforts_top_check, "\nSame as before :", np.allclose(efforts_top_check, efforts_top)) # efforts_top: [ 7.62679442e+04 -7.64347615e-09 -3.19640835e+06]

# %%
pb.save_paraview(u_check.reshape((3, -1)), "out_homogenized_rot", "IGA_results", n_eval_per_elem=2)
# %%
