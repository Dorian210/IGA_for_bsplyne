# %% Step 1 : Make the geometry by fitting the exact one with a B-spline patch
import os
import numpy as np
import scipy.sparse as sps
from sksparse.cholmod import cholesky
import pyvista as pv
from bsplyne import BSpline

out_folder = os.path.join(os.path.dirname(__file__), "out_IGA_pinched_cylinder")

# Define the B-spline basis for the fitting of the shape
knots = [np.array([0, 1])]*3
degrees = [0]*3
spline = BSpline(degrees, knots)
spline.orderElevation(np.zeros([1] + [b.n + 1 for b in spline.bases]), [4, 4, 4])
spline.knotInsertion(np.zeros([1] + [b.n + 1 for b in spline.bases]), [7, 7, 0])

# Discretize the isoparametric space and compute the operators at these points
(xi, eta, zeta), (dxi, deta, dzeta) = spline.gauss_legendre_for_integration([np.sum(spline.getDegrees()) + 1]*3)
weights = np.outer(np.outer(dxi, deta).ravel(), dzeta).ravel()
N = spline.DN([xi, eta, zeta])
N_weighted_T = N.multiply(weights[:, None]).T
H = N_weighted_T@N
ctrl_shape = [b.n + 1 for b in spline.bases]
ctrl_size = np.prod(ctrl_shape)
nodes_inds = np.arange(ctrl_size).reshape(ctrl_shape)

# Create the exact geometry (CF geometry.svg)
R = 300
L = 600
h = 3
Xi, Eta, Zeta = np.meshgrid(xi, eta, zeta, indexing='ij', sparse=True)
Xi, Eta, Zeta = Xi[None], Eta[None], Zeta[None]
e_x, e_y, e_z = np.eye(3)[:, :, None, None, None]
points_on_C = Eta*L/2*e_x + (R - h/2 + Zeta*h)*(np.cos(Xi*np.pi/2)*e_y + np.sin(Xi*np.pi/2)*e_z)

# Solve for x component of the control points (with BC x[:, 0, :]=0 and x[:, -1, :]=L/2)
# Without BC : x = cholesky(N.T@N)(N.T@points_on_C[0].ravel())
rows = nodes_inds[:, 1:-1, :].ravel()
cols = nodes_inds[:, 1:-1, :].ravel()
_, cols = np.unique(cols, return_inverse=True)
data = np.ones(rows.size)
n_cols = cols.max() + 1
C = sps.coo_matrix((data, (rows, cols)), shape=(ctrl_size, n_cols))
k = np.zeros(ctrl_size)
k[nodes_inds[:, -1, :].flat] = L/2
x = C@cholesky((C.T@H@C))(C.T@N_weighted_T@(points_on_C[0].ravel() - N@k)) + k

# Solve for y component of the control points (with BC y[0, :, :]=y[1, :, :] and y[-1, :, :]=0)
# Without BC : y = cholesky(N.T@N)(N.T@points_on_C[1].ravel())
rows = nodes_inds[:-1, :, :].ravel()
cols = nodes_inds.copy()
cols[0, :, :] = cols[1, :, :]
cols = cols[:-1, :, :]
_, cols = np.unique(cols.ravel(), return_inverse=True)
data = np.ones(rows.size)
n_cols = cols.max() + 1
C = sps.coo_matrix((data, (rows, cols)), shape=(ctrl_size, n_cols))
y = C@cholesky((C.T@H@C))(C.T@N_weighted_T@points_on_C[1].ravel())

# Solve for z component of the control points (with BC z[0, :, :]=0 and z[-1, :, :]=z[-2, :, :])
# Without BC : z = cholesky(N.T@N)(N.T@points_on_C[2].ravel())
rows = nodes_inds[1:, :, :].ravel()
cols = nodes_inds.copy()
cols[-1, :, :] = cols[-2, :, :]
cols = cols[1:, :, :]
_, cols = np.unique(cols.ravel(), return_inverse=True)
data = np.ones(rows.size)
n_cols = cols.max() + 1
C = sps.coo_matrix((data, (rows, cols)), shape=(ctrl_size, n_cols))
z = C@cholesky((C.T@H@C))(C.T@N_weighted_T@points_on_C[2].ravel())

# Stack the results
ctrl_pts = np.array([x, y, z]).reshape((3, *ctrl_shape))

# Compute relative error, save and plot
err_rel = np.linalg.norm(points_on_C - spline(ctrl_pts, [xi, eta, zeta]))/np.linalg.norm(points_on_C)
print(f"Erreur relative de {err_rel*100:.1E}%")
spline.saveParaview(ctrl_pts, out_folder, "fitted_patch")

plotter = pv.Plotter()
ctrl_mesh, elem_borders, interior = pv.get_reader(os.path.join(out_folder, "fitted_patch.pvd")).read()
plotter.add_mesh(ctrl_mesh, color="orange", label="Control mesh", show_edges=False)
plotter.add_mesh(elem_borders, color="blue", label="Element borders", show_edges=False)
interior = pv.StructuredGrid(*spline(ctrl_pts, [xi, eta, zeta]))
plotter.add_mesh(interior, color="red", label="Interpolated geometry", show_edges=False)
obj_grid = pv.StructuredGrid(*points_on_C)
plotter.add_mesh(obj_grid, color="green", label="Objective geometry", show_edges=False)
plotter.add_legend()
plotter.show()

# %%
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from IGA_for_bsplyne.IGAPatch import IGAPatch


E, nu = 3e6, 0.3
F = 0.25

patch = IGAPatch(spline, ctrl_pts, E, nu)
K = patch.stiffness()
f = np.zeros(ctrl_pts.size)
f[-1] = -F

coords_inds = np.arange(ctrl_pts.size).reshape(ctrl_pts.shape)
colsx, colsy, colsz = coords_inds.copy()
# C1 BC
colsx[:, -1, :] = colsx[:, -2, :]
colsy[-1, :, :] = colsy[-2, :, :]
colsz[0, :, :] = colsz[1, :, :]
# Null displacement BC
colsx = colsx[:, :-1, :]
colsy = colsy[:-1, 1:, :]
colsz = colsz[1:, 1:, :]
cols = np.hstack((colsx.ravel(), colsy.ravel(), colsz.ravel()))
_, cols = np.unique(cols, return_inverse=True)
rows = np.hstack((coords_inds[0, :, :-1, :].ravel(),
                  coords_inds[1, :-1, 1:, :].ravel(),
                  coords_inds[2, 1:, 1:, :].ravel()))
data = np.ones(rows.size)
n_cols = cols.max() + 1
C = sps.coo_matrix((data, (rows, cols)), shape=(ctrl_pts.size, n_cols))
# Solving the problem
u = C@cholesky((C.T@K@C))(C.T@f)
# Saving the results
patch.save_paraview(u.reshape((3, -1)), out_folder, "IGA_patch")
print(f"Reference displacement : {-u[-1]:.4E}")
ref_displ = 1.8248e-5
print(f"Relative error of the reference displacement : {abs((-u[-1] - ref_displ)/ref_displ * 100):.2E}%")

# %%
from bsplyne import MultiPatchBSplineConnectivity
from IGA_for_bsplyne.ProblemIGA import ProblemIGA
from IGA_for_bsplyne.Dirichlet import Dirichlet

splines = [spline]*8
separated_ctrl_pts = [ctrl_pts*np.array([[[[ 1]]], [[[ 1]]], [[[ 1]]]]), 
                      ctrl_pts*np.array([[[[ 1]]], [[[-1]]], [[[ 1]]]]), 
                      ctrl_pts*np.array([[[[ 1]]], [[[-1]]], [[[-1]]]]), 
                      ctrl_pts*np.array([[[[ 1]]], [[[ 1]]], [[[-1]]]]), 
                      ctrl_pts*np.array([[[[ 1]]], [[[ 1]]], [[[ 1]]]]) + np.array([[[[L/2]]], [[[0]]], [[[0]]]]), 
                      ctrl_pts*np.array([[[[ 1]]], [[[-1]]], [[[ 1]]]]) + np.array([[[[L/2]]], [[[0]]], [[[0]]]]), 
                      ctrl_pts*np.array([[[[ 1]]], [[[-1]]], [[[-1]]]]) + np.array([[[[L/2]]], [[[0]]], [[[0]]]]), 
                      ctrl_pts*np.array([[[[ 1]]], [[[ 1]]], [[[-1]]]]) + np.array([[[[L/2]]], [[[0]]], [[[0]]]])]

connectivity, nodes_couples = MultiPatchBSplineConnectivity.from_separated_ctrlPts(separated_ctrl_pts, eps=1e-5, return_nodes_couples=True)
unique_ctrl_pts = connectivity.pack(connectivity.agglomerate(separated_ctrl_pts))

patches = [IGAPatch(spl, ctrl, E, nu) for spl, ctrl in zip(splines, separated_ctrl_pts)]

lock_x, = np.isclose(unique_ctrl_pts[0], L/2).nonzero()
lock_y, = (np.isclose(unique_ctrl_pts[0], 0) | np.isclose(unique_ctrl_pts[0], L)).nonzero()
lock_z, = (np.isclose(unique_ctrl_pts[0], 0) | np.isclose(unique_ctrl_pts[0], L)).nonzero()
inds = np.hstack((np.ravel_multi_index((0, lock_x), (3, connectivity.nb_unique_nodes)), 
                  np.ravel_multi_index((1, lock_y), (3, connectivity.nb_unique_nodes)), 
                  np.ravel_multi_index((2, lock_z), (3, connectivity.nb_unique_nodes))))
k = np.zeros(3*connectivity.nb_unique_nodes, dtype='float')
dirichlet = Dirichlet.lock_disp_inds(inds, k)

pb = ProblemIGA(patches, connectivity, dirichlet)

# Assemble lhs and rhs
lhs, rhs = pb.lhs_rhs(verbose=True)

# Modify rhs to apply "ponctual" forces
node_ind, = np.isclose(unique_ctrl_pts, np.array([L/2, 0, R + h/2])[:, None]).all(axis=0).nonzero()[0]
rhs[np.ravel_multi_index((2, node_ind), (3, connectivity.nb_unique_nodes))] = -4*F
node_ind, = np.isclose(unique_ctrl_pts, np.array([L/2, 0, -R - h/2])[:, None]).all(axis=0).nonzero()[0]
rhs[np.ravel_multi_index((2, node_ind), (3, connectivity.nb_unique_nodes))] = 4*F

# Apply Dirichlet boundary conditions
lhs, rhs = pb.apply_dirichlet(lhs, rhs, verbose=True)

# Solve the system
dof = pb.solve_from_lhs_rhs(lhs, rhs, iterative_solve=False, verbose=True)

# Cancel Dirichlet subspace
u = pb.dirichlet.u(dof).reshape((3, -1))

print(f"Reference displacement : {u[2, node_ind]:.4E}")
ref_displ = 1.8248e-5
print(f"Relative error of the reference displacement : {abs((u[2, node_ind] - ref_displ)/ref_displ * 100):.2E}%")

# Save to Paraview
pb.save_paraview(u, out_folder, "IGA_multipatch")

# %%
