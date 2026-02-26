# %%
import numpy as np
import os
from copy import deepcopy
from bsplyne import BSpline, MultiPatchBSplineConnectivity
from IGA_for_bsplyne.Dirichlet import Dirichlet
from IGA_for_bsplyne.IGAPatch import IGAPatch
from IGA_for_bsplyne.ProblemIGA import ProblemIGA


# Define material properties
E, nu = 3e6, 0.3

# Define B-Spline parameters
knots = [np.array([0, 0, 1, 1])]*3
degrees = [1, 1, 1]

# Make B-spline
spline = BSpline(degrees, knots)
ctrl_pts = np.stack(np.meshgrid([0, 1], [0, 1], [0, 1], indexing='ij'))
ctrl_pts = spline.orderElevation(ctrl_pts, [2, 2, 2])
ctrl_pts = spline.knotInsertion(ctrl_pts, [2, 2, 2])

# Tile the B-spline to create the unit cube
def tile_spline_to_unit_cube(l, m, n, spline, ctrl_pts, duplicate_splines=False):
    shape = ctrl_pts.shape
    ctrl_pts = ctrl_pts.reshape((3, -1))/np.array([l, m, n], dtype='float')[:, None]
    splines = []
    separated_ctrl_pts = []
    pad = np.empty(3, dtype='float')
    for i in range(l):
        pad[0] = i/l
        for j in range(m):
            pad[1] = j/m
            for k in range(n):
                pad[2] = k/n
                if duplicate_splines:
                    splines.append(deepcopy(spline))
                else:
                    splines.append(spline)
                separated_ctrl_pts.append((ctrl_pts + pad[:, None]).reshape(shape))
    connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts(separated_ctrl_pts)
    return splines, separated_ctrl_pts, connectivity

l, m, n = 4, 4, 4
splines, separated_ctrl_pts, connectivity = tile_spline_to_unit_cube(l, m, n, spline, ctrl_pts)

# Create IGA patchs
iga_patchs = []
for spline, ctrl_pts in zip(splines, separated_ctrl_pts):
    iga_patchs.append(IGAPatch(spline, ctrl_pts, E, nu))

# Apply Dirichlet boundary conditions
dirichlet = Dirichlet.eye(3*connectivity.nb_unique_nodes)
# Lock displacement on the back side ( u_x(x=0) = 0, u_y(x=0, y=0) = 0, u_z(x=0, z=0) = 0 )
unique_ctrl_pts = connectivity.pack(connectivity.agglomerate(separated_ctrl_pts))
zero_x, = np.isclose(unique_ctrl_pts[0], 0).nonzero()
zero_x_and_y, = (np.isclose(unique_ctrl_pts[0], 0) &
                 np.isclose(unique_ctrl_pts[1], 0)).nonzero()
zero_x_and_z, = (np.isclose(unique_ctrl_pts[0], 0) &
                 np.isclose(unique_ctrl_pts[2], 0)).nonzero()
inds = np.hstack((zero_x + 0*connectivity.nb_unique_nodes,        # u_x(x=0) = 0
                  zero_x_and_y + 1*connectivity.nb_unique_nodes,  # u_y(x=0, y=0) = 0
                  zero_x_and_z + 2*connectivity.nb_unique_nodes)) # u_z(x=0, z=0) = 0
vals = np.zeros(inds.size, dtype='float')
dirichlet.set_u_inds_vals(inds, vals)
# Link the x displacements on the front side ( u_x(x=1) = u_x(x=1, y=0, z=0) )
reference = (np.isclose(unique_ctrl_pts[0], 1) &
             np.isclose(unique_ctrl_pts[1], 0.5) &
             np.isclose(unique_ctrl_pts[2], 0.5)).nonzero()[0][0]
reference_x = reference + 0*connectivity.nb_unique_nodes
slaves, = np.isclose(unique_ctrl_pts[0], 1).nonzero()
slaves = slaves[slaves!=reference]
slaves_x = slaves + 0*connectivity.nb_unique_nodes
references = np.repeat([reference_x], slaves.size)[:, None]
slaves = np.hstack((slaves_x))
dirichlet.slave_reference_linear_relation(slaves, references)

# Define the IGA problem
problem = ProblemIGA(iga_patchs, connectivity, dirichlet)

# Assemble lhs and rhs
lhs, rhs = problem.lhs_rhs(verbose=True)
rhs[reference_x] = E

# Apply Dirichlet boundary conditions
lhs, rhs = problem.apply_dirichlet(lhs, rhs, verbose=True)

# Solve the system
dof = problem.solve_from_lhs_rhs(lhs, rhs, iterative_solve=False, verbose=True)

# Cancel Dirichlet subspace
u = problem.dirichlet.u(dof).reshape((3, -1))

# Save results for visualization
problem.save_paraview(u, os.path.join(os.getcwd(), "out_unit_cube_traction"), "IGA", n_eval_per_elem=2)

# Expected result : u_x(x, y, z) = x; u_y(x, y, z) = -nu*y; u_z(x, y, z) = -nu*z; 

# %%
