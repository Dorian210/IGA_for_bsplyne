# %%
import numpy as np
import os
from bsplyne import BSpline, MultiPatchBSplineConnectivity
from IGA_for_bsplyne.Dirichlet import Dirichlet
from IGA_for_bsplyne.IGAPatch import IGAPatch
from IGA_for_bsplyne.ProblemIGA import ProblemIGA

# Define B-Spline parameters
knots = [np.array([0, 0, 1, 1])]*3
degrees = [1, 1, 1]

# Define material properties
E, nu = 3e6, 0.3

# Create first IGA patch
spline1 = BSpline(degrees, knots)
ctrl_pts1 = np.stack(np.meshgrid([0, 1], [0, 1], [0, 1], indexing='ij'))
ctrl_pts1 = spline1.orderElevation(ctrl_pts1, [2, 2, 2])
ctrl_pts1 = spline1.knotInsertion(ctrl_pts1, [2, 2, 2])
patch1 = IGAPatch(spline1, ctrl_pts1, E, nu)

# Create second IGA patch
spline2 = BSpline(degrees, knots)
ctrl_pts2 = np.stack(np.meshgrid([1, 2], [0, 1], [0, 1], indexing='ij'))
ctrl_pts2 = spline2.orderElevation(ctrl_pts2, [2, 2, 2])
ctrl_pts2 = spline2.knotInsertion(ctrl_pts2, [2, 2, 2])
patch2 = IGAPatch(spline2, ctrl_pts2, E, nu, F_N=np.array([[[0, 0,  0  ],     # No force on xi back
                                                            [0, 0, -3e4]],    # Force applied on xi front
                                                           [[0, 0,  0  ],     # No force on eta back
                                                            [0, 0,  0  ]],    # No force on eta front
                                                           [[0, 0,  0  ],     # No force on zeta back
                                                            [0, 0,  0  ]]]))  # No force on zeta front

# Define multipatch connectivity
separated_ctrl_pts = [ctrl_pts1, ctrl_pts2]
connectivity = MultiPatchBSplineConnectivity.from_separated_ctrlPts(separated_ctrl_pts)

# Apply Dirichlet boundary conditions (fixing displacements on xi back of the first patch)
dof_indices = connectivity.unique_field_indices((3,))
fixed_dof_indices = dof_indices[0][:, 0, :, :].ravel() # For the first patch (index 0):
                                                       # - Lock all displacement components (x, y, z) (first :)
                                                       # - At the xi back boundary (xi index 0)
                                                       # - For all eta and zeta values (second and third :)
k = np.zeros(connectivity.nb_unique_nodes*3, dtype='float')
dirichlet = Dirichlet.lock_disp_inds(fixed_dof_indices, k)

# Define and solve the IGA problem
problem = ProblemIGA([patch1, patch2], connectivity, dirichlet)
u = problem.solve()

# Save results for visualization
problem.save_paraview(u, os.path.join(os.getcwd(), "out_IGA_beam"), "IGA")

# %%
