import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sps
from scipy.sparse.linalg import cg
import inspect
from tqdm import trange

from bsplyne import MultiPatchBSplineConnectivity, parallel_blocks
from IGA_for_bsplyne.Dirichlet import Dirichlet
from IGA_for_bsplyne.IGAPatch import IGAPatch
from IGA_for_bsplyne.solvers import solve_sparse


class ProblemIGA:
    """
    `ProblemIGA` class for linear elasticity on 3D multipatch B-spline volumes.

    This class assembles the global stiffness matrix and right-hand side vector
    from multiple :py:class:`IGAPatch` or :py:class:`IGAPatchDensity` objects,
    applies Dirichlet boundary conditions, and solves the linear system to
    compute the displacement field.

    Attributes
    ----------
    patches : list[IGAPatch]
        List of patch objects representing the B-spline patches of the problem.
    connectivity : MultiPatchBSplineConnectivity
        Connectivity information linking the patches to global degrees of freedom.
    dirichlet : Dirichlet
        Object defining the Dirichlet boundary conditions applied to the system.

    Notes
    -----
    - The class handles multipatch assembly and global degrees of freedom mapping.
    - Individual patch computations (stiffness, rhs, fields) are delegated to each patch.
    - Post-processing quantities (strain, stress, von Mises) can be exported to ParaView
    using :py:meth:`save_paraview`.
    - Parallelization is used internally during assembly via `parallel_blocks`.
    """

    patches: list[IGAPatch]
    connectivity: MultiPatchBSplineConnectivity
    dirichlet: Dirichlet

    def __init__(
        self,
        patches: list[IGAPatch],
        connectivity: MultiPatchBSplineConnectivity,
        dirichlet: Dirichlet,
    ):
        """
        Initialize a `ProblemIGA` instance.

        This constructor sets up a multipatch linear elasticity problem by
        storing the patches, the connectivity mapping between them, and the
        Dirichlet boundary conditions. The object itself does not assemble
        or solve the system until :py:meth:`lhs_rhs` or :py:meth:`solve` is called.

        Parameters
        ----------
        patches : list[IGAPatch]
            List of patch objects (instances of :py:class:`IGAPatch` or its subclasses)
            defining the geometry, material, and local operators.
        connectivity : MultiPatchBSplineConnectivity
            Connectivity object that maps local patch degrees of freedom
            to global degrees of freedom for assembly.
        dirichlet : Dirichlet
            Dirichlet boundary condition object defining the constrained degrees
            of freedom and their prescribed values.
        """
        self.patches = patches
        self.connectivity = connectivity
        self.dirichlet = dirichlet

    def lhs_rhs(
        self, verbose: bool = False, disable_parallel: bool = False
    ) -> tuple[sps.spmatrix, NDArray[np.floating]]:
        """
        Assemble the global linear system for the multipatch problem.

        This method computes the global left-hand side (LHS) matrix and
        right-hand side (RHS) vector for the linear elasticity problem
        represented by the set of patches in the :py:class:`ProblemIGA`.

        The RHS is assembled from the individual patch contributions
        using the connectivity mapping. The LHS (stiffness matrix) is
        assembled by computing each patch's local stiffness matrix
        and mapping its entries to the global system.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints progress messages during assembly, by default False.
        disable_parallel : bool, optional
            If True, disables parallel execution for LHS assembly, by default False.

        Returns
        -------
        lhs : sps.spmatrix
            The assembled global sparse stiffness matrix of shape
            (3 * nb_unique_nodes, 3 * nb_unique_nodes).
        rhs : NDArray[np.floating]
            The assembled global right-hand side vector of shape
            (3 * nb_unique_nodes,).

        Notes
        -----
        - The factor 3 corresponds to the three physical displacement components
        (x, y, z) per node.
        - The patch contributions are gathered and summed according to the
        connectivity mapping defined in :py:attr:`connectivity`.
        - Parallelization is used by default for computing patch LHS and RHS blocks,
        and can be disabled via `disable_parallel`.
        """

        # Compute RHS
        separated_rhs = parallel_blocks(
            [patch.rhs for patch in self.patches],
            pbar_title="Assembling RHS",
            verbose=verbose,
            disable_parallel=True,
        )
        unpacked_rhs = np.hstack(separated_rhs)
        rhs = np.zeros(3 * self.connectivity.nb_unique_nodes, dtype="float")
        np.add.at(
            rhs,
            self.connectivity.unique_field_indices(
                (3,), representation="unpacked"
            ).ravel(),  # type: ignore
            unpacked_rhs,
        )

        # Compute elementary LHSs
        lhs_list = parallel_blocks(
            [patch.stiffness for patch in self.patches],
            pbar_title="Computing elementary LHSs",
            verbose=verbose,
            disable_parallel=disable_parallel,
        )

        # Assemble LHS from elementary matrices
        lhs_data = []
        lhs_rows = []
        lhs_cols = []
        separated_inds = self.connectivity.unique_field_indices((3,))
        for patch in trange(
            self.connectivity.nb_patchs, desc=f"Assembling LHS", disable=not verbose
        ):
            inds_elem = separated_inds[patch].ravel()
            K = lhs_list[patch].tocoo()
            lhs_data.append(K.data)
            lhs_rows.append(inds_elem[K.row])
            lhs_cols.append(inds_elem[K.col])

        # Assemble the sparse matrix
        if verbose:
            print("Format sparse matrix...", end="")
        lhs_data = np.hstack(lhs_data)
        lhs_rows = np.hstack(lhs_rows)
        lhs_cols = np.hstack(lhs_cols)
        shape = (
            3 * self.connectivity.nb_unique_nodes,
            3 * self.connectivity.nb_unique_nodes,
        )
        lhs = sps.coo_matrix((lhs_data, (lhs_rows, lhs_cols)), shape=shape)
        if verbose:
            print("done")

        return lhs, rhs

    def apply_dirichlet(
        self, lhs: sps.spmatrix, rhs: NDArray[np.floating], verbose: bool = False
    ) -> tuple[sps.spmatrix, NDArray[np.floating]]:
        """
        Apply Dirichlet boundary conditions to the global linear system.

        This method modifies the global LHS matrix and RHS vector
        to enforce prescribed Dirichlet conditions. The Dirichlet object
        (:py:attr:`dirichlet`) provides the selection matrix `C` and
        prescribed values `k` such that:

            rhs = C.T @ (rhs - lhs @ k)
            lhs = C.T @ lhs @ C

        Parameters
        ----------
        lhs : sps.spmatrix
            Global sparse stiffness matrix of shape (3*nb_unique_nodes, 3*nb_unique_nodes).
        rhs : NDArray[np.floating]
            Global right-hand side vector of shape (3*nb_unique_nodes,).
        verbose : bool, optional
            If True, prints progress messages during application of Dirichlet
            conditions, by default False.

        Returns
        -------
        lhs : sps.spmatrix
            Reduced stiffness matrix with Dirichlet conditions applied.
        rhs : NDArray[np.floating]
            Reduced RHS vector corresponding to the free degrees of freedom.

        Notes
        -----
        - The operation effectively reduces the system to the unconstrained
        degrees of freedom.
        - Use :py:meth:`solve_from_lhs_rhs` after applying Dirichlet conditions
        to obtain the solution.
        """

        if verbose:
            print("Apply Dirichlet boundary conditions...", end="")
        rhs = self.dirichlet.C.T @ (rhs - lhs @ self.dirichlet.k)  # type: ignore
        lhs = self.dirichlet.C.T @ lhs @ self.dirichlet.C
        if verbose:
            print("done")

        return lhs, rhs

    def solve_from_lhs_rhs(
        self,
        lhs: sps.spmatrix,
        rhs: NDArray[np.floating],
        iterative_solve: bool = False,
        verbose: bool = True,
    ) -> NDArray[np.floating]:
        """
        Solve a linear system for the given left-hand side matrix and right-hand side vector.

        This method solves the linear system
        `lhs @ dof = rhs` for the unknown degrees of freedom.
        It does not apply Dirichlet boundary conditions: the inputs should
        already be modified accordingly (e.g., via :py:meth:`apply_dirichlet`).

        Parameters
        ----------
        lhs : sps.spmatrix
            Sparse matrix representing the system after applying Dirichlet conditions.
            Can have arbitrary shape as long as it matches `rhs`.
        rhs : NDArray[np.floating]
            Right-hand side vector, already accounting for Dirichlet conditions.
        iterative_solve : bool, optional
            If True, use an iterative solver (conjugate gradient with diagonal preconditioner),
            otherwise use a direct solver (Cholesky factorization).
            Specs for 1_534_278 dof :
                - CG preconditioned with diagonal (`scipy.sparse.diags(1/lhs.diagonal())`):
                    solved to 1e-5 tol in 131 min
                - CG preconditioned with AMG (`pyamg.smoothed_aggregation_solver(lhs).aspreconditioner()`):
                    solved to 1e-5 tol in 225 min
                - Cholesky: solved in 32 min
            By default False.
        verbose : bool, optional
            If True, print progress messages during the solving process. Default is True.

        Returns
        -------
        dof : NDArray[np.floating]
            Solution vector representing the degrees of freedom in the
            unconstrained/reduced system.

        Notes
        -----
        - The iterative solver prints residuals at each iteration if `verbose=True`.
        - For large systems, iterative solvers can be more memory efficient, but may require
        tuning the preconditioner.
        - Use :py:meth:`apply_dirichlet` to handle Dirichlet constraints before calling this method.
        """

        if iterative_solve:
            # Iterative solve with preconditioned conjugate gradient algorithm
            if verbose:
                print("Preconditioner creation...", end="")
            M = sps.diags(1 / lhs.diagonal())  # type: ignore
            if verbose:
                print("done")
            if verbose:
                print("Conjugate gradient solve...", end="")
            i = 0

            def callback(xk):
                nonlocal i
                try:
                    resid = inspect.currentframe().f_back.f_locals["resid"]  # type: ignore
                    if verbose:
                        print(f"iter {i}, residu = {resid:1.2E}")
                except:
                    pass
                i += 1

            dof, info = cg(lhs, rhs, M=M, tol=0.01, callback=callback)
            if verbose:
                print("done")
        else:
            # Solve using Cholesky factorization
            if verbose:
                print("Cholesky solve...", end="")
            dof = solve_sparse(lhs, rhs)
            if verbose:
                print("done")

        return dof

    def solve(self, iterative_solve=False) -> NDArray[np.floating]:
        """
        Solve the linear elasticity problem for the multipatch IGA system.

        This method assembles the global system, applies Dirichlet boundary conditions,
        and solves for the displacements. The solution returned includes all degrees
        of freedom in packed notation, with Dirichlet values restored.

        Parameters
        ----------
        iterative_solve : bool, optional
            If True, solve the system using an iterative solver (preconditioned
            conjugate gradient), otherwise use a direct solver (Cholesky). Default is False.
            Exemple notes on performance for large systems (~1.5 million DOFs):
                - CG with diagonal preconditioner: 1e-5 tol in ~131 min
                - CG with AMG preconditioner: 1e-5 tol in ~225 min
                - Cholesky factorization: ~32 min

        Returns
        -------
        u : NDArray[np.floating]
            Displacement vector including all degrees of freedom, in packed notation.
            Shape: (3, nb_unique_nodes), with the first dimension corresponding
            to the physical coordinates (x, y, z).

        See Also
        --------
        lhs_rhs : assemble the global system.
        apply_dirichlet : apply Dirichlet boundary conditions.
        solve_from_lhs_rhs : solve a reduced linear system.
        """

        # Assemble lhs and rhs
        lhs, rhs = self.lhs_rhs(verbose=True)

        # Apply Dirichlet boundary conditions
        lhs, rhs = self.apply_dirichlet(lhs, rhs, verbose=True)

        # Solve the system
        dof = self.solve_from_lhs_rhs(
            lhs, rhs, iterative_solve=iterative_solve, verbose=True
        )

        # Cancel Dirichlet subspace
        u = self.dirichlet.u(dof).reshape((3, -1))

        return u

    def save_paraview(
        self,
        u: NDArray[np.floating],
        path: str,
        name: str,
        n_eval_per_elem: int = 10,
        disable_parallel: bool = False,
    ):
        """
        Export displacements and derived fields to Paraview files for visualization.

        This method separates the global displacement vector into patch-local fields,
        evaluates additional post-processing fields (strain, stress, von Mises, etc.)
        at a regular grid of points per element, and saves all data in a format readable
        by Paraview. The evaluation is done in parallel using `parallel_blocks` unless
        `disable_parallel` is True.

        Parameters
        ----------
        u : NDArray[np.floating]
            Displacement field in packed notation, including Dirichlet values.
            Shape: (3, nb_unique_nodes), where the first dimension corresponds to
            physical coordinates (x, y, z).
        path : str
            Directory path where the Paraview files will be saved.
        name : str
            Base name for the Paraview files.
        n_eval_per_elem : int, optional
            Number of evaluation points per element in each parametric direction.
            Default is 10.
        disable_parallel : bool, optional
            If True, disables parallel evaluation of patch fields. Default is False.

        Notes
        -----
        - For each patch, the method evaluates fields at `n_eval_per_elem` points along
        each parametric direction and computes derived quantities via
        `IGAPatch.make_paraview_fields`.
        - The output can include displacement, strain, stress, von Mises stress, and
        density if applicable (e.g., in `IGAPatchDensity`).
        - The `connectivity` object handles assembling the patches into a global
        representation for Paraview.

        See Also
        --------
        IGAPatch.make_paraview_fields : compute local fields for each patch.
        MultiPatchBSplineConnectivity.save_paraview : handles saving the assembled global data.
        """
        separated_u = self.connectivity.separate(self.connectivity.unpack(u))
        separated_ctrl_pts = [patch.ctrl_pts for patch in self.patches]
        splines = [patch.spline for patch in self.patches]
        XI_list = [
            patch.spline.linspace(n_eval_per_elem=n_eval_per_elem)
            for patch in self.patches
        ]
        separated_fields = parallel_blocks(
            [patch.make_paraview_fields for patch in self.patches],
            all_args=[
                (
                    separated_u[patch].reshape(
                        (3, *self.connectivity.shape_by_patch[patch])
                    )[None],
                    XI,
                )
                for patch, XI in zip(range(self.connectivity.nb_patchs), XI_list)
            ],
            pbar_title="Computing data fields",
            verbose=True,
            disable_parallel=disable_parallel,
        )
        self.connectivity.save_paraview(
            splines,
            separated_ctrl_pts,
            path,
            name,
            n_step=1,
            XI_list=XI_list,
            separated_fields=separated_fields,
            disable_parallel=disable_parallel,
        )
