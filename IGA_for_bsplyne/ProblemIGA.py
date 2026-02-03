import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import cg
from sksparse.cholmod import cholesky
from concurrent.futures import ProcessPoolExecutor
import inspect
from tqdm import trange
import os
import functools

from bsplyne import MultiPatchBSplineConnectivity, parallel_blocks
from IGA_for_bsplyne.Dirichlet import Dirichlet
from IGA_for_bsplyne.IGAPatch import IGAPatch


class ProblemIGA:
    """
    ProblemIGA class to compute linear elasticity on 3D multipatch B-spline volumes.
    This class computes the stiffness matrix and the right hand side and solves the linear problem.

    Attributes
    ----------
    patches : list[IGAPatch]
        List of IGAPatch objects representing the patches for the ProblemIGA.
    connectivity : MultiPatchBSplineConnectivity
        MultiPatchBSplineConnectivity object defining the connectivity information.
    dirichlet : Dirichlet
        Dirichlet object specifying the Dirichlet boundary conditions.
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
        Initialize the ProblemIGA class with the provided patches, connectivity, and dirichlet.

        Parameters
        ----------
        patches : list[IGAPatch]
            List of IGAPatch objects representing the patches for the ProblemIGA.
        connectivity : MultiPatchBSplineConnectivity
            MultiPatchBSplineConnectivity object defining the connectivity information.
        dirichlet : Dirichlet
            Dirichlet object specifying the Dirichlet boundary conditions.
        """
        self.patches = patches
        self.connectivity = connectivity
        self.dirichlet = dirichlet

    def lhs_rhs(
        self, verbose: bool = False, disable_parallel: bool = False
    ) -> tuple[sps.spmatrix, np.ndarray[np.floating]]:
        """
        Assemble the global left-hand side (lhs) matrix and right-hand side (rhs) vector
        for the linear system of equations.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints progress messages during the assembly process, by default False.
        disable_parallel : bool, optional
            Wether to disable parallel exectution. By default, False.

        Returns
        -------
        lhs, rhs : tuple[sps.spmatrix, np.ndarray[np.floating]]
            The assembled sparse left-hand side matrix and right-hand side vector.
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
            ).ravel(),
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
        self, lhs: sps.spmatrix, rhs: np.ndarray[np.floating], verbose: bool = False
    ) -> tuple[sps.spmatrix, np.ndarray[np.floating]]:
        """
        Apply Dirichlet boundary conditions to the system of equations.

        Parameters
        ----------
        lhs : sps.spmatrix
            The left-hand side sparse matrix of the system.
        rhs : np.ndarray[np.floating]
            The right-hand side vector of the system.
        verbose : bool, optional
            If True, prints progress messages, by default False.

        Returns
        -------
        lhs, rhs : tuple[sps.spmatrix, np.ndarray[np.floating]]
            The modified left-hand side matrix and right-hand side vector
            after applying Dirichlet boundary conditions.
        """

        if verbose:
            print("Apply Dirichlet boundary conditions...", end="")
        rhs = self.dirichlet.C.T @ (rhs - lhs @ self.dirichlet.k)
        lhs = self.dirichlet.C.T @ lhs @ self.dirichlet.C
        if verbose:
            print("done")

        return lhs, rhs

    def solve_from_lhs_rhs(
        self,
        lhs: sps.spmatrix,
        rhs: np.ndarray[np.floating],
        iterative_solve: bool = False,
        verbose: bool = True,
    ) -> np.ndarray[np.floating]:
        """
        Solve the linear system defined by the left-hand side (lhs) matrix and right-hand side (rhs) vector.

        Parameters
        ----------
        lhs : sps.spmatrix
            The left-hand side sparse matrix of the system.
        rhs : np.ndarray[np.floating]
            The right-hand side vector of the system.
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
            If True, print progress messages during the solving process, by default True.

        Returns
        -------
        dof : np.ndarray[np.floating]
            The solution vector representing the degrees of freedom.
        """

        if iterative_solve:
            # Iterative solve with preconditioned conjugate gradient algorithm
            if verbose:
                print("Preconditioner creation...", end="")
            M = sps.diags(1 / lhs.diagonal())
            if verbose:
                print("done")
            if verbose:
                print("Conjugate gradient solve...", end="")
            i = 0

            def callback(xk):
                nonlocal i
                resid = inspect.currentframe().f_back.f_locals["resid"]  # type: ignore
                if verbose:
                    print(f"iter {i}, residu = {resid:1.2E}")
                i += 1

            dof, info = cg(lhs, rhs, M=M, tol=0.01, callback=callback)
            if verbose:
                print("done")
        else:
            # Solve using Cholesky factorization
            if verbose:
                print("Cholesky factorization...", end="")
            factor = cholesky(lhs)
            if verbose:
                print("done")
            if verbose:
                print("Cholesky solve...", end="")
            dof = factor(rhs)
            if verbose:
                print("done")

        return dof

    def solve(self, iterative_solve=False) -> np.ndarray[np.floating]:
        """
        Solve the linear system for the ProblemIGA class using ProcessPoolExecutor with block splitting.

        Parameters
        ----------
        iterative_solve : bool, optional
            Whether to use an iterative solver or not, default is False.
            The iterative solver is a sparse conjugate gradient with a diagonal preconditioner.
            The direct solver is a Cholesky sparse solver.
            Specs for 1_534_278 dof :
                - CG preconditioned with diagonal (`scipy.sparse.diags(1/lhs.diagonal())`):
                    solved to 1e-5 tol in 131 min
                - CG preconditioned with AMG (`pyamg.smoothed_aggregation_solver(lhs).aspreconditioner()`):
                    solved to 1e-5 tol in 225 min
                - Cholesky: solved in 32 min

        Returns
        -------
        u : np.ndarray[np.floating]
            Solution vector representing the computed displacements in packed notation.
            Shape : (3(phy), nb_unique_nodes)
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
        u: np.ndarray[np.floating],
        path: str,
        name: str,
        n_eval_per_elem: int = 10,
    ):
        """
        Save the computed displacements and related fields to Paraview format for visualization.

        Parameters
        ----------
        u : np.ndarray[np.floating]
            Displacement field in packed notation as a numpy array of shape (3(phy), nb_unique_nodes).
        path : str
            Path to save the Paraview files.
        name : str
            Name of the Paraview files.
        n_eval_per_elem : int, optional
            Number of evaluations per element, default is 10.
        """
        separated_u = self.connectivity.separate(self.connectivity.unpack(u))
        separated_ctrl_pts = []
        separated_fields = []
        splines = []
        for patch in range(self.connectivity.nb_patchs):
            U = separated_u[patch].reshape(
                (3, *self.connectivity.shape_by_patch[patch])
            )[None]
            fields = self.patches[patch].make_paraview_fields(U)
            separated_ctrl_pts.append(self.patches[patch].ctrl_pts)
            separated_fields.append(fields)
            splines.append(self.patches[patch].spline)
        self.connectivity.save_paraview(splines, separated_ctrl_pts, path, name, n_step=1, n_eval_per_elem=n_eval_per_elem, separated_fields=separated_fields)  # type: ignore
