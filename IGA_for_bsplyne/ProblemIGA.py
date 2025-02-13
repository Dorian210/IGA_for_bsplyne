import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from scipy.sparse.linalg import cg
from sksparse.cholmod import cholesky
from concurrent.futures import ProcessPoolExecutor
import inspect
from tqdm import tqdm
import os

from bsplyne import MultiPatchBSplineConnectivity
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
    
    def __init__(self, patches: list[IGAPatch], connectivity: MultiPatchBSplineConnectivity, dirichlet: Dirichlet):
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
    
    def assembly_block(self, block):
        """
        Process a block of patches, accumulating contributions to rhs and lhs.
        Each block has its own progress bar.
        """
        block_rhs = np.zeros(3 * self.connectivity.nb_unique_nodes, dtype='float')
        block_lhs_data = []
        block_lhs_rows = []
        block_lhs_cols = []
        separated_inds = self.connectivity.unique_field_indices((3,))

        # Progress bar for processing patches within a block
        with tqdm(total=len(block), desc=f"Assembly of Blocks {block[0]}-{block[-1]}") as block_pbar:
            for patch in block:
                inds_elem = separated_inds[patch].ravel()
                F = self.patches[patch].rhs()
                K = self.patches[patch].stiffness().tocoo()
                np.add.at(block_rhs, inds_elem, F)
                block_lhs_data.append(K.data)
                block_lhs_rows.append(inds_elem[K.row])
                block_lhs_cols.append(inds_elem[K.col])
                block_pbar.update(1)

        return block_rhs, block_lhs_data, block_lhs_rows, block_lhs_cols
    
    def lhs_rhs(self, verbose: bool=False) -> tuple[sps.spmatrix, npt.NDArray[np.float_]]:
        """
        Assemble the global left-hand side (lhs) matrix and right-hand side (rhs) vector
        for the linear system of equations.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints progress messages during the assembly process, by default False.

        Returns
        -------
        lhs, rhs : tuple[sps.spmatrix, npt.NDArray[np.float_]]
            The assembled sparse left-hand side matrix and right-hand side vector.
        """
        
        n = 3 * self.connectivity.nb_unique_nodes
        rhs: npt.NDArray[np.float_] = np.zeros(n, dtype='float')
        lhs_data = []
        lhs_rows = []
        lhs_cols = []
        
        # Split patches into blocks
        num_blocks = int(os.cpu_count()) # type: ignore
        patch_indices = np.array_split(range(self.connectivity.nb_patchs), num_blocks)

        # Parallel processing of blocks with progress bar for blocks
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(self.assembly_block, patch_indices),
                total=num_blocks,
                desc="Assembly of Blocks", 
                position=1,
            ))

        # Aggregate results from blocks
        if verbose: print("Aggregate results from blocks...", end="")
        for block_rhs, block_lhs_data, block_lhs_rows, block_lhs_cols in results:
            rhs += block_rhs
            lhs_data.extend(block_lhs_data)
            lhs_rows.extend(block_lhs_rows)
            lhs_cols.extend(block_lhs_cols)
        if verbose: print("done")

        # Assemble the sparse matrix
        if verbose: print("Assemble the sparse matrix...", end="")
        lhs = sps.coo_matrix(
            (np.hstack(lhs_data), (np.hstack(lhs_rows), np.hstack(lhs_cols))),
            shape=(n, n))
        if verbose: print("done")
        
        return lhs, rhs
    
    def apply_dirichlet(self, lhs: sps.spmatrix, rhs: npt.NDArray[np.float_], verbose: bool=False) -> tuple[sps.spmatrix, npt.NDArray[np.float_]]:
        """
        Apply Dirichlet boundary conditions to the system of equations.

        Parameters
        ----------
        lhs : sps.spmatrix
            The left-hand side sparse matrix of the system.
        rhs : npt.NDArray[np.float_]
            The right-hand side vector of the system.
        verbose : bool, optional
            If True, prints progress messages, by default False.

        Returns
        -------
        lhs, rhs : tuple[sps.spmatrix, npt.NDArray[np.float_]]
            The modified left-hand side matrix and right-hand side vector
            after applying Dirichlet boundary conditions.
        """
        
        if verbose: print("Apply Dirichlet boundary conditions...", end="")
        rhs = self.dirichlet.C.T @ (rhs - lhs @ self.dirichlet.k)
        lhs = self.dirichlet.C.T @ lhs @ self.dirichlet.C
        if verbose: print("done")
        
        return lhs, rhs
    
    def solve_from_lhs_rhs(self, lhs: sps.spmatrix, rhs: npt.NDArray[np.float_], iterative_solve: bool=False, verbose: bool=True) -> npt.NDArray[np.float_]:
        """
        Solve the linear system defined by the left-hand side (lhs) matrix and right-hand side (rhs) vector.

        Parameters
        ----------
        lhs : sps.spmatrix
            The left-hand side sparse matrix of the system.
        rhs : npt.NDArray[np.float_]
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
        dof : npt.NDArray[np.float_]
            The solution vector representing the degrees of freedom.
        """
        
        if iterative_solve:
        # Iterative solve with preconditioned conjugate gradient algorithm
            if verbose: print("Preconditioner creation...", end="")
            M = sps.diags(1/lhs.diagonal())
            if verbose: print("done")
            if verbose: print("Conjugate gradient solve...", end="")
            i = 0
            def callback(xk):
                nonlocal i
                resid = inspect.currentframe().f_back.f_locals['resid'] # type: ignore
                if verbose: print(f"iter {i}, residu = {resid:1.2E}")
                i += 1
            dof, info = cg(lhs, rhs, M=M, tol=0.01, callback=callback)
            if verbose: print("done")
        else:
            # Solve using Cholesky factorization
            if verbose: print("Cholesky factorization...", end="")
            factor = cholesky(lhs)
            if verbose: print("done")
            if verbose: print("Cholesky solve...", end="")
            dof = factor(rhs)
            if verbose: print("done")
        
        return dof

    def solve(self, iterative_solve=False) -> npt.NDArray[np.float_]:
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
        u : npt.NDArray[np.float_]
            Solution vector representing the computed displacements in packed notation.
            Shape : (3(phy), nb_unique_nodes)
        """
        
        # Assemble lhs and rhs
        lhs, rhs = self.lhs_rhs(verbose=True)

        # Apply Dirichlet boundary conditions
        lhs, rhs = self.apply_dirichlet(lhs, rhs, verbose=True)
        
        # Solve the system
        dof = self.solve_from_lhs_rhs(lhs, rhs, iterative_solve=iterative_solve, verbose=True)
        
        # Cancel Dirichlet subspace
        u = self.dirichlet.u(dof).reshape((3, -1))
        
        return u
    
    def save_paraview(self, u: npt.NDArray[np.float_], path: str, name: str, n_eval_per_elem: int=10):
        """
        Save the computed displacements and related fields to Paraview format for visualization.

        Parameters
        ----------
        u : npt.NDArray[np.float_]
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
            U = separated_u[patch].reshape((3, -1))[None]
            epsilon = lambda spline, XI, patch=patch: self.patches[patch].epsilon(separated_u[patch], XI).T[None]
            sigma = lambda spline, XI, patch=patch: self.patches[patch].sigma(separated_u[patch], XI).T[None]
            sigma_eig = lambda spline, XI, patch=patch: self.patches[patch].sigma_eig(separated_u[patch], XI).T[None]
            von_mises = lambda spline, XI, patch=patch: self.patches[patch].von_mises(separated_u[patch], XI)[None, :, None]
            fields = {"U": U, "epsilon": epsilon, "sigma": sigma, "sigma_eig": sigma_eig, "von_mises": von_mises}
            separated_ctrl_pts.append(self.patches[patch].ctrl_pts)
            separated_fields.append(fields)
            splines.append(self.patches[patch].spline)
        self.connectivity.save_paraview(splines, separated_ctrl_pts, path, name, n_step=1, n_eval_per_elem=n_eval_per_elem, separated_fields=separated_fields) # type: ignore
    