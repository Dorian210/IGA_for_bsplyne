import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

class Dirichlet:
    """
    A class to handle Dirichlet boundary conditions (BC) for a problem using an affine mapping.

    The Dirichlet class provides methods to apply Dirichlet BCs by mapping degrees of freedom (dof)
    to displacements (u) using the relation `u = C @ dof + k`. It supports creating instances with
    identity mappings, locking specific displacement indices, and computing dof from displacements
    via least squares approximation.

    Attributes
    ----------
    C : sps.csc_matrix
        The matrix used in the affine mapping from `dof` to `u`.
    k : npt.NDArray[np.float_]
        The vector used in the affine mapping from `dof` to `u`.
    """
    C: sps.csc_matrix
    k: npt.NDArray[np.float_]
    
    def __init__(self, C: sps.spmatrix, k: npt.NDArray[np.float_]):
        """
        Initializes a Dirichlet instance with the given matrix and vector for affine mapping.

        Parameters
        ----------
        C : sps.spmatrix of float
            The matrix used to map degrees of freedom `dof` to displacements `u`.
        k : npt.NDArray[np.float_]
            The vector used to map degrees of freedom `dof` to displacements `u`.
        """
        self.C = C.tocsc()
        self.k = k
    
    @classmethod
    def eye(cls, size: int):
        """
        Create a `Dirichlet` instance with an identity mapping, where no degrees of freedom `dof` are locked.
        Sets `C` to the identity matrix and `k` to a zero-filled vector.

        Parameters
        ----------
        size : int
            Size of the `dof` and `u` vectors.

        Returns
        -------
        dirichlet : Dirichlet
            The identity `Dirichlet` instance.
        """
        C = sps.eye(size)
        k = np.zeros(size, dtype='float')
        return cls(C, k)
    
    @classmethod
    def lock_disp_inds(cls, inds: npt.NDArray[np.int_], k: npt.NDArray[np.float_]):
        """
        Creates a `Dirichlet` instance with specified displacement `u` indices locked to given values.

        Parameters
        ----------
        inds : npt.NDArray[np.int_]
            Indices of the displacement field `u` to be locked.
        k : npt.NDArray[np.float_]
            Values to lock the specified indices of `u` to.
            After this, `u[inds]` are set to `k[inds]` while other compontents are left free.

        Returns
        -------
        dirichlet : Dirichlet
            A `Dirichlet` instance with specified displacements locked.
        """
        C = sps.eye(k.size)
        dirichlet = cls(C, k.copy())
        dirichlet.set_u_inds_vals(inds, dirichlet.k[inds])
        return dirichlet
    
    def set_u_inds_vals(self, inds: npt.NDArray[np.int_], vals: npt.NDArray[np.float_]):
        """
        Locks specified indices of the displacement field `u` to given values by modifying
        the matrix `C` and vector `k` accordingly. This involves zeroing out the specified
        rows in `C` and adjusting `k` to reflect the locked values.

        Parameters
        ----------
        inds : npt.NDArray[np.int_]
            Indices of the displacement field `u` to be locked.
        vals : npt.NDArray[np.float_]
            Values to lock the specified indices of `u` to.
        """
        
        def zero_rows(C: sps.coo_matrix, rows: npt.NDArray[np.int_]):
            rm = np.hstack([(C.row==row).nonzero()[0] for row in rows])
            row = np.delete(C.row, rm)
            col = np.delete(C.col, rm)
            data = np.delete(C.data, rm)
            C = sps.coo_matrix((data, (row, col)), shape=(C.shape))
            return C

        def remove_zero_cols(C: sps.coo_matrix):
            unique, inverse = np.unique(C.col, return_inverse=True)
            C.col = np.arange(unique.size)[inverse]
            C._shape = (C.shape[0], unique.size)
        
        if inds.size!=0:
            C_coo = zero_rows(self.C.tocoo(), inds)
            remove_zero_cols(C_coo)
            self.C = C_coo.tocsc()
            self.k[inds] = vals
    
    def u_du_ddof(self, dof: npt.NDArray[np.float_]) -> tuple[npt.NDArray[np.float_], sps.csc_matrix]:
        """
        Computes the displacement field `u` and its derivative with respect to the degrees of freedom `dof`.
        The displacement field is calculated as `u = C @ dof + k`, and its derivative is `C`.

        Parameters
        ----------
        dof : npt.NDArray[np.float_]
            The degrees of freedom of the problem, representing the input to the affine mapping.

        Returns
        -------
        u, du_ddof : tuple[npt.NDArray[np.float_], sps.csc_matrix]
            A tuple containing the displacement field `u` and its derivative with respect to `dof`.
        """
        u = self.C @ dof + self.k
        du_ddof = self.C
        return u, du_ddof
    
    def u(self, dof: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """
        Computes the displacement field `u` using the affine mapping `u = C @ dof + k`.

        Parameters
        ----------
        dof : npt.NDArray[np.float_]
            The degrees of freedom of the problem, representing the input to the affine mapping.

        Returns
        -------
        u : npt.NDArray[np.float_]
            The computed displacement field `u`.
        """
        if dof.ndim==1:
            u = self.C @ dof + self.k
        else:
            m, n = self.C.shape
            dof_shape = dof.shape[:-1]
            u = (self.C @ dof.reshape((-1, n)).T + self.k.reshape((m, 1))).T.reshape((*dof_shape, m))
        return u
    
    def dof_lsq(self, u: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """
        Computes the degrees of freedom `dof` from the displacement field `u` using a least squares approximation.
        This method performs a least squares 'inversion' of the affine mapping `u = C @ dof + k`.
        It solves the linear problem `C.T @ C @ dof = C.T @ (u - k)` for `dof`.

        Parameters
        ----------
        u : npt.NDArray[np.float_]
            The displacement field from which to compute the degrees of freedom.

        Returns
        -------
        dof : npt.NDArray[np.float_]
            The computed degrees of freedom corresponding to the given displacement field.
        """
        if u.ndim==1:
            dof = spsolve(self.C.T@self.C, self.C.T@(u - self.k))
        else:
            m, n = self.C.shape
            u_shape = u.shape[:-1]
            dof = spsolve(self.C.T@self.C, (u.reshape((-1, m)) - self.k[None])@self.C).reshape((*u_shape, n))
        return dof