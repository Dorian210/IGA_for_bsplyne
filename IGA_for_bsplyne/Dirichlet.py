import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import numba as nb

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
    
    def slave_reference_linear_relation(self, slaves: np.ndarray[int], references: np.ndarray[int], coefs: np.ndarray[float]):
        """
        This function modifies the sparse matrix `C` and the vector `k` to enforce 
        reference-slave constraints in an optimization problem. The goal is to eliminate 
        the degrees of freedom (DOFs) associated with slave nodes, keeping only the reference 
        DOFs. The relation `u = C@dof + k` is updated so that slave DOFs are expressed as 
        linear combinations of reference DOFs, reducing the problem's size while maintaining 
        the imposed constraints.
        
        Parameters
        ----------
        slaves : np.ndarray[int]
            Array of slave indices.
        references : np.ndarray[int]
            2D array where each row contains the reference indices controlling a slave.
        coefs : np.ndarray[float]
            2D array of coefficients defining the linear relationship between references and 
            slaves.
        """
        sorted_slaves = slave_reference_linear_relation_sort(slaves, references)
        
        C = self.C.tocsr()
        rows, cols, data, self.k = slave_reference_linear_relation_inner(C.indices, C.indptr, C.data, 
                                                                         self.k, slaves, references, coefs, sorted_slaves)
        C = sps.coo_matrix((data, (rows, cols)), shape=C.shape)
        
        unique, inverse = np.unique(C.col, return_inverse=True)
        C.col = np.arange(unique.size)[inverse]
        C._shape = (C.shape[0], unique.size)
        self.C = C.tocsc()
    
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


@nb.njit(cache=True)
def slave_reference_linear_relation_sort(slaves: np.ndarray[int], references: np.ndarray[int]) -> np.ndarray[int]:
    """
    Sorts the slave nodes based on reference indices to respect 
    hierarchical dependencies (each slave is processed after its references).
    
    Parameters
    ----------
    slaves : np.ndarray[int]
        Array of slave indices.
    references : np.ndarray[int]
        2D array where each row contains the reference indices controlling a slave.
    
    Returns
    -------
    sorted_slaves : np.ndarray[int]
        Array of slave indices sorted based on dependencies.
    """
    slaves_set = set(slaves)
    graph = {s: nb.typed.List.empty_list(nb.int64) for s in slaves}
    indegree = {s: 0 for s in slaves}
    
    # Building the graph
    for i in range(len(slaves)):
        for reference in references[i]:
            if reference in slaves_set:
                indegree[reference] += 1
                graph[slaves[i]].append(reference)
    
    # Queue for slaves with no dependencies (in-degree 0)
    queue = [s for s in slaves if indegree[s]==0]
    
    # Topological sorting via BFS
    sorted_slaves = np.empty(len(slaves), dtype='int')
    i = 0
    while queue:
        slave = queue.pop(0)
        sorted_slaves[i] = slave
        i += 1
        for dependent_slave in graph[slave]:
            indegree[dependent_slave] -= 1
            if indegree[dependent_slave]==0:
                queue.append(dependent_slave)
    
    if i != len(slaves):
        raise ValueError("Cyclic dependency detected in slaves.")
    
    sorted_slaves = sorted_slaves[::-1]
    return sorted_slaves

@nb.njit(cache=True)
def slave_reference_linear_relation_inner(
    indices: np.ndarray[int], 
    indptr: np.ndarray[int], 
    data: np.ndarray[float], 
    k: np.ndarray[float], 
    slaves: np.ndarray[int], 
    references: np.ndarray[int], 
    coefs: np.ndarray[float], 
    sorted_slaves: np.ndarray[int]
) -> tuple[np.ndarray[int], np.ndarray[int], np.ndarray[float], np.ndarray[float]]:
    """
    Applies slave-reference relations directly to CSR matrix arrays.
    
    Parameters
    ----------
    indices : np.ndarray[int]
        Column indices of CSR matrix.
    indptr : np.ndarray[int]
        Row pointers of CSR matrix.
    data : np.ndarray[float]
        Nonzero values of CSR matrix.
    k : np.ndarray[float]
        Vector to be updated.
    slaves : np.ndarray[int]
        Array of slave indices.
    references : np.ndarray[int]
        2D array where each row contains the reference indices controlling a slave.
    coefs : np.ndarray[float]
        2D array of coefficients defining the linear relationship between references and 
        slaves.
    sorted_slaves : np.ndarray[int]
        Array of slave indices sorted in topological order.

    Returns
    -------
    rows : np.ndarray[int]
        Updated row indices of COO matrix.
    cols : np.ndarray[int]
        Updated column indices of COO matrix.
    data : np.ndarray[float]
        Updated nonzero values of COO matrix.
    k : np.ndarray[float]
        Updated vector.
    """

    # Convert CSR to list of dicts
    dict_list = [{indices[j]: data[j] for j in range(indptr[i], indptr[i + 1])} for i in range(indptr.size - 1)]
    
    # Apply linear relation
    slaves_to_index = {s: i for i, s in enumerate(slaves)}
    for slave in sorted_slaves:
        i = slaves_to_index[slave]
        new_row = {}
        for reference_ind, coef in zip(references[i], coefs[i]):
            reference_row = dict_list[reference_ind]
            for ind, val in reference_row.items():
                if ind in new_row:
                    new_row[ind] += coef*val
                else:
                    new_row[ind] = coef*val
        new_row = {ind: val for ind, val in new_row.items() if val!=0}
        dict_list[slave] = new_row
        k[slave] = np.dot(k[references[i]], coefs[i])
    
    # Convert list of dicts to COO
    nnz = sum([len(row) for row in dict_list])
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    pos = 0
    for i, row in enumerate(dict_list):
        for j, val in row.items():
            rows[pos] = i
            cols[pos] = j
            data[pos] = val
            pos += 1

    return rows, cols, data, k