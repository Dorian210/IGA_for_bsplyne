from typing import Union
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from sksparse.cholmod import cholesky
import numba as nb
import copy


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
    k : np.ndarray[np.floating]
        The vector used in the affine mapping from `dof` to `u`.
    """

    C: sps.csc_matrix
    k: np.ndarray[np.floating]

    def __init__(self, C: sps.spmatrix, k: np.ndarray[np.floating]):
        """
        Initializes a Dirichlet instance with the given matrix and vector for affine mapping.

        Parameters
        ----------
        C : sps.spmatrix of float
            The matrix used to map degrees of freedom `dof` to displacements `u`.
        k : np.ndarray[np.floating]
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
        k = np.zeros(size, dtype="float")
        return cls(C, k)

    @classmethod
    def lock_disp_inds(cls, inds: np.ndarray[np.integer], k: np.ndarray[np.floating]):
        """
        Creates a `Dirichlet` instance with specified displacement `u` indices locked to given values.

        Parameters
        ----------
        inds : np.ndarray[np.integer]
            Indices of the displacement field `u` to be locked.
        k : np.ndarray[np.floating]
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

    def set_u_inds_vals(
        self, inds: np.ndarray[np.integer], vals: np.ndarray[np.floating]
    ):
        """
        Locks specified indices of the displacement field `u` to given values by modifying
        the matrix `C` and vector `k` accordingly. This involves zeroing out the specified
        rows in `C` and adjusting `k` to reflect the locked values.

        Parameters
        ----------
        inds : np.ndarray[np.integer]
            Indices of the displacement field `u` to be locked.
        vals : np.ndarray[np.floating]
            Values to lock the specified indices of `u` to.
        """

        def zero_rows(C: sps.coo_matrix, rows: np.ndarray[np.integer]):
            rm = np.hstack([(C.row == row).nonzero()[0] for row in rows])
            row = np.delete(C.row, rm)
            col = np.delete(C.col, rm)
            data = np.delete(C.data, rm)
            C = sps.coo_matrix((data, (row, col)), shape=(C.shape))
            return C

        def remove_zero_cols(C: sps.coo_matrix):
            unique, inverse = np.unique(C.col, return_inverse=True)
            C.col = np.arange(unique.size)[inverse]
            C._shape = (C.shape[0], unique.size)

        if inds.size != 0:
            C_coo = zero_rows(self.C.tocoo(), inds)
            remove_zero_cols(C_coo)
            self.C = C_coo.tocsc()
            self.k[inds] = vals

    def slave_reference_linear_relation(
        self,
        slaves: np.ndarray[int],
        references: np.ndarray[int],
        coefs: Union[np.ndarray[float], None] = None,
    ):
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
        coefs : Union[np.ndarray[float], None], optional
            2D array of coefficients defining the linear relationship between references and
            slaves.
            If None, the coefficients are set so that the slaves are the average of the references.
            By default None.
        """
        if coefs is None:
            coefs = np.ones(references.shape, dtype="float") / references.shape[1]

        sorted_slaves = slave_reference_linear_relation_sort(slaves, references)

        C = self.C.tocsr()
        rows, cols, data, self.k = slave_reference_linear_relation_inner(
            C.indices,
            C.indptr,
            C.data,
            self.k,
            slaves,
            references,
            coefs,
            sorted_slaves,
        )
        C = sps.coo_matrix((data, (rows, cols)), shape=C.shape)

        unique, inverse = np.unique(C.col, return_inverse=True)
        C.col = np.arange(unique.size)[inverse]
        C._shape = (C.shape[0], unique.size)
        self.C = C.tocsc()

    def u_du_ddof(
        self, dof: np.ndarray[np.floating]
    ) -> tuple[np.ndarray[np.floating], sps.csc_matrix]:
        """
        Computes the displacement field `u` and its derivative with respect to the degrees of freedom `dof`.
        The displacement field is calculated as `u = C @ dof + k`, and its derivative is `C`.

        Parameters
        ----------
        dof : np.ndarray[np.floating]
            The degrees of freedom of the problem, representing the input to the affine mapping.

        Returns
        -------
        u, du_ddof : tuple[np.ndarray[np.floating], sps.csc_matrix]
            A tuple containing the displacement field `u` and its derivative with respect to `dof`.
        """
        u = self.C @ dof + self.k
        du_ddof = self.C
        return u, du_ddof

    def u(self, dof: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Computes the displacement field `u` using the affine mapping `u = C @ dof + k`.

        Parameters
        ----------
        dof : np.ndarray[np.floating]
            The degrees of freedom of the problem, representing the input to the affine mapping.

        Returns
        -------
        u : np.ndarray[np.floating]
            The computed displacement field `u`.
        """
        if dof.ndim == 1:
            u = self.C @ dof + self.k
        else:
            m, n = self.C.shape
            dof_shape = dof.shape[:-1]
            u = (self.C @ dof.reshape((-1, n)).T + self.k.reshape((m, 1))).T.reshape(
                (*dof_shape, m)
            )
        return u

    def dof_lsq(self, u: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Computes the degrees of freedom `dof` from the displacement field `u` using a least squares approximation.
        This method performs a least squares 'inversion' of the affine mapping `u = C @ dof + k`.
        It solves the linear problem `C.T @ C @ dof = C.T @ (u - k)` for `dof`.

        Parameters
        ----------
        u : np.ndarray[np.floating]
            The displacement field from which to compute the degrees of freedom.

        Returns
        -------
        dof : np.ndarray[np.floating]
            The computed degrees of freedom corresponding to the given displacement field.
        """
        if u.ndim == 1:
            dof = cholesky(self.C.T @ self.C)(self.C.T @ (u - self.k))
        else:
            m, n = self.C.shape
            u_shape = u.shape[:-1]
            dof = cholesky(self.C.T @ self.C)(
                (u.reshape((-1, m)) - self.k[None]) @ self.C
            ).reshape((*u_shape, n))
        return dof


@nb.njit(cache=True)
def slave_reference_linear_relation_sort(
    slaves: np.ndarray[int], references: np.ndarray[int]
) -> np.ndarray[int]:
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
    queue = [s for s in slaves if indegree[s] == 0]

    # Topological sorting via BFS
    sorted_slaves = np.empty(len(slaves), dtype="int")
    i = 0
    while queue:
        slave = queue.pop(0)
        sorted_slaves[i] = slave
        i += 1
        for dependent_slave in graph[slave]:
            indegree[dependent_slave] -= 1
            if indegree[dependent_slave] == 0:
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
    sorted_slaves: np.ndarray[int],
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
    dict_list = [
        {indices[j]: data[j] for j in range(indptr[i], indptr[i + 1])}
        for i in range(indptr.size - 1)
    ]

    # Apply linear relation
    slaves_to_index = {s: i for i, s in enumerate(slaves)}
    for slave in sorted_slaves:
        i = slaves_to_index[slave]
        new_row = {}
        for reference_ind, coef in zip(references[i], coefs[i]):
            reference_row = dict_list[reference_ind]
            for ind, val in reference_row.items():
                if ind in new_row:
                    new_row[ind] += coef * val
                else:
                    new_row[ind] = coef * val
        new_row = {ind: val for ind, val in new_row.items() if val != 0}
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


# %%
class DirichletConstraintHandler:
    """
    Manage linear Dirichlet constraints for variational or optimization problems.

    This class accumulates linear equations of the form `D @ u = c` representing Dirichlet
    boundary conditions or linear relationships between degrees of freedom (DOFs), and
    computes a reduced basis representation that parametrizes the set of admissible solutions.

    Specifically, it computes matrices `C` and `k` such that any vector `u` satisfying
    `D @ u = c` can be written as:

        u = C @ dof + k

    where `dof` is a reduced vector of free parameters.

    Attributes
    ----------
    nb_dofs_init : int
        Number of DOFs in the original unconstrained system, before adding any reference DOFs.
    lhs : sps.spmatrix
        Accumulated constraint matrix D (left-hand side of the Dirichlet conditions).
    rhs : np.ndarray[np.floating]
        Accumulated right-hand side vector c of the Dirichlet conditions.
    """

    nb_dofs_init: int
    lhs: sps.spmatrix
    rhs: np.ndarray[np.floating]

    def __init__(self, nb_dofs_init: int):
        """
        Initialize a Dirichlet constraint handler.

        Parameters
        ----------
        nb_dofs_init : int
            The number of initial degrees of freedom in the unconstrained system.
            This value is used to size the initial constraint matrix and manage
            later extensions with reference DOFs.
        """
        try:
            from sparseqr import qr

            self._qr = qr
        except ImportError as e:
            raise ModuleNotFoundError(
                "The package 'sparseqr' is required for QR decomposition but is not installed.\n"
                "\n"
                "Note: Installing 'sparseqr' may require a C++ compiler, CMake, and the SuiteSparse library.\n"
                "On Linux, you can typically install these dependencies via your package manager. For example:\n"
                "    sudo apt install libsuitesparse-dev cmake build-essential\n"
                "\n"
                "Then install sparseqr with pip:\n"
                "    pip install sparseqr\n"
                "\n"
                "If installation continues to fail, refer to the official documentation:\n"
                "    https://pypi.org/project/sparseqr/"
            ) from e
        self.nb_dofs_init = nb_dofs_init
        self.lhs = sps.coo_matrix(np.empty((0, nb_dofs_init), dtype="float"))
        self.rhs = np.empty(0, dtype="float")

    def copy(self) -> "DirichletConstraintHandler":
        """
        Create a deep copy of this DirichletConstraintHandler instance.

        Returns
        -------
        DirichletConstraintHandler
            A new instance with the same initial number of DOFs, constraint matrix, and right-hand side vector.
            All internal data is copied, so modifications to the returned handler do not affect the original.
        """
        return copy.deepcopy(self)

    def add_eqs(self, lhs: sps.spmatrix, rhs: np.ndarray[np.floating]):
        """
        Add linear constraint equations of the form D_new @ u = c_new.

        Appends the given equations to the existing Dirichlet constraint system.
        If the number of columns in `lhs` matches the initial DOF count, it is
        automatically extended with zero-padding to match the current DOF count
        (in case reference DOFs have been added).

        Parameters
        ----------
        lhs : sps.spmatrix
            Constraint matrix D_new of shape (n_eqs, nb_dofs) to be added to the system.
        rhs : np.ndarray[np.floating]
            Right-hand side values c_new of shape (n_eqs,) corresponding to the constraint.

        Raises
        ------
        ValueError
            If the number of columns in `lhs` does not match the initial or current DOF count.
        """
        nb_eqs, nb_dofs = lhs.shape
        assert nb_eqs == rhs.size
        if nb_dofs == self.lhs.shape[1]:
            self.lhs = sps.vstack((self.lhs, lhs))
        elif nb_dofs == self.nb_dofs_init:
            zero = sps.coo_matrix((nb_eqs, self.lhs.shape[1] - self.nb_dofs_init))
            new_lhs = sps.hstack((lhs, zero))
            self.lhs = sps.vstack((self.lhs, new_lhs))
        else:
            raise ValueError(
                "lhs.shape[1] must match either the initial number of dofs or the current one."
            )

        self.rhs = np.hstack((self.rhs, rhs))

    def add_ref_dofs(self, nb_dofs: int):
        """
        Extend the system by adding new reference DOFs.

        This increases the number of columns in the constraint matrix by `nb_dofs`,
        initializing them with zeros in all existing constraint equations.

        Parameters
        ----------
        nb_dofs : int
            Number of new reference DOFs to append at the end of the current DOF vector.
        """
        zero = sps.coo_matrix((self.lhs.shape[0], nb_dofs))
        self.lhs = sps.hstack((self.lhs, zero))

    def add_ref_dofs_with_behavior(
        self, behavior_mat: sps.spmatrix, slave_inds: np.ndarray[np.integer]
    ):
        """
        Add new reference DOFs and define their influence on existing DOFs via a behavioral relation.

        This method appends new reference DOFs and enforces their relationship to existing
        DOFs (called "slaves") through a linear behavior matrix `behavior_mat`. The resulting constraints
        take the form:

            `behavior_mat @ ref_dofs - u[slave_inds] = 0`

        and are added to the global constraint system.

        Parameters
        ----------
        behavior_mat : sps.spmatrix
            Matrix of shape `(n_slaves, n_ref_dofs)` defining how each reference DOF influences
            the corresponding slave DOFs.

        slave_inds : np.ndarray[np.integer]
            Indices of the slave DOFs. Must have length `n_slaves` and must be in the same
            order as the rows of `behavior_mat`.

        Raises
        ------
        AssertionError
            If the number of rows in `behavior_mat` does not match the size of `slave_inds`.
        """
        nb_slaves, nb_ref_dofs = behavior_mat.shape
        assert nb_slaves == slave_inds.size
        data = -np.ones(nb_slaves, dtype="float")
        rows = np.arange(nb_slaves)
        slaves_counterpart = sps.coo_matrix(
            (data, (rows, slave_inds)), shape=(nb_slaves, self.lhs.shape[1])
        )
        lhs_to_add = sps.hstack((slaves_counterpart, behavior_mat))
        rhs_to_add = np.zeros(nb_slaves, dtype="float")
        self.add_ref_dofs(nb_ref_dofs)
        self.add_eqs(lhs_to_add, rhs_to_add)

    def add_rigid_body_constraint(
        self,
        ref_point: np.ndarray[np.floating],
        slaves_inds: np.ndarray[np.integer],
        slaves_positions: np.ndarray[np.floating],
    ):
        """
        Add a reference node and impose a rigid body motion constraint on a set of slave nodes.

        This method introduces new reference degrees of freedom (DOFs) corresponding to a reference node
        located at `ref_point`, and constrains the displacements of a set of slave nodes (given by
        `nodes_inds` and `nodes_positions`) to follow a rigid body motion defined by the reference node.
        The rigid body motion includes both translation and rotation about the reference point.

        The imposed constraint ensures that the displacement of each slave node is a linear combination of
        the reference node's translation and rotation, enforcing a rigid connection between the reference
        and the slaves.

        Parameters
        ----------
        ref_point : np.ndarray[np.floating]
            Reference point (origin for rotation and translation), array of shape (3,) representing (x, y, z).
        slaves_inds : np.ndarray[np.integer]
            Indices of the degrees of freedom of the slave displacements.
            Shape (3, n), where n is the number of slave nodes; each column contains the x, y, z DOF indices for a node.
        slaves_positions : np.ndarray[np.floating]
            Initial positions of the slave nodes.
            Shape (3, n), where each column contains the (x, y, z) coordinates of a slave node in the physical space.
        """
        x, y, z = (
            slaves_positions[:, :, None] - ref_point[:, None, None]
        )  # x is of shape (n, 1)
        ones = np.ones_like(x)
        behavior = sps.bmat(
            [
                [None, z, -y, ones, None, None],
                [-z, None, x, None, ones, None],
                [y, -x, None, None, None, ones],
            ]
        )  # behavior matrix for U(X; theta, t) = theta \carat X + t
        inds = slaves_inds.ravel()
        self.add_ref_dofs_with_behavior(behavior, inds)

    def add_eqs_from_inds_vals(
        self, inds: np.ndarray[np.integer], vals: np.ndarray[np.floating] = None
    ):
        """
        Add pointwise Dirichlet conditions by prescribing values at specific DOFs.

        This is a convenience method for adding equations of the form:

            u[i] = v

        for given indices `i` and corresponding values `v`.

        Parameters
        ----------
        inds : np.ndarray[np.integer]
            Indices of the DOFs to constrain.

        vals : np.ndarray[np.floating], optional
            Values to prescribe at the corresponding indices. If None, zeros are used.
            Must have the same size as `inds`.

        Raises
        ------
        AssertionError
            If `vals` is provided and does not match the shape of `inds`.
        """
        nb_eqs = inds.size
        vals = np.zeros(nb_eqs, dtype="float") if vals is None else vals
        assert vals.size == nb_eqs
        data = np.ones(nb_eqs, dtype="float")
        rows = np.arange(nb_eqs)
        lhs = sps.coo_matrix((data, (rows, inds)), shape=(nb_eqs, self.lhs.shape[1]))
        self.add_eqs(lhs, vals)

    def make_C_k(self) -> tuple[sps.spmatrix, np.ndarray[np.floating]]:
        """
        Construct the transformation (C, k) that enforces the Dirichlet constraints.

        Solves the linear constraint system D @ u = c by computing:
        - a basis `C` for the nullspace of D (i.e., D @ C = 0),
        - a particular solution `k` such that D @ k = c.

        Any admissible vector u satisfying the constraints can then be written as:

            u = C @ dof + k

        where `dof` is a reduced set of unconstrained degrees of freedom.

        Returns
        -------
        C : sps.spmatrix
            Basis of the nullspace of D, of shape (n_full_dofs, n_free_dofs), such that D @ C = 0.

        k : np.ndarray[np.floating]
            Particular solution of shape (n_full_dofs,) such that D @ k = c.
        """
        # Step 1: Perform sparse QR factorization of D^T
        # D^T = Q @ R @ P^T, where D = self.lhs and P is a column permutation
        Q, R, perm, rank = self._qr(self.lhs.T, economy=False)
        Q = Q.tocsc()
        R = R.tocsc()

        # Step 2: Extract a basis C of the nullspace of D (i.e., such that D @ C = 0)
        # These correspond to the last n - rank columns of Q
        C = Q[:, rank:]

        # Step 3: Compute a particular solution k such that D @ k = c
        # (c = self.rhs contains the prescribed values)

        # 3.1: Apply the column permutation to the RHS vector
        c_tilde = self.rhs[np.array(perm)]

        # 3.2: Extract the leading square block R1 of R (size rank × rank)
        R1 = R[:rank, :rank].tocsc()

        # 3.3: Take the first 'rank' entries of the permuted RHS
        c1 = c_tilde[:rank]

        # 3.4: Solve the triangular system R1^T @ y1 = c1
        y1 = sps.linalg.spsolve(R1.T, c1)

        # 3.5: Complete the vector y by padding with zeros (for the nullspace part)
        y = np.zeros(Q.shape[0])
        y[:rank] = y1

        # 3.6: Recover the particular solution k = Q @ y
        k = Q @ y

        # Step 4: Return the nullspace basis C and the particular solution k
        return C, k

    def get_reduced_Ck(self) -> tuple[sps.spmatrix, np.ndarray]:
        """
        Compute and return the reduced transformation matrices `(C_u, k_u)` for the physical degrees of freedom.

        This method extracts only the first `nb_dofs_init` rows of the full constraint transformation matrices `(C, k)`
        produced by `make_C_k()`, yielding a pair `(C_u, k_u)` such that the physical solution vector
        `u_phys = C_u @ dof_ext + k_u` satisfies all imposed Dirichlet constraints for the original system.

        Returns
        -------
        tuple[sps.spmatrix, np.ndarray]
            A tuple `(C_u, k_u)` where:
            - `C_u` (`sps.spmatrix`): The reduced nullspace basis matrix of shape (`nb_dofs_init`, n_free_dofs).
            - `k_u` (`np.ndarray`): The reduced particular solution vector of shape (`nb_dofs_init`,).

        Notes
        -----
        - Only the rows corresponding to the initial (physical) degrees of freedom are returned.
        - The full transformation matrices may include additional reference DOFs, which are omitted here.
        """
        C_ext, k_ext = self.make_C_k()
        C_ref = C_ext[self.nb_dofs_init :, :]
        k_ref = k_ext[self.nb_dofs_init :]
        C_u = C_ext[: self.nb_dofs_init, :]
        k_u = k_ext[: self.nb_dofs_init]
        if sps.linalg.norm(C_ref) != 0:
            print("Warning : not all reference point DoFs are specified.")
            print(
                f"Try specifying reference dofs number {np.abs(C_ref).sum(axis=1).A.ravel().nonzero()[0].tolist()}."
            )
        return C_u, k_u

    def create_dirichlet(self):
        C, k = self.get_reduced_Ck()
        return Dirichlet(C, k)

    def get_ref_multipliers_from_internal_residual(self, K_u_minus_f):
        """
        Compute the Lagrange multipliers associated with reference point constraints
        from the internal residual vector of the mechanical problem.

        This method reconstructs the multipliers λ enforcing the constraints linked to
        reference degrees of freedom (DOFs) using the internal residual
        `K_u_minus_f = K @ u - f`. The derivation relies on the relation:

            C_ref.T @ λ = - C_u.T @ (K @ u - f),

        where the transformation matrix C = [C_u; C_ref] maps reduced DOFs to the full set
        (physical + reference) while satisfying all constraint equations
        (built via a QR decomposition in `make_C_k`).

        Solving for λ in a least-squares sense yields:

            λ = - (C_ref @ C_ref.T)^{-1} @ C_ref @ C_u.T @ (K @ u - f).

        Parameters
        ----------
        K_u_minus_f : np.ndarray
            The internal residual vector (K @ u - f) of size `(nb_dofs_init,)`,
            expressed only for the physical DOFs.

        Returns
        -------
        np.ndarray
            The Lagrange multipliers λ associated with the reference point constraints.

        Notes
        -----
        - The result corresponds to the reaction forces (or generalized forces) transmitted
        by the reference DOFs onto the system, ensuring equilibrium.
        - The internal residual `K @ u - f` must be assembled consistently
        with the stiffness matrix `K` and the load vector `f`.
        - This method assumes that `C_ref @ C_ref.T` is invertible,
        which is guaranteed if the reference constraints are linearly independent.
        """
        C, _ = self.make_C_k()
        C_ref = C[self.nb_dofs_init :, :]
        C_u = C[: self.nb_dofs_init, :]
        lamb = -cholesky(C_ref @ C_ref.T)(C_ref @ C_u.T @ K_u_minus_f)
        return lamb
