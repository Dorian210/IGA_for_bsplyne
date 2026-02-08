from typing import Union
import numpy as np
import scipy.sparse as sps
import numba as nb
import copy
from IGA_for_bsplyne.solvers import solve_sparse, qr_sparse


class Dirichlet:
    """
    Affine representation of linear Dirichlet constraints.

    This class represents Dirichlet boundary conditions through an affine
    transformation between a reduced vector of free parameters (`dof`)
    and the full physical displacement vector (`u`):

        u = C @ dof + k

    where:

    - `u` is the full displacement vector satisfying all imposed constraints.
    - `dof` is the reduced vector of unconstrained degrees of freedom.
    - `C` is a sparse matrix whose columns form a basis of admissible variations.
    - `k` is a particular displacement vector satisfying the constraints.

    This representation allows:
    - elimination of constrained DOFs,
    - support for general linear multi-point constraints.

    Attributes
    ----------
    C : sps.csc_matrix of shape (n_full_dofs, n_free_dofs)
        Sparse matrix defining the linear mapping from reduced DOFs to full DOFs.
    k : np.ndarray of shape (n_full_dofs,)
        Particular solution ensuring that `u` satisfies the Dirichlet constraints.
    """

    C: sps.csc_matrix
    k: np.ndarray[np.floating]

    def __init__(self, C: sps.spmatrix, k: np.ndarray[np.floating]):
        """
        Initialize an affine Dirichlet constraint representation.

        Parameters
        ----------
        C : sps.spmatrix
            Sparse matrix of shape (n_full_dofs, n_free_dofs) defining
            the linear mapping from reduced DOFs to full DOFs.
        k : np.ndarray[np.floating]
            Vector of shape (n_full_dofs,) defining the affine offset.

        Notes
        -----
        The matrix `C` is internally converted to CSC format for efficient
        matrix-vector products.
        """
        self.C = C.tocsc()
        self.k = k

    @classmethod
    def eye(cls, size: int):
        """
        Create an unconstrained Dirichlet mapping.

        This method returns a `Dirichlet` object corresponding to the identity
        transformation:

            u = dof

        i.e. no Dirichlet constraints are applied.

        Parameters
        ----------
        size : int
            Number of degrees of freedom.

        Returns
        -------
        dirichlet : Dirichlet
            Identity `Dirichlet` object with:
            - C = identity(size)
            - k = 0
        """
        C = sps.eye(size)
        k = np.zeros(size, dtype="float")
        return cls(C, k)

    @classmethod
    def lock_disp_inds(cls, inds: np.ndarray[np.integer], k: np.ndarray[np.floating]):
        """
        Create Dirichlet constraints by prescribing displacement values at selected DOFs.

        This method enforces pointwise Dirichlet conditions of the form:

            u[i] = k[i]    for i in inds

        All other DOFs remain unconstrained.

        Parameters
        ----------
        inds : np.ndarray[np.integer]
            Indices of the displacement vector `u` to be constrained.
        k : np.ndarray[np.floating]
            Target displacement vector of shape (n_full_dofs,). Only the
            values at `inds` are enforced.

        Returns
        -------
        dirichlet : Dirichlet
            `Dirichlet` instance representing the prescribed displacement constraints.

        Notes
        -----
        Constrained DOFs are removed from the reduced DOF vector. The resulting
        number of reduced DOFs will therefore be smaller than `k.size`.
        """
        C = sps.eye(k.size)
        dirichlet = cls(C, k.copy())
        dirichlet.set_u_inds_vals(inds, dirichlet.k[inds])
        return dirichlet

    def set_u_inds_vals(
        self, inds: np.ndarray[np.integer], vals: np.ndarray[np.floating]
    ):
        """
        Impose pointwise Dirichlet conditions by prescribing displacement values
        at selected full DOFs.

        This method enforces constraints of the form:

            u[i] = vals[j]     for i = inds[j]

        by modifying the affine mapping:

            u = C @ dof + k

        The modification removes the corresponding admissible variations in `C`
        and updates `k` so that the prescribed values are always satisfied.
        As a result, constrained DOFs are eliminated from the reduced DOF vector.

        Parameters
        ----------
        inds : np.ndarray[np.integer]
            Indices of the full displacement vector `u` to constrain.
        vals : np.ndarray[np.floating]
            Prescribed displacement values associated with `inds`.
            Must have the same length as `inds`.

        Notes
        -----
        - Rows of `C` corresponding to constrained DOFs are zeroed.
        - Columns of `C` that become unused are removed, which may reduce
          the number of reduced DOFs.
        - The affine offset `k` is updated to enforce the prescribed values.
        - This operation modifies the current Dirichlet object in place.
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
        Impose linear multi-point constraints between full DOFs.

        This method enforces relations of the form:

            u[slave_i] = sum_j coefs[i, j] * u[references[i, j]]

        by modifying the affine mapping:

            u = C @ dof + k

        Slave DOFs are expressed as linear combinations of reference DOFs.
        The corresponding admissible variations are propagated into `C`
        and `k`, effectively eliminating slave DOFs from the reduced parameter
        vector while preserving the imposed constraints.

        Parameters
        ----------
        slaves : np.ndarray[int] of shape (n_slaves,)
            Indices of DOFs that are constrained (slave DOFs).
        references : np.ndarray[int] of shape (n_slaves, n_refs)
            Reference DOF indices controlling each slave DOF.
            Row `i` contains the references associated with `slaves[i]`.
        coefs : np.ndarray[float], optional, shape (n_slaves, n_refs)
            Linear combination coefficients linking references to slaves.
            If None, slaves are defined as the average of their references.

        Notes
        -----
        - Slave DOFs are removed from the reduced DOF vector.
        - The constraint propagation accounts for hierarchical dependencies
          between slave DOFs using a topological ordering.
        - This operation is typically faster than using `DirichletConstraintHandler`,
          as it directly modifies the affine mapping without constructing
          intermediate constraint objects.
        - Cyclic dependencies between slaves are not supported and will raise an error.
        - This operation modifies the current Dirichlet object in place.

        Examples
        --------
        To impose u[i] = 0.5 * (u[j] + u[k]):

        >>> slaves = np.array([i])
        >>> references = np.array([[j, k]])
        >>> coefs = np.array([[0.5, 0.5]])
        >>> dirichlet.slave_reference_linear_relation(slaves, references, coefs)
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
        Evaluate the displacement field and its derivative with respect to the reduced DOFs.

        The displacement field is obtained from the affine mapping:

            u = C @ dof + k

        Since the mapping is affine, the derivative of `u` with respect to `dof`
        is constant and equal to `C`.

        Parameters
        ----------
        dof : np.ndarray[np.floating] of shape (n_dof,)
            Reduced degrees of freedom.

        Returns
        -------
        u : np.ndarray[np.floating] of shape (n_u,)
            Displacement field.
        du_ddof : sps.csc_matrix of shape (n_u, n_dof)
            Jacobian of `u` with respect to `dof` : the matrix `C`.

        Notes
        -----
        The returned derivative is independent of `dof` because the mapping is affine.
        """
        u = self.C @ dof + self.k
        du_ddof = self.C
        return u, du_ddof

    def u(self, dof: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Evaluate the displacement field from reduced DOFs.

        The displacement field is computed using the affine mapping:

            u = C @ dof + k

        Parameters
        ----------
        dof : np.ndarray[np.floating]
            Reduced degrees of freedom. Can be either:

            - shape (n_dof,)
            - shape (..., n_dof) for batched evaluation

        Returns
        -------
        u : np.ndarray[np.floating]
            Displacement field with shape:

            - (n_u,) if `dof` is 1D
            - (..., n_u) if `dof` is batched

        Notes
        -----
        This method supports vectorized evaluation over multiple DOF vectors.
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
        Compute reduced DOFs from a displacement field using a least-squares projection.

        This method computes `dof` such that:

            u ≈ C @ dof + k

        by solving the normal equations:

            (Cᵀ C) dof = Cᵀ (u - k)

        Parameters
        ----------
        u : np.ndarray[np.floating]
            Displacement field. Can be either:

            - shape (n_u,)
            - shape (..., n_u) for batched projection

        Returns
        -------
        dof : np.ndarray[np.floating]
            Reduced degrees of freedom with shape:

            - (n_dof,) if `u` is 1D
            - (..., n_dof) if `u` is batched

        Notes
        -----
        This operation performs a least-squares inversion of the affine mapping.
        If `C` does not have full column rank, the solution corresponds to a
        minimum-norm least-squares solution.

        The system `(Cᵀ C)` is solved using a sparse linear solver.
        """
        if u.ndim == 1:
            dof = solve_sparse(self.C.T @ self.C, self.C.T @ (u - self.k))
        else:
            m, n = self.C.shape
            u_shape = u.shape[:-1]
            dof = solve_sparse(
                self.C.T @ self.C, (u.reshape((-1, m)) - self.k[None]) @ self.C
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
    Accumulate affine linear constraints and construct the associated Dirichlet mapping.

    This class is designed to impose linear affine constraints of the form:

        D @ u = c

    where `u` is the full displacement (or state) vector. Constraints are progressively
    accumulated and, once fully specified, converted into an affine parametrization of
    the admissible solution space:

        u = C @ dof + k

    where:

    - `C` is a basis of the nullspace of `D` (i.e. `D @ C = 0`) obtained by
    QR decomposition,
    - `k` is a particular solution satisfying the constraints,
    obtained through the QR decomposition which helps solve:

        D k = c

    - `dof` is a reduced vector of unconstrained degrees of freedom.

    The main purpose of this class is to provide a flexible and robust interface
    to define constraints before constructing a `Dirichlet` object representing
    the reduced parametrization.

    Typical workflow
    ----------------
    1. Create a handler with the number of physical DOFs (`u`).
    2. Add constraint equations using helper methods.
    3. **Ensure that all reference DOFs (e.g., translations or rotations introduced for
    rigid-body relations) are fully constrained before computing `C` and `k`.**
    4. Build the reduced representation `(C, k)` or directly create a `Dirichlet` object.

    This separation allows constraints to be assembled incrementally and validated
    before generating the final affine mapping.

    Attributes
    ----------
    nb_dofs_init : int
        Number of DOFs in the original unconstrained system (physical DOFs `u`).
    lhs : sps.spmatrix
        Accumulated constraint matrix `D`.
    rhs : np.ndarray[np.floating]
        Accumulated right-hand side vector `c`.

    Notes
    -----
    - The constraint system may include additional reference DOFs introduced
    to express kinematic relations or rigid-body behaviors.
    - **All reference DOFs must be fully constrained before computing `C` and `k`;**
    otherwise, DOFs that lie in the kernel of `D` cannot be controlled and imposed values
    (e.g., prescribed translations) may not appear in the resulting solution `k`.
    - The resulting affine mapping guarantees that any generated vector `u`
    satisfies all imposed constraints.
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
        Append linear affine constraint equations to the global constraint system.

        Adds equations of the form:

            lhs @ u = rhs

        to the accumulated constraint system `D @ u = c`.

        If `lhs` is expressed only in terms of the initial physical DOFs
        (`nb_dofs_init` columns), it is automatically extended with zero-padding
        to match the current number of DOFs (e.g. after reference DOFs have been added).

        Parameters
        ----------
        lhs : sps.spmatrix of shape (n_eqs, n_dofs)
            Constraint matrix defining the left-hand side of the equations.
            The number of columns must match either:
            - the initial number of DOFs (`nb_dofs_init`), or
            - the current number of DOFs in the handler.

        rhs : np.ndarray[np.floating] of shape (n_eqs,)
            Right-hand side values associated with the constraint equations.

        Raises
        ------
        ValueError
            If the number of columns of `lhs` is incompatible with the current
            constraint system.

        Notes
        -----
        Constraints are appended to the existing system and are not simplified
        or checked for redundancy.
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
        Append additional reference DOFs to the constraint system.

        This method increases the size of the global DOF vector by adding
        `nb_dofs` new reference DOFs. These DOFs are introduced without
        imposing any constraint, i.e. they appear with zero coefficients
        in all existing equations.

        Parameters
        ----------
        nb_dofs : int
            Number of reference DOFs to append to the global DOF vector.

        Notes
        -----
        Reference DOFs are typically used to express kinematic relations,
        rigid body motions, or other auxiliary constraint parametrizations.
        """
        zero = sps.coo_matrix((self.lhs.shape[0], nb_dofs))
        self.lhs = sps.hstack((self.lhs, zero))

    def add_ref_dofs_with_behavior(
        self, behavior_mat: sps.spmatrix, slave_inds: np.ndarray[np.integer]
    ):
        """
        Introduce reference DOFs and constrain slave DOFs through a linear relation.

        This method appends new reference DOFs and enforces a linear behavioral
        relation linking these reference DOFs to existing DOFs (called *slave DOFs*).
        The imposed constraints take the form:

            behavior_mat @ ref_dofs - u[slave_inds] = 0


        Parameters
        ----------
        behavior_mat : sps.spmatrix of shape (n_slaves, n_ref_dofs)
            Linear operator defining how each reference DOF contributes to the
            corresponding slave DOFs.

        slave_inds : np.ndarray[np.integer] of shape (n_slaves,)
            Indices of slave DOFs that are controlled by the reference DOFs.
            The ordering must match the rows of `behavior_mat`.

        Raises
        ------
        AssertionError
            If the number of slave DOFs is inconsistent with the number of
            rows of `behavior_mat`.

        Notes
        -----
        This method first adds the reference DOFs to the global system and then
        appends the corresponding constraint equations.
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
        Constrain slave nodes to follow a rigid body motion defined by a reference point.

        This method introduces six reference DOFs representing a rigid body motion:
        three rotations and three translations (in this order) around a reference point.
        The displacement of each slave node is constrained to follow the rigid body motion:

            u(X) = θ × (X - X_ref) + t

        where `θ` is the rotation vector and `t` is the translation vector.

        Parameters
        ----------
        ref_point : np.ndarray[np.floating] of shape (3,)
            Reference point defining the center of rotation.

        slaves_inds : np.ndarray[np.integer] of shape (3, n_nodes)
            DOF indices of the slave nodes. Each column contains the
            x, y, z DOF indices of a slave node.

        slaves_positions : np.ndarray[np.floating] of shape (3, n_nodes)
            Physical coordinates of the slave nodes.

        Notes
        -----
        - Six reference DOFs (θx, θy, θz, tx, ty, tz) are added to represent
          the rigid body motion.
        - The constraint is expressed as a linear relation between the
          reference DOFs and the slave displacements.
        - This method is commonly used to impose rigid connections,
          master node formulations, or reference frame constraints.
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
        Impose pointwise Dirichlet conditions on selected DOFs.

        This is a convenience method that adds constraint equations of the form:

            u[inds] = vals


        Parameters
        ----------
        inds : np.ndarray[np.integer] of shape (n_eqs,)
            Indices of DOFs to constrain.

        vals : np.ndarray[np.floating] of shape (n_eqs,), optional
            Prescribed values associated with each constrained DOF.
            If None, zero values are imposed.

        Raises
        ------
        AssertionError
            If `vals` is provided and its size differs from `inds`.

        Notes
        -----
        This method is equivalent to adding rows of the identity matrix
        to the constraint operator.
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
        Construct the affine transformation (C, k) enforcing all Dirichlet constraints.

        This method solves the linear constraint system:

            D @ u = c

        by computing:

        - a matrix `C` forming a basis of the nullspace of `D`
        (i.e., D @ C = 0),
        - a particular solution `k` such that D @ k = c.

        Any admissible vector `u` satisfying the constraints can then be written as:

            u = C @ dof + k

        where `dof` is the reduced vector of unconstrained degrees of freedom.

        The nullspace basis and the particular solution are obtained through
        a sparse QR factorization of Dᵀ.

        Returns
        -------
        C : sps.spmatrix
            Sparse matrix of shape (n_full_dofs, n_free_dofs) whose columns
            form a basis of the nullspace of `D`.

        k : np.ndarray[np.floating]
            Vector of shape (n_full_dofs,) representing a particular solution
            satisfying the constraints.

        Notes
        -----
        - The transformation (C, k) defines an affine parametrization of the
        admissible displacement space.
        - The reduced DOF vector `dof` corresponds to the coordinates of `u`
        in the nullspace basis.
        - The QR factorization is performed on Dᵀ to efficiently extract both
        the nullspace basis and the particular solution.
        """
        # Step 1: Perform sparse QR factorization of D^T
        # D^T = Q @ R @ P^T, where D = self.lhs and P is a column permutation
        Q, R, perm, rank = qr_sparse(self.lhs.T)
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
        Extract the affine constraint transformation restricted to physical DOFs.

        This method computes the full transformation (C, k) using `self.make_C_k()`
        and returns only the rows associated with the initial (physical) degrees
        of freedom. The resulting pair (C_u, k_u) defines:

            u_phys = C_u @ dof + k_u

        where `u_phys` satisfies all imposed Dirichlet constraints.

        Returns
        -------
        C_u : sps.spmatrix
            Reduced nullspace basis matrix of shape (nb_dofs_init, n_free_dofs).

        k_u : np.ndarray
            Reduced particular solution vector of shape (nb_dofs_init,).

        Notes
        -----
        - The full transformation (C, k) may include auxiliary reference DOFs
        introduced by multi-point or hierarchical constraints.
        - Only the rows corresponding to physical DOFs are returned.
        - A warning is emitted if reference DOFs remain unconstrained,
        which may indicate missing boundary specifications.
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
        """
        Build a `Dirichlet` object representing the reduced constraint transformation.

        This is a convenience wrapper around `get_reduced_Ck()` that constructs
        a `Dirichlet` object directly from the reduced affine mapping:

            u_phys = C_u @ dof + k_u

        Returns
        -------
        dirichlet : Dirichlet
            A `Dirichlet` instance encapsulating the reduced transformation matrices.
        """
        C, k = self.get_reduced_Ck()
        return Dirichlet(C, k)

    def get_ref_multipliers_from_internal_residual(self, K_u_minus_f):
        """
        Recover Lagrange multipliers associated with reference DOF constraints.

        This method reconstructs the Lagrange multipliers λ corresponding to
        reference DOFs using the internal residual vector of the mechanical system:

            r = K @ u - f

        The derivation relies on the partition of the transformation matrix:

            C = [C_u;
                C_ref]

        where `C_u` maps reduced DOFs to physical DOFs and `C_ref` maps them
        to reference DOFs.

        The multipliers satisfy:

            C_ref.T @ λ = - C_u.T @ r

        and are obtained via the least-squares solution:

            λ = - (C_ref @ C_ref.T)⁻¹ @ C_ref @ C_u.T @ r

        Parameters
        ----------
        K_u_minus_f : np.ndarray
            Internal residual vector of shape (nb_dofs_init,),
            corresponding to K @ u - f restricted to physical DOFs.

        Returns
        -------
        lamb : np.ndarray
            Vector of Lagrange multipliers associated with reference DOF constraints.

        Notes
        -----
        - The multipliers correspond to reaction forces or generalized constraint
        forces transmitted through reference DOFs.
        - The residual must be assembled consistently with the stiffness matrix
        and load vector used to compute the displacement field.
        - The matrix C_ref @ C_ref.T is assumed to be invertible, which holds
        when reference constraints are linearly independent.
        """
        C, _ = self.make_C_k()
        C_ref = C[self.nb_dofs_init :, :]
        C_u = C[: self.nb_dofs_init, :]
        lamb = -solve_sparse(C_ref @ C_ref.T, C_ref @ C_u.T @ K_u_minus_f)
        return lamb
