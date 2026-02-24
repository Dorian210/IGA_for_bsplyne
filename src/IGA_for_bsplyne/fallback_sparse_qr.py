import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sps
import numba as nb
from numba.typed.typedlist import List as nb_List


@nb.njit
def get_givens_params(a, b):
    """
    Computes the Givens rotation parameters (c, s) such that the
    rotation matrix applied to [a, b]^T results in [r, 0]^T.
    """
    if b == 0:
        return 1.0, 0.0
    r = np.hypot(a, b)
    return a / r, -b / r


@nb.njit
def apply_rotation_sparse(
    idx_j, dat_j, idx_i, dat_i, c, s, start_col, end_col, w_j, w_i, active
):
    """
    Applies a Givens rotation to two sparse rows.

    Parameters:
    - idx_j, dat_j: Indices and data of the j-th row.
    - idx_i, dat_i: Indices and data of the i-th row.
    - c, s: Rotation parameters.
    - start_col, end_col: Column range to apply the rotation.
    - w_j, w_i: Pre-allocated dense work vectors for processing.
    - active: Pre-allocated boolean mask to track non-zero entries.
    """
    # Reset dense buffers
    w_j.fill(0)
    w_i.fill(0)
    active.fill(False)

    # Decompress sparse to dense
    for k in range(len(idx_j)):
        col = idx_j[k]
        w_j[col] = dat_j[k]
        active[col] = True
    for k in range(len(idx_i)):
        col = idx_i[k]
        w_i[col] = dat_i[k]
        active[col] = True

    new_idx_j, new_dat_j = [], []
    new_idx_i, new_dat_i = [], []

    # Perform rotation and re-compress
    for k in range(start_col, end_col):
        if active[k]:
            v_j, v_i = w_j[k], w_i[k]
            res_j = c * v_j - s * v_i
            res_i = s * v_j + c * v_i
            if abs(res_j) > 1e-15:
                new_idx_j.append(k)
                new_dat_j.append(res_j)
            if abs(res_i) > 1e-15:
                new_idx_i.append(k)
                new_dat_i.append(res_i)

    return (
        np.array(new_idx_j, dtype=np.int32),
        np.array(new_dat_j, dtype=np.float64),
        np.array(new_idx_i, dtype=np.int32),
        np.array(new_dat_i, dtype=np.float64),
    )


@nb.njit
def to_csr(rows_idx, rows_dat, n_rows):
    """
    Efficiently converts typed lists of sparse rows into a flat CSR structure.
    Uses pre-allocation to ensure O(nnz) performance.
    """
    nnz = 0
    for i in range(n_rows):
        nnz += len(rows_idx[i])

    out_data = np.empty(nnz, dtype=np.float64)
    out_indices = np.empty(nnz, dtype=np.int32)
    out_indptr = np.empty(n_rows + 1, dtype=np.int32)

    out_indptr[0] = 0
    curr_pos = 0
    for i in range(n_rows):
        row_len = len(rows_idx[i])
        next_pos = curr_pos + row_len
        out_data[curr_pos:next_pos] = rows_dat[i]
        out_indices[curr_pos:next_pos] = rows_idx[i]
        out_indptr[i + 1] = next_pos
        curr_pos = next_pos

    return out_data, out_indices, out_indptr


@nb.njit(
    nb.types.UniTuple(
        nb.types.Tuple(
            (
                nb.types.float64[::1],
                nb.types.int32[::1],
                nb.types.int32[::1],
            )
        ),
        2,
    )(
        nb.types.float64[:],
        nb.types.int32[:],
        nb.types.int32[:],
        nb.types.UniTuple(nb.types.int64, 2),
    ),
    cache=True,
)
def sparse_qr_numba(data, indices, indptr, shape):
    """
    Core Sparse QR decomposition using Givens rotations.

    Iterates through columns to eliminate sub-diagonal elements
    while maintaining a sparse representation using Numba typed lists.
    Constructs Q^T (stored in Q_idx/Q_dat) and R.
    """
    rows, cols = shape

    # Initialize R with A (CSR format)
    R_idx = nb_List()
    R_dat = nb_List()
    for r in range(rows):
        R_idx.append(indices[indptr[r] : indptr[r + 1]].copy())
        R_dat.append(data[indptr[r] : indptr[r + 1]].copy())

    # Initialize Q as a sparse Identity matrix
    Q_idx = nb_List()
    Q_dat = nb_List()
    for r in range(rows):
        Q_idx.append(np.array([r], dtype=np.int32))
        Q_dat.append(np.array([1.0], dtype=np.float64))

    # Pre-allocate work buffers to avoid repeated allocations
    max_dim = max(rows, cols)
    work_j = np.zeros(max_dim)
    work_i = np.zeros(max_dim)
    active_cols = np.zeros(max_dim, dtype=np.bool_)

    for j in range(cols):
        for i in range(rows - 1, j, -1):
            # 1. Look for val_ij to eliminate
            val_ij = 0.0
            for k_idx, col in enumerate(R_idx[i]):
                if col == j:
                    val_ij = R_dat[i][k_idx]
                    break

            if abs(val_ij) < 1e-15:
                continue

            # 2. Look for pivot val_jj
            val_jj = 0.0
            for k_idx, col in enumerate(R_idx[j]):
                if col == j:
                    val_jj = R_dat[j][k_idx]
                    break

            c, s = get_givens_params(val_jj, val_ij)

            # 3. Rotate R rows (from column j onwards)
            R_idx[j], R_dat[j], R_idx[i], R_dat[i] = apply_rotation_sparse(
                R_idx[j],
                R_dat[j],
                R_idx[i],
                R_dat[i],
                c,
                s,
                j,
                cols,
                work_j,
                work_i,
                active_cols,
            )

            # 4. Rotate Q_transpose rows (all columns)
            Q_idx[j], Q_dat[j], Q_idx[i], Q_dat[i] = apply_rotation_sparse(
                Q_idx[j],
                Q_dat[j],
                Q_idx[i],
                Q_dat[i],
                c,
                s,
                0,
                rows,
                work_j,
                work_i,
                active_cols,
            )

    Q_data, Q_indices, Q_indptr = to_csr(Q_idx, Q_dat, rows)
    R_data, R_indices, R_indptr = to_csr(R_idx, R_dat, rows)

    return (Q_data, Q_indices, Q_indptr), (R_data, R_indices, R_indptr)


def my_qr_sparse(
    A: sps.spmatrix,
) -> tuple[sps.csr_matrix, sps.csr_matrix, NDArray[np.integer], int]:
    """
    Computes the Sparse QR decomposition of matrix A.
    Uses column reordering to minimize fill-in.

    Returns:
    - Q: Orthogonal matrix (m x m)
    - R_perm: Upper triangular matrix (m x n)
    - permutation: Column permutation array
    - rank: Numerical rank of the matrix
    """
    A = A.tocsr()  # type: ignore
    m, n = A.shape

    # 1. Permutation to reduce fill-in
    col_counts = np.zeros(n, dtype=np.int32)
    np.add.at(col_counts, A.indices, 1)  # type: ignore
    permutation = np.argsort(col_counts).astype(np.int32)
    A_perm = A[:, permutation]  # type: ignore

    # 2. Numba QR
    (QT_data, QT_indices, QT_indptr), (R_data, R_indices, R_indptr) = sparse_qr_numba(
        np.ascontiguousarray(A_perm.data.astype(np.float64)),
        np.ascontiguousarray(A_perm.indices.astype(np.int32)),
        np.ascontiguousarray(A_perm.indptr.astype(np.int32)),
        (m, n),
    )

    # 3. Assemble results
    QT = sps.csr_matrix((QT_data, QT_indices, QT_indptr), shape=(m, m))
    R_perm = sps.csr_matrix((R_data, R_indices, R_indptr), shape=(m, n))

    # 4. Rank calculation
    diag_R = R_perm.diagonal()
    tol = (
        max(m, n) * np.finfo(float).eps * np.abs(diag_R).max() if len(diag_R) > 0 else 0
    )
    rank = np.count_nonzero(np.abs(diag_R) > tol)

    return QT.T, R_perm, permutation, rank


if __name__ == "__main__":

    n, m = 10_000, 100
    nnz = 7 * m
    A_rows = np.random.randint(0, n, nnz)
    A_cols = np.random.randint(0, m, nnz)
    A_data = np.random.randn(nnz)
    A_sparse = sps.coo_matrix((A_data, (A_rows, A_cols)), shape=(n, m)).tocsr()

    Q, R, perm, rank = my_qr_sparse(A_sparse)  # type: ignore
    P = sps.coo_matrix(
        (np.ones(perm.size), (perm, np.arange(perm.size))),
        shape=(A_sparse.shape[1], A_sparse.shape[1]),
    )

    print(np.abs(Q @ R @ P.T - A_sparse).max())

# %%
