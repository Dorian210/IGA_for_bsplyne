import warnings
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from IGA_for_bsplyne.fallback_sparse_qr import my_qr_sparse

# --- Cholmod / scikit-sparse ---
try:
    from sksparse.cholmod import cholesky

    HAS_CHOLMOD = True
except ImportError:
    warnings.warn(
        "\n[IGA_for_bsplyne] High-performance solver 'scikit-sparse' (CHOLMOD) is not installed.\n"
        "To enable fast Cholesky factorization for large 3D problems, we recommend:\n"
        "\n"
        "    conda install -c conda-forge scikit-sparse\n"
        "\n"
        "Alternatively, manual installation via pip is possible but complex and requires SuiteSparse headers:\n"
        "    (Linux) sudo apt install libsuitesparse-dev\n"
        "    (macOS) brew install suitesparse\n"
        "    pip install scikit-sparse\n"
        "\n"
        "Fallback: Using 'scipy.sparse.linalg.spsolve' (slower)."
    )
    HAS_CHOLMOD = False


def solve_sparse(A: sps.spmatrix, b: NDArray[np.floating]) -> NDArray[np.floating]:
    """Solve Ax=b using Cholmod if available, otherwise fallback to spsolve."""
    if HAS_CHOLMOD:
        try:
            factor = cholesky(A)  # type: ignore
            return factor(b)
        except:
            # Handle cases where A might not be PD
            print("Matrix not positive definite. Falling back to scipy solver.")
            return spsolve(A, b)  # type: ignore
    else:
        return spsolve(A, b)  # type: ignore


# --- SparseQR ---
try:
    from sparseqr import qr

    HAS_SPARSEQR = True
except ImportError:
    warnings.warn(
        "\n[IGA_for_bsplyne] High-performance solver 'sparseqr' is not installed.\n"
        "To enable fast QR decomposition, we recommend:\n"
        "\n"
        "    conda install -c conda-forge scikit-sparse\n"
        "    pip install sparseqr\n"
        "\n"
        "Note: 'sparseqr' will link to SuiteSparse installed via conda.\n"
        "Fallback: Using internal 'my_qr_sparse' implementation (slower)."
    )
    HAS_SPARSEQR = False


def qr_sparse(
    A: sps.spmatrix,
) -> tuple[sps.spmatrix, sps.spmatrix, NDArray[np.integer], int]:
    """Compute QR decomposition using sparseqr if available, otherwise fallback to my_qr_sparse."""
    if HAS_SPARSEQR:
        return qr(A, economy=False)  # type: ignore
    else:
        return my_qr_sparse(A)
