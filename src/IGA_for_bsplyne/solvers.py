import warnings
import scipy.sparse as sps
from IGA_for_bsplyne.fallback_sparse_qr import my_qr_sparse

# --- Cholmod / scikit-sparse ---
try:
    from sksparse.cholmod import cholesky

    HAS_CHOLMOD = True
except ImportError:
    warnings.warn(
        "The package 'scikit-sparse' (sksparse) is required for Cholesky decomposition but is not installed.\n"
        "\n"
        "Note: Installing scikit-sparse may require the SuiteSparse library, CMake, and a C++ compiler.\n"
        "On Linux, you can typically install SuiteSparse via your package manager. For example:\n"
        "    sudo apt install libsuitesparse-dev cmake build-essential\n"
        "\n"
        "Then install scikit-sparse with pip:\n"
        "    pip install scikit-sparse\n"
        "\n"
        "If installation continues to fail, refer to the official documentation:\n"
        "    https://pypi.org/project/scikit-sparse/"
    )
    HAS_CHOLMOD = False


def solve_sparse(A, b):
    """Solve Ax=b using Cholmod if available, otherwise fallback to spsolve."""
    if HAS_CHOLMOD:
        factor = cholesky(A)
        return factor(b)
    else:
        return sps.linalg.spsolve(A, b)


# --- SparseQR ---
try:
    from sparseqr import qr

    HAS_SPARSEQR = True
except ImportError:
    warnings.warn(
        "The package 'sparseqr' is required for QR decomposition but is not installed.\n"
        "\n"
        "Note: Installing 'sparseqr' may require SuiteSparse, CMake, and a C++ compiler.\n"
        "On Linux, you can typically install SuiteSparse via your package manager. For example:\n"
        "    sudo apt install libsuitesparse-dev cmake build-essential\n"
        "\n"
        "Then install sparseqr with pip:\n"
        "    pip install sparseqr\n"
        "\n"
        "If installation continues to fail, refer to the official documentation:\n"
        "    https://pypi.org/project/sparseqr/"
    )
    HAS_SPARSEQR = False


def qr_sparse(A):
    """Compute QR decomposition using sparseqr if available, otherwise fallback to my_qr_sparse."""
    if HAS_SPARSEQR:
        return qr(A, economy=False)
    else:
        return my_qr_sparse(A)
