import functools
import numpy as np
import scipy.sparse as sps

from bsplyne import BSpline


class IGAPatch:
    """
    Local representation of a 3D B-spline patch for linear elasticity.

    This class provides the building blocks to compute linear elasticity operators
    (stiffness matrix, right-hand side, strain, stress, von Mises stress) for a
    single B-spline volume. It is intended **solely as a component of a multipatch
    problem** and is not designed to be used as a standalone solver.

    Typically, IGAPatch objects are managed by a `ProblemIGA` instance, which
    assembles multiple patches, enforces Dirichlet constraints, and solves the
    global linear elasticity problem.

    Features provided by IGAPatch include:
        - computation of the Jacobian and its determinant,
        - gradients of shape functions in physical space,
        - linear elastic stiffness matrix assembly for one patch,
        - evaluation of prescribed surface forces,
        - post-processing quantities such as displacement, strain, stress, and von Mises stress,
        - export of results to visualization tools (e.g., ParaView).

    Attributes
    ----------
    spline : BSpline
        The 3D B-spline volume representing the patch geometry and shape functions.
    ctrl_pts : np.ndarray[np.floating]
        Array of control points of shape (3, n_xi, n_eta, n_zeta) defining the patch geometry.
    xi : np.ndarray[np.floating]
        Isoparametric integration points along the xi direction.
    dxi : np.ndarray[np.floating]
        Corresponding quadrature weights for xi.
    eta : np.ndarray[np.floating]
        Isoparametric integration points along the eta direction.
    deta : np.ndarray[np.floating]
        Corresponding quadrature weights for eta.
    zeta : np.ndarray[np.floating]
        Isoparametric integration points along the zeta direction.
    dzeta : np.ndarray[np.floating]
        Corresponding quadrature weights for zeta.
    F_N : np.ndarray[np.floating]
        Prescribed surface forces, with shape (3 (direction), 2 (side: front/back), 3 (physical components)).
        Used for computing the right-hand side vector.
    H : np.ndarray[np.floating]
        Constitutive matrix in Voigt notation (6x6) for linear isotropic elasticity.
        Defined from `E` and `nu`.

    Notes
    -----
    - All integration points are generated using Gaussian quadrature with a number
      of points determined from the spline degree.
    - This class **does not handle multipatch connectivity or global constraints**.
      Use `ProblemIGA` to assemble and solve multipatch problems.
    - Post-processing quantities can be evaluated at the integration points or exported
      to visualization tools like ParaView.
    """

    spline: BSpline
    ctrl_pts: np.ndarray[np.floating]
    xi: np.ndarray[np.floating]
    dxi: np.ndarray[np.floating]
    eta: np.ndarray[np.floating]
    deta: np.ndarray[np.floating]
    zeta: np.ndarray[np.floating]
    dzeta: np.ndarray[np.floating]
    F_N: np.ndarray[np.floating]
    H: np.ndarray[np.floating]

    def __init__(
        self,
        spline: BSpline,
        ctrl_pts: np.ndarray[np.floating],
        E: float,
        nu: float,
        F_N: np.ndarray[np.floating] = np.zeros((3, 2, 3), dtype="float"),
    ):
        """
        Initialize a local `IGAPatch` for linear elasticity computations.

        This constructor sets up all necessary information for assembling
        the stiffness matrix and right-hand side for a single B-spline patch.
        Note that `IGAPatch` objects are intended to be used as components of
        a `ProblemIGA` multipatch object, and are **not standalone solvers**.

        Parameters
        ----------
        spline : BSpline
            The 3D B-spline volume defining the patch geometry and the shape
            functions. Provides methods to compute basis functions and derivatives.
        ctrl_pts : np.ndarray[np.floating]
            Array of control points defining the patch geometry.
            Shape should be (3, n_xi, n_eta, n_zeta) corresponding to
            (physical_dim, n_ctrl_pts_xi, n_ctrl_pts_eta, n_ctrl_pts_zeta).
        E : float
            Young's modulus of the material.
        nu : float
            Poisson's ratio of the material.
        F_N : np.ndarray[np.floating], optional
            Prescribed surface forces applied on the patch boundaries.
            Shape should be (3 (direction: x, y, z), 2 (side: front/back), 3 (physical components)),
            representing forces at the integration points on each face. Default is zero (no surface forces).

        Attributes set
        ----------------
        H : np.ndarray[np.floating]
            6×6 constitutive matrix in Voigt notation, built from `E` and `nu`.
        xi, eta, zeta : np.ndarray[np.floating]
            Isoparametric coordinates for Gaussian quadrature in each direction.
        dxi, deta, dzeta : np.ndarray[np.floating]
            Quadrature weights associated with the isoparametric coordinates.
        F_N : np.ndarray[np.floating]
            Prescribed surface forces, stored for use in the rhs computation.

        Notes
        -----
        - Quadrature points and weights are automatically generated using
        `spline.gauss_legendre_for_integration` with `degree+1` points.
        - The constitutive matrix `H` assumes linear isotropic elasticity in Voigt notation.
        - This patch only defines the local operators; global assembly and
        enforcement of boundary conditions are handled by `ProblemIGA`.
        """
        self.spline = spline
        self.ctrl_pts = ctrl_pts
        self.H = (
            np.array(
                [
                    [1 - nu, nu, nu, 0, 0, 0],
                    [nu, 1 - nu, nu, 0, 0, 0],
                    [nu, nu, 1 - nu, 0, 0, 0],
                    [0, 0, 0, (0.5 - nu), 0, 0],
                    [0, 0, 0, 0, (0.5 - nu), 0],
                    [0, 0, 0, 0, 0, (0.5 - nu)],
                ],
                dtype="float",
            )
            * E
            / ((1 + nu) * (1 - 2 * nu))
        )
        xi_eta_zeta_pts_pds = self.spline.gauss_legendre_for_integration(
            self.spline.getDegrees() + 1
        )
        ((self.xi, self.eta, self.zeta), (self.dxi, self.deta, self.dzeta)) = (
            xi_eta_zeta_pts_pds
        )
        self.F_N = (
            F_N  # shape (3 (xi, eta, or zeta), 2 (front or back), 3 (vector (x, y, z)))
        )

    def jacobian(
        self, dN_dXI: tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix]
    ) -> tuple[
        np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating]
    ]:
        """
        Compute the Jacobian matrix of the mapping from parametric to physical space,
        its inverse, and its determinant at all integration points.

        The Jacobian describes how the parametric (xi, eta, zeta) coordinates
        are mapped to physical coordinates using the patch's control points.

        Parameters
        ----------
        dN_dXI : tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix]
            Derivatives of the shape functions with respect to the parametric coordinates.
            Tuple of sparse matrices `(dN_dxi, dN_deta, dN_dzeta)`, each of shape
            (n_intg_pts, n_ctrl_pts).

        Returns
        -------
        J : np.ndarray[np.floating]
            Jacobian matrices at each integration point,
            shape `(3, 3, n_intg_pts)` corresponding to (physical_dim, param_dim, n_intg_pts).
        Jinv : np.ndarray[np.floating]
            Inverse of the Jacobian matrices, shape `(3, 3, n_intg_pts)` corresponding to
            (param_dim, physical_dim, n_intg_pts).
        detJ : np.ndarray[np.floating]
            Determinant of the Jacobian at each integration point, shape `(n_intg_pts,)`.

        Notes
        -----
        - `J[:, :, i]` gives the 3×3 Jacobian matrix at the i-th integration point.
        - `Jinv[:, :, i]` is the inverse of that matrix, used to compute derivatives
        with respect to physical coordinates.
        - `detJ[i]` is the determinant of `J[:, :, i]`, used for integration weights.
        """
        J = np.concatenate(
            [(self.ctrl_pts.reshape((3, -1)) @ dN_dxi.T)[:, None] for dN_dxi in dN_dXI],
            axis=1,
        )  # shape(3(phy), 3(param), nb_intg_pts)
        Jinv = np.linalg.inv(J.transpose((2, 0, 1))).transpose(
            (1, 2, 0)
        )  # shape(3(param), 3(phy), nb_intg_pts)
        detJ = np.linalg.det(J.transpose((2, 0, 1)))  # vector of nb_intg_pts values
        return J, Jinv, detJ

    def grad_N(
        self,
        Jinv: np.ndarray[np.floating],
        dN_dXI: tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix],
    ) -> np.ndarray[np.object_]:
        """
        Compute the gradient of the B-spline shape functions with respect to physical coordinates.

        The gradient is obtained by applying the chain rule:
            dN/dX_i = [J⁻¹]_ij @ dN/dXI_j
        for each integration point.

        Parameters
        ----------
        Jinv : np.ndarray[np.floating]
            Inverse Jacobian matrices at all integration points,
            shape `(3, 3, n_intg_pts)` corresponding to (param_dim, physical_dim, n_intg_pts).
        dN_dXI : tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix]
            Derivatives of shape functions with respect to parametric coordinates
            `(dN_dxi, dN_deta, dN_dzeta)`.
            Each sparse matrix has shape `(n_intg_pts, n_ctrl_pts)`.

        Returns
        -------
        dN_dX : np.ndarray[np.object_]
            Gradients of the shape functions with respect to physical coordinates.
            Numpy array of shape `(3,)`, where each element is a `sps.spmatrix` of shape
            `(n_intg_pts, n_ctrl_pts)` corresponding to the derivative along each physical axis.

        Notes
        -----
        - `dN_dX[i]` gives the derivative of all shape functions along the i-th physical direction
        at all integration points.
        - Each sparse matrix can be directly used in the assembly of the stiffness matrix or other
        integrals over the physical domain.
        - The use of `np.object_` array allows each entry to hold a separate sparse matrix.
        """
        dN_dX = np.array(
            [
                sum([dN_dXI[j].multiply(Jinv[j, i, :, None]) for j in range(3)])
                for i in range(3)
            ]
        )
        return dN_dX

    def make_W(self, detJ: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Compute the integration weights in physical space for Gaussian quadrature.

        The integration measure W at each quadrature point combines:
        1. The local volume scaling from the Jacobian determinant `detJ`.
        2. The parametric quadrature weights (`dxi`, `deta`, `dzeta`) in each direction.

        Mathematically:
            W = |detJ| * dxi ⊗ deta ⊗ dzeta
        with * the pointwise multiplication and ⊗ the Kronecker product.

        Parameters
        ----------
        detJ : np.ndarray[np.floating]
            Determinant of the Jacobian at each integration point.
            Shape: (n_intg_pts,)

        Returns
        -------
        W : np.ndarray[np.floating]
            Integration weights in physical space for each quadrature point.
            Shape: (n_intg_pts,)
        """
        W = np.abs(detJ) * np.outer(np.outer(self.dxi, self.deta), self.dzeta).ravel()
        return W

    def stiffness(self) -> sps.spmatrix:
        """
        Compute the linear elasticity stiffness matrix for this IGAPatch.

        This method constructs the 3D linear elasticity stiffness matrix
        using Gaussian quadrature in the parametric space of the B-spline patch.
        The computation follows standard isogeometric analysis (IGA) procedure:

        1. Evaluate the derivatives of the B-spline shape functions with respect to
        the parametric coordinates (dN/dXI).
        2. Compute the Jacobian, its inverse, and determinant at all integration points.
        3. Transform shape function derivatives to the physical space (dN/dX).
        4. Build the strain-displacement matrices B (Bxx, Byy, Bzz, Bxy, Byz, Bxz).
        5. Assemble the stiffness contributions using the material elasticity tensor H
        and the integration weights W derived from the Jacobian.
        6. Sum all contributions to obtain the global stiffness matrix `K`.

        Returns
        -------
        K : sps.spmatrix
            Sparse stiffness matrix of the patch.
            Shape: (3 * n_ctrl_pts, 3 * n_ctrl_pts), in physical coordinates.
        """
        dN_dXI = tuple(self.spline.DN([self.xi, self.eta, self.zeta], k=1).tolist())
        _, Jinv, detJ = self.jacobian(dN_dXI)
        dN_dx, dN_dy, dN_dz = self.grad_N(Jinv, dN_dXI)
        W = self.make_W(detJ)
        # B = sps.bmat([[dN_dx, None, None],
        #               [None, dN_dy, None],
        #               [None, None, dN_dz],
        #               [dN_dy, dN_dx, None],
        #               [None, dN_dz, dN_dy],
        #               [dN_dz, None, dN_dx]])

        Z = sps.coo_matrix(dN_dx.shape, dtype="float")
        Bxx = sps.hstack((dN_dx, Z, Z))
        Byy = sps.hstack((Z, dN_dy, Z))
        Bzz = sps.hstack((Z, Z, dN_dz))
        Bxy = sps.hstack((dN_dy, dN_dx, Z))
        Byz = sps.hstack((Z, dN_dz, dN_dy))
        Bxz = sps.hstack((dN_dz, Z, dN_dx))
        K0 = Bxx.T @ sum((self.H[0, 0] * Bxx, self.H[0, 1] * Byy, self.H[0, 2] * Bzz, self.H[0, 3] * Bxy, self.H[0, 4] * Byz, self.H[0, 5] * Bxz)).T.multiply(W).T  # type: ignore
        K1 = Byy.T @ sum((self.H[1, 0] * Bxx, self.H[1, 1] * Byy, self.H[1, 2] * Bzz, self.H[1, 3] * Bxy, self.H[1, 4] * Byz, self.H[1, 5] * Bxz)).T.multiply(W).T  # type: ignore
        K2 = Bzz.T @ sum((self.H[2, 0] * Bxx, self.H[2, 1] * Byy, self.H[2, 2] * Bzz, self.H[2, 3] * Bxy, self.H[2, 4] * Byz, self.H[2, 5] * Bxz)).T.multiply(W).T  # type: ignore
        K3 = Bxy.T @ sum((self.H[3, 0] * Bxx, self.H[3, 1] * Byy, self.H[3, 2] * Bzz, self.H[3, 3] * Bxy, self.H[3, 4] * Byz, self.H[3, 5] * Bxz)).T.multiply(W).T  # type: ignore
        K4 = Byz.T @ sum((self.H[4, 0] * Bxx, self.H[4, 1] * Byy, self.H[4, 2] * Bzz, self.H[4, 3] * Bxy, self.H[4, 4] * Byz, self.H[4, 5] * Bxz)).T.multiply(W).T  # type: ignore
        K5 = Bxz.T @ sum((self.H[5, 0] * Bxx, self.H[5, 1] * Byy, self.H[5, 2] * Bzz, self.H[5, 3] * Bxy, self.H[5, 4] * Byz, self.H[5, 5] * Bxz)).T.multiply(W).T  # type: ignore
        K: sps.spmatrix = sum((K0, K1, K2, K3, K4, K5))  # type: ignore
        return K

    def rhs(self) -> np.ndarray[np.floating]:
        """
        Compute the right-hand side (load) vector for this `IGAPatch`.

        This method assembles the contribution of the surface forces applied on the
        patch boundary. It uses Gaussian quadrature along the patch faces where
        forces are defined. The procedure is:

        1. Loop over each physical axis (x, y, z) and each side of the patch (0=lower, 1=upper).
        2. Skip faces with zero surface forces.
        3. For each face with a force:
            a. Select the isoparametric coordinate corresponding to the face.
            b. Compute the shape functions `N` and their derivatives along the face.
            c. Compute the surface measure `dS` using the cross product of
                tangent vectors to the surface and parametric quadrature weights.
            d. Integrate the contribution of the shape functions over the surface.
        4. Sum all face contributions into the global right-hand side vector.

        Returns
        -------
        rhs : np.ndarray[np.floating]
            Right-hand side vector for the patch.
            Shape: (3 * n_ctrl_pts,), in physical coordinates.
        """
        rhs = np.zeros(self.ctrl_pts.size, dtype="float")
        XI = [self.xi, self.eta, self.zeta]
        dXI = [self.dxi, self.deta, self.dzeta]
        for axis in range(3):
            for side in range(2):
                if not np.allclose(self.F_N[axis, side], 0):
                    xi = [xi_i for xi_i in XI]
                    if side == 0:
                        xi[axis] = np.array([self.spline.bases[axis].span[0]])
                    else:
                        xi[axis] = np.array([self.spline.bases[axis].span[1]])
                    dxi = [dxi_i for dxi_i in dXI]
                    dxi[axis] = np.ones(1)
                    N = self.spline.DN(xi)
                    k1 = np.eye(3)[(axis + 1) % 3].tolist()
                    k2 = np.eye(3)[(axis + 2) % 3].tolist()
                    dX1 = self.spline(self.ctrl_pts, xi, k=k1).reshape((3, -1))
                    dX2 = self.spline(self.ctrl_pts, xi, k=k2).reshape((3, -1))
                    J = np.linalg.norm(np.cross(dX1, dX2, axis=0), axis=0)
                    dxi = np.outer(dxi[0], np.outer(dxi[1], dxi[2]).ravel()).ravel()
                    dS = J * dxi
                    intg = N.multiply(dS[:, None]).sum(axis=0).A.ravel()  # type: ignore
                    rhs += np.outer(self.F_N[axis, side], intg).ravel()
        return rhs

    def area_border(self, axis: int, front_side: bool) -> float:
        """
        Compute the total surface area of a patch face across a given axis.

        The face is defined by fixing one parametric coordinate (`xi`, `eta`, or `zeta`)
        to either its front or back boundary. The surface area is computed by
        integrating the differential surface measure over the face using
        the Gaussian quadrature points and weights.

        Parameters
        ----------
        axis : int
            Axis across which the face is oriented:
            0 → xi, 1 → eta, 2 → zeta.
        front_side : bool
            If True, select the "front" boundary across the given axis;
            otherwise, select the "back" boundary.

        Returns
        -------
        area : float
            Total surface area of the specified patch face in physical space.
        """
        XI = [self.xi, self.eta, self.zeta]
        xi = [
            (
                XI[i]
                if i != axis
                else np.array(
                    [
                        (
                            self.spline.bases[i].span[1]
                            if front_side
                            else self.spline.bases[i].span[0]
                        )
                    ]
                )
            )
            for i in range(3)
        ]
        dXI = [self.dxi, self.deta, self.dzeta]
        dxi = [dXI[i] for i in range(3) if i != axis]
        dN1, dN2 = [
            self.spline.DN(xi, k=k.tolist())
            for ax, k in enumerate(np.eye(3))
            if ax != axis
        ]
        dSdXI = np.linalg.norm(
            np.cross(
                self.ctrl_pts.reshape((3, -1)) @ dN1.T,
                self.ctrl_pts.reshape((3, -1)) @ dN2.T,
                axis=0,
            ),
            axis=0,
        )
        dXI_area = np.outer(*dxi).ravel()
        area = np.dot(dSdXI, dXI_area)
        return area

    def epsilon(
        self, U: np.ndarray[np.floating], XI: list[np.ndarray[np.floating]]
    ) -> np.ndarray[np.floating]:
        """
        Compute the strain tensor in Voigt notation for the patch.

        The strain tensor is computed as:
            eps = [ε_xx, ε_yy, ε_zz, ε_xy, ε_yz, ε_xz]^T
        using the displacement field `U` and the isoparametric coordinates `XI`.
        Small adjustments are applied at the parametric boundaries to avoid
        evaluation exactly on the control point spans.

        Parameters
        ----------
        U : np.ndarray[np.floating]
            Displacement field of shape (3, nb_ctrl_pts), with the first dimension
            corresponding to the physical coordinates (x, y, z).
        XI : list[np.ndarray[np.floating]]
            List of arrays for the parametric coordinates in each direction
            [xi, eta, zeta].

        Returns
        -------
        eps : np.ndarray[np.floating]
            Strain tensor in Voigt notation, shape (6, n_param_pts),
            where n_param_pts = product of lengths of XI arrays.
        """
        for xi, basis in zip(XI, self.spline.bases):
            xi[np.isclose(xi, basis.span[0], atol=1e-5)] += 1e-5
            xi[np.isclose(xi, basis.span[1], atol=1e-5)] -= 1e-5
        dN_dXI = tuple(self.spline.DN(XI, k=1).tolist())
        _, Jinv, _ = self.jacobian(dN_dXI)  # dXI_dX
        dN_dx, dN_dy, dN_dz = self.grad_N(Jinv, dN_dXI)
        eps = np.array(
            [
                dN_dx @ U[0].ravel(),
                dN_dy @ U[1].ravel(),
                dN_dz @ U[2].ravel(),
                0.5 * (dN_dy @ U[0].ravel() + dN_dx @ U[1].ravel()),
                0.5 * (dN_dz @ U[1].ravel() + dN_dy @ U[2].ravel()),
                0.5 * (dN_dz @ U[0].ravel() + dN_dx @ U[2].ravel()),
            ]
        )
        return eps

    def sigma(self, eps: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Compute the stress tensor in Voigt notation for the patch.

        The stress tensor is computed as:
            sigma = H @ eps
        where H is the linear elasticity matrix of the material.

        Parameters
        ----------
        eps : np.ndarray[np.floating]
            Strain tensor in Voigt notation, shape (6, n_param_pts).

        Returns
        -------
        sig : np.ndarray[np.floating]
            Stress tensor in Voigt notation, shape (6, n_param_pts).
        """
        eps[3:] *= 2
        sig = self.H @ eps
        return sig

    def sigma_eig(self, sig: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Compute the principal stresses (eigenvalues of the stress tensor)
        at the specified parametric points.

        The stress tensor is first converted from Voigt notation to
        full 3x3 tensors before computing eigenvalues.

        Parameters
        ----------
        sig : np.ndarray[np.floating]
            Stress tensor in Voigt notation, shape (6, n_param_pts).

        Returns
        -------
        sig_eig : np.ndarray[np.floating]
            Principal stresses (eigenvalues) of shape (n_param_pts, 3),
            sorted by magnitude.
        """
        sig = sig[np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2]])]
        sig_eig = np.linalg.eigvalsh(sig.transpose((2, 0, 1))).T
        order = np.argsort(np.abs(sig_eig), axis=0)
        sig_eig = sig_eig[order, np.arange(sig_eig.shape[1])[None]]
        return sig_eig

    def von_mises(self, sig_eig: np.ndarray[np.floating]) -> np.ndarray[np.floating]:
        """
        Compute the von Mises equivalent stress at the specified parametric points.

        The von Mises stress is computed from the principal stresses (eigenvalues
        of the stress tensor) as a scalar measure of the deviatoric stress.

        Parameters
        ----------
        sig_eig : np.ndarray[np.floating]
            Principal stresses (eigenvalues) of shape (n_param_pts, 3),
            sorted by magnitude.

        Returns
        -------
        vm : np.ndarray[np.floating]
            Von Mises stress at each parametric point, shape (n_param_pts,).
        """
        vm = (1 / np.sqrt(2)) * np.linalg.norm(
            sig_eig - np.roll(sig_eig, -1, axis=0), axis=0
        )
        return vm

    def save_paraview(
        self,
        U: np.ndarray[np.floating],
        path: str,
        name: str,
        n_eval_per_elem: int = 10,
    ):
        """
        Export the patch data to a ParaView-readable format.

        The saved data includes the displacement field and derived fields
        (strain, stress, principal stresses, von Mises stress) for visualization.

        Parameters
        ----------
        U : np.ndarray[np.floating]
            Displacement field, shape (3, nb_ctrl_pts), with physical coordinates first.
        path : str
            Directory path where the ParaView file will be saved.
        name : str
            Base name of the saved file.
        n_eval_per_elem : int, optional
            Number of evaluation points per element in each parametric direction
            (higher values produce smoother visualizations). Default is 10.
        """
        XI = self.spline.linspace(n_eval_per_elem=n_eval_per_elem)
        fields = self.make_paraview_fields(U, XI)
        self.spline.saveParaview(self.ctrl_pts, path, name, XI=XI, fields=fields)

    def make_paraview_fields(
        self, U: np.ndarray[np.floating], XI: list[np.ndarray[np.floating]]
    ) -> dict[str, np.ndarray[np.floating]]:
        """
        Generate fields for visualization in ParaView from the displacement field.

        This function computes the following fields:
            - Displacement `U`
            - Strain tensor `epsilon` in Voigt notation
            - Stress tensor `sigma` in Voigt notation
            - Principal stresses `sigma_eig`
            - Von Mises stress `von_mises`

        The resulting arrays are reshaped to match the expected ParaView format:
            - The first dimension is for potential time/step (here 1)
            - Subsequent dimensions correspond to physical or Voigt components,
            followed by the parametric grid shape defined by `XI`.

        Parameters
        ----------
        U : np.ndarray[np.floating]
            Displacement field of shape (3, nb_ctrl_pts), with the first dimension
            corresponding to physical coordinates (x, y, z).
        XI : list[np.ndarray[np.floating]]
            List of parametric coordinates for each direction [xi, eta, zeta].
            Determines the evaluation grid for the derived fields.

        Returns
        -------
        fields : dict[str, np.ndarray[np.floating]]
            Dictionary of fields ready for ParaView, with the following shapes:
                - "U" : shape (1, 3, nxi, neta, nzeta)
                - "epsilon" : shape (1, 6, nxi, neta, nzeta)
                - "sigma" : shape (1, 6, nxi, neta, nzeta)
                - "sigma_eig" : shape (1, 3, nxi, neta, nzeta)
                - "von_mises" : shape (1, 1, nxi, neta, nzeta)
            where (nxi, neta, nzeta) = tuple(xi.size for xi in XI)
        """
        u = U.reshape(self.ctrl_pts.shape)
        epsilon = self.epsilon(u, XI)
        sigma = self.sigma(epsilon)
        sigma_eig = self.sigma_eig(sigma)
        von_mises = self.von_mises(sigma_eig)
        XI_shape = tuple(xi.size for xi in XI)
        fields = {
            "U": u[None],
            "epsilon": epsilon.reshape((1, 6, *XI_shape)),
            "sigma": sigma.reshape((1, 6, *XI_shape)),
            "sigma_eig": sigma_eig.reshape((1, 3, *XI_shape)),
            "von_mises": von_mises.reshape((1, 1, *XI_shape)),
        }
        return fields


class IGAPatchDensity(IGAPatch):
    """
    Density-weighted representation of a 3D B-spline patch for linear elasticity.

    This class extends :py:class:`IGAPatch` by allowing a spatially varying
    material density. The density field is defined at the control points and
    interpolated over the parametric domain, scaling the constitutive matrix locally
    for both stiffness assembly and stress evaluation.

    IGAPatchDensity is intended as a component of a multipatch problem and is
    not a standalone solver. Typically, objects of this class are managed by a
    `ProblemIGA` instance that assembles multiple patches, enforces Dirichlet
    constraints, and solves the global elasticity problem.

    Features provided by IGAPatchDensity include:
        - computation of the Jacobian and its determinant,
        - gradients of shape functions in physical space,
        - density-weighted linear elastic stiffness matrix assembly,
        - evaluation of prescribed surface forces,
        - post-processing quantities such as displacement, strain, stress,
        principal stresses, von Mises stress, and local density,
        - export of results to visualization tools (e.g., ParaView).

    Attributes
    ----------
    spline : BSpline
        The 3D B-spline volume representing the patch geometry and shape functions.
    ctrl_pts : np.ndarray[np.floating]
        Array of control points of shape (3, n_xi, n_eta, n_zeta) defining the patch geometry.
    xi : np.ndarray[np.floating]
        Isoparametric integration points along the xi direction.
    dxi : np.ndarray[np.floating]
        Corresponding quadrature weights for xi.
    eta : np.ndarray[np.floating]
        Isoparametric integration points along the eta direction.
    deta : np.ndarray[np.floating]
        Corresponding quadrature weights for eta.
    zeta : np.ndarray[np.floating]
        Isoparametric integration points along the zeta direction.
    dzeta : np.ndarray[np.floating]
        Corresponding quadrature weights for zeta.
    F_N : np.ndarray[np.floating]
        Prescribed surface forces, shape (3 (direction), 2 (side: front/back), 3 (physical components)),
        used for computing the right-hand side vector.
    H : np.ndarray[np.floating]
        Constitutive matrix in Voigt notation (6x6) for linear isotropic elasticity.
        Defined from `E` and `nu`.
    d : np.ndarray[np.floating]
        Density field expressed in the B-spline basis, used to scale the
        constitutive matrix locally. At evaluation, values are clipped to [0, 1].

    Notes
    -----
    - All integration points are generated using Gaussian quadrature with a number
    of points determined from the spline degree.
    - The density field modifies the stiffness and stress at each integration point.
    - This class does not handle multipatch connectivity or global constraints.
    Use `ProblemIGA` to assemble and solve multipatch problems.
    - Post-processing quantities can be evaluated at the integration points or exported
    to visualization tools like ParaView.
    """

    def __init__(
        self,
        spline: BSpline,
        ctrl_pts: np.ndarray[np.floating],
        E: float,
        nu: float,
        d: np.ndarray[np.floating],
        F_N: np.ndarray[np.floating] = np.zeros((3, 2, 3), dtype="float"),
    ):
        """
        Initialize a density-weighted IGA patch.

        This constructor extends :py:meth:`IGAPatch.__init__` by introducing
        a control-point-based density field and by initializing Gaussian
        quadrature points for volumetric integration including the density field.

        The density field is reshaped to match the control point grid.

        Parameters
        ----------
        spline : BSpline
            B-spline volume defining the parametric domain.
        ctrl_pts : np.ndarray[np.floating]
            Control points defining the patch geometry.
        E : float
            Young's modulus of the material.
        nu : float
            Poisson's ratio of the material.
        d : np.ndarray[np.floating]
            Density field expressed in the B-spline basis.
            Evaluated density (N @ d) will be clipped to the range [0, 1].
        F_N : np.ndarray[np.floating], optional
            Surfacic Neumann forces applied on the patch boundaries.
            Shape: (3, 2, 3). Default is zero.
        """
        super().__init__(spline, ctrl_pts, E, nu, F_N)
        xi_eta_zeta_pts_pds = self.spline.gauss_legendre_for_integration(
            (3 * self.spline.getDegrees()) // 2 + 1
        )
        ((self.xi, self.eta, self.zeta), (self.dxi, self.deta, self.dzeta)) = (
            xi_eta_zeta_pts_pds
        )
        self.d = d.reshape(self.ctrl_pts.shape[1:])

    def stiffness(self) -> sps.spmatrix:
        """
        Assemble the density-weighted stiffness matrix.

        This method follows the same formulation as
        :py:meth:`IGAPatch.stiffness`, but the constitutive matrix is scaled
        pointwise by the interpolated density field.

        At each integration point, the elasticity tensor is multiplied by
        the local density value, resulting in a spatially varying stiffness.

        Returns
        -------
        K : sps.spmatrix
            Sparse stiffness matrix of shape
            (3 * nb_ctrl_pts, 3 * nb_ctrl_pts).

        See Also
        --------
        IGAPatch.stiffness :
            Stiffness assembly without density weighting.
        IGAPatchDensity.density :
            Evaluation of the density field at parametric points.
        """
        dN_dXI = tuple(self.spline.DN([self.xi, self.eta, self.zeta], k=1).tolist())
        _, Jinv, detJ = self.jacobian(dN_dXI)
        dN_dx, dN_dy, dN_dz = self.grad_N(Jinv, dN_dXI)
        W = self.make_W(detJ)
        Z = sps.coo_matrix(dN_dx.shape, dtype="float")
        Bxx = sps.hstack((dN_dx, Z, Z))
        Byy = sps.hstack((Z, dN_dy, Z))
        Bzz = sps.hstack((Z, Z, dN_dz))
        Bxy = sps.hstack((dN_dy, dN_dx, Z))
        Byz = sps.hstack((Z, dN_dz, dN_dy))
        Bxz = sps.hstack((dN_dz, Z, dN_dx))
        H = self.H[:, :, None] * self.density([self.xi, self.eta, self.zeta]).reshape(
            (1, 1, -1)
        )
        K0 = Bxx.T @ sum((Bxx.multiply(H[0, 0, :, None]), Byy.multiply(H[0, 1, :, None]), Bzz.multiply(H[0, 2, :, None]), Bxy.multiply(H[0, 3, :, None]), Byz.multiply(H[0, 4, :, None]), Bxz.multiply(H[0, 5, :, None]))).T.multiply(W).T  # type: ignore
        K1 = Byy.T @ sum((Bxx.multiply(H[1, 0, :, None]), Byy.multiply(H[1, 1, :, None]), Bzz.multiply(H[1, 2, :, None]), Bxy.multiply(H[1, 3, :, None]), Byz.multiply(H[1, 4, :, None]), Bxz.multiply(H[1, 5, :, None]))).T.multiply(W).T  # type: ignore
        K2 = Bzz.T @ sum((Bxx.multiply(H[2, 0, :, None]), Byy.multiply(H[2, 1, :, None]), Bzz.multiply(H[2, 2, :, None]), Bxy.multiply(H[2, 3, :, None]), Byz.multiply(H[2, 4, :, None]), Bxz.multiply(H[2, 5, :, None]))).T.multiply(W).T  # type: ignore
        K3 = Bxy.T @ sum((Bxx.multiply(H[3, 0, :, None]), Byy.multiply(H[3, 1, :, None]), Bzz.multiply(H[3, 2, :, None]), Bxy.multiply(H[3, 3, :, None]), Byz.multiply(H[3, 4, :, None]), Bxz.multiply(H[3, 5, :, None]))).T.multiply(W).T  # type: ignore
        K4 = Byz.T @ sum((Bxx.multiply(H[4, 0, :, None]), Byy.multiply(H[4, 1, :, None]), Bzz.multiply(H[4, 2, :, None]), Bxy.multiply(H[4, 3, :, None]), Byz.multiply(H[4, 4, :, None]), Bxz.multiply(H[4, 5, :, None]))).T.multiply(W).T  # type: ignore
        K5 = Bxz.T @ sum((Bxx.multiply(H[5, 0, :, None]), Byy.multiply(H[5, 1, :, None]), Bzz.multiply(H[5, 2, :, None]), Bxy.multiply(H[5, 3, :, None]), Byz.multiply(H[5, 4, :, None]), Bxz.multiply(H[5, 5, :, None]))).T.multiply(W).T  # type: ignore
        K: sps.spmatrix = sum((K0, K1, K2, K3, K4, K5))  # type: ignore
        return K

    def density(self, XI: list[np.ndarray[np.floating]]) -> np.ndarray[np.floating]:
        """
        Evaluate the interpolated density field at parametric points.

        The density is obtained by composition of the evaluated B-spline
        density field with a clipping operator to the interval [0, 1].

        Parameters
        ----------
        XI : list[np.ndarray[np.floating]]
            Parametric coordinates [xi, eta, zeta] at which to evaluate the density.

        Returns
        -------
        density : np.ndarray[np.floating]
            Density values at the parametric points.
            Shape corresponds to the tensor-product grid defined by `XI`.
        """
        return np.clip(self.spline(self.d[None], XI), 0, 1)

    def sigma(
        self, eps: np.ndarray[np.floating], density: np.ndarray[np.floating]
    ) -> np.ndarray[np.floating]:
        """
        Compute the density-weighted stress tensor in Voigt notation.

        The stress is computed from the strain tensor using a linear elastic
        constitutive law, where the constitutive matrix is scaled pointwise
        by the provided density field.

        Voight notation shear strains are internally converted to tensorial
        shear strains before applying the constitutive law.

        Parameters
        ----------
        eps : np.ndarray[np.floating]
            Strain tensor in Voigt notation.
            Shape: (6, n_param_pts).
        density : np.ndarray[np.floating]
            Density values at the same parametric points.
            Shape: (n_param_pts,).

        Returns
        -------
        sig : np.ndarray[np.floating]
            Stress tensor in Voigt notation.
            Shape: (6, n_param_pts).

        See Also
        --------
        IGAPatch.epsilon :
            Strain computation from the displacement field.
        """
        eps[3:] *= 2
        H = self.H[:, :, None] * density.reshape((1, 1, -1))  # (6, 6, nb_param_pts)
        sig = (H * eps[None, :, :]).sum(axis=1)  # sig_ij = H_ikj * eps_kj
        return sig  # (6, nb_param_pts)

    def make_paraview_fields(self, U: np.ndarray[np.floating], XI):
        """
        Generate ParaView fields including density-dependent quantities.

        This method extends :py:meth:`IGAPatch.make_paraview_fields` by adding
        the interpolated density field and by ensuring that stress-related
        quantities are computed using the density-weighted constitutive law.

        The following fields are exported:
            - displacement `U`
            - strain tensor `epsilon`
            - density field `density`
            - stress tensor `sigma`
            - principal stresses `sigma_eig`
            - von Mises stress `von_mises`

        Parameters
        ----------
        U : np.ndarray[np.floating]
            Displacement field of shape (3, nb_ctrl_pts).
        XI : list[np.ndarray[np.floating]]
            Parametric coordinates [xi, eta, zeta] defining the evaluation grid.

        Returns
        -------
        fields : dict[str, np.ndarray[np.floating]]
            Dictionary of fields formatted for ParaView export.
        """
        u = U.reshape(self.ctrl_pts.shape)
        epsilon = self.epsilon(u, XI)
        density = self.density(XI)
        sigma = self.sigma(epsilon, density)
        sigma_eig = self.sigma_eig(sigma)
        von_mises = self.von_mises(sigma_eig)
        XI_shape = tuple(xi.size for xi in XI)
        fields = {
            "U": u[None],
            "epsilon": epsilon.reshape((1, 6, *XI_shape)),
            "density": density.reshape((1, 1, *XI_shape)),
            "sigma": sigma.reshape((1, 6, *XI_shape)),
            "sigma_eig": sigma_eig.reshape((1, 3, *XI_shape)),
            "von_mises": von_mises.reshape((1, 1, *XI_shape)),
        }
        return fields
