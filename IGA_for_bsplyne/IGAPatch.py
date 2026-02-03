import functools
import numpy as np
import scipy.sparse as sps

from bsplyne import BSpline


class IGAPatch:
    """
    IGAPatch class to compute linear elasticity operators on 3D B-spline volumes.
    This class computes the stiffness matrix and the right hand side on one B-spline patch.

    Attributes
    ----------
    spline : BSpline
        B-spline volume object used as the patch.
        Contains the methods to compte the shape functions.
    ctrl_pts : np.ndarray[np.floating]
        Control points defining the patch geometry.
    E : float
        Young's modulus of the material.
    nu : float
        Poisson's ratio of the material.
    xi : np.ndarray[np.floating]
        Discretization of the isoparametric coordinate xi.
    dxi : np.ndarray[np.floating]
        Corresponding weights of xi.
    eta : np.ndarray[np.floating]
        Discretization of the isoparametric coordinate eta.
    deta : np.ndarray[np.floating]
        Corresponding weights of eta.
    zeta : np.ndarray[np.floating]
        Discretization of the isoparametric coordinate zeta.
    dzeta : np.ndarray[np.floating]
        Corresponding weights of zeta.
    F_N : np.ndarray[np.floating]
        Surfacic forces applied on the corresponding side of the patch.
    """

    spline: BSpline
    ctrl_pts: np.ndarray[np.floating]
    E: float
    nu: float
    xi: np.ndarray[np.floating]
    dxi: np.ndarray[np.floating]
    eta: np.ndarray[np.floating]
    deta: np.ndarray[np.floating]
    zeta: np.ndarray[np.floating]
    dzeta: np.ndarray[np.floating]
    F_N: np.ndarray[np.floating]

    def __init__(
        self,
        spline: BSpline,
        ctrl_pts: np.ndarray[np.floating],
        E: float,
        nu: float,
        F_N: np.ndarray[np.floating] = np.zeros((3, 2, 3), dtype="float"),
    ):
        """
        Initialize the IGAPatch with the given parameters.

        Parameters
        ----------
        spline : BSpline
            B-spline volume used as the patch.
        ctrl_pts : np.ndarray[np.floating]
            Control points defining the patch geometry.
        E : float
            Young's modulus of the material.
        nu : float
            Poisson's ratio of the material.
        F_N : np.ndarray[np.floating], optional
            Surfacic forces applied on the corresponding side of the patch.
            Its shape should be (3(param), 2(side), 3(phy)).
            By default np.zeros((3, 2, 3), dtype='float').
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
        )  # TODO integration sampling
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
        Calculate the Jacobian matrix (derivative of the mesh wrt its isoparametric space),
        its inverse, and its determinant.

        Parameters
        ----------
        dN_dXI : tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix]
            Tuple of sparse matrices representing the derivatives of shape
            functions wrt the isoparametric space.
            Contains dN_dxi, dN_deta and dN_dzeta.

        Returns
        -------
        J : np.ndarray[np.floating]
            Jacobian matrix, with shape (3(phy), 3(param), nb_intg_pts).
        Jinv : np.ndarray[np.floating]
            Inverse of the Jacobian matrix, with shape (3(param), 3(phy), nb_intg_pts).
        detJ : np.ndarray[np.floating]
            Determinant of the Jacobian matrix, with shape (nb_intg_pts,).
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
        Calculate the gradient of shape functions with respect to physical space.

        Parameters
        ----------
        Jinv : np.ndarray[np.floating]
            Inverse of the Jacobian matrix, with shape (3(phy), 3(param), nb_intg_pts).
        dN_dXI : tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix]
            Tuple of sparse matrices representing the derivatives of shape
            functions with respect to the isoparametric space.

        Returns
        -------
        dN_dX : np.ndarray[np.object_]
            Gradient of shape functions with respect to physical space.
            Numpy array of shape (3(phy),) containing `sps.spmatrix` objects of shape (nb_intg_pts, nb_ctrl_pts).
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
        Compute the measure for intagrating in the physical space as ( abs(det(dX_dXI))*dxi*deta*dzeta ).
        This is the element-wise product of the absolute value of detJ and the Kronecker product
        of dxi, deta, and dzeta.

        Parameters
        ----------
        detJ : np.ndarray[np.floating]
            Array containing the determinant values.
            Its shape is (nb_intg_pts,).

        Returns
        -------
        W : np.ndarray[np.floating]
            Array containing the measure for intagrating in the physical space.
            Its shape is (nb_intg_pts,).
        """
        W = np.abs(detJ) * np.outer(np.outer(self.dxi, self.deta), self.dzeta).ravel()
        return W

    def stiffness(self) -> sps.spmatrix:
        """
        Calculate the stiffness matrix for the IGAPatch.

        Returns
        -------
        K : sps.spmatrix
            Stiffness matrix computed based on the given parameters and operations.
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
        Calculate the right-hand side (rhs) vector for the IGAPatch.

        Returns
        -------
        rhs : np.ndarray[np.floating]
            The computed rhs vector based on the given parameters and operations.
        """
        rhs = np.zeros(self.ctrl_pts.size, dtype="float")
        XI = [self.xi, self.eta, self.zeta]
        dXI = [self.dxi, self.deta, self.dzeta]
        for axis in range(3):
            for side in range(2):
                if not np.allclose(self.F_N[axis, side], 0):
                    xi = [
                        (
                            XI[i]
                            if i != axis
                            else np.array(
                                [
                                    (
                                        self.spline.bases[i].span[0]
                                        if side == 0
                                        else self.spline.bases[i].span[1]
                                    )
                                ]
                            )
                        )
                        for i in range(3)
                    ]
                    N = self.spline.DN(xi)
                    dN1, dN2 = [
                        self.spline.DN(xi, k=k.tolist())
                        for ax, k in enumerate(np.eye(3))
                        if ax != axis
                    ]
                    dS = (
                        np.linalg.norm(
                            np.cross(
                                self.ctrl_pts.reshape((3, -1)) @ dN1.T,
                                self.ctrl_pts.reshape((3, -1)) @ dN2.T,
                                axis=0,
                            ),
                            axis=0,
                        )
                        * np.outer(*[dXI[i] for i in range(3) if i != axis]).ravel()
                    )
                    intg = N.multiply(dS[:, None]).sum(axis=0).A.ravel()  # type: ignore
                    rhs += np.outer(self.F_N[axis, side], intg).ravel()
        return rhs

    def area_border(self, axis: int, front_side: bool) -> float:
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
        Calculate the strain tensor (epsilon) for the IGAPatch based on
        the displacement field `U` and the isoparametric coordinates `XI`.

        Parameters
        ----------
        U : np.ndarray[np.floating]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).

        XI : list[np.ndarray[np.floating]]
            List of isoparametric coordinates for each direction xi, eta, and zeta.

        Returns
        -------
        eps : np.ndarray[np.floating]
            Strain tensor epsilon in voight notation
            computed as a numpy array of shape (6, nb_param_pts).
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

    def sigma(
        self, U: np.ndarray[np.floating], XI: list[np.ndarray[np.floating]]
    ) -> np.ndarray[np.floating]:
        """
        Calculate the stress tensor (sigma) for the IGAPatch based on
        the displacement field `U` and the isoparametric coordinates `XI`.

        Parameters
        ----------
        U : np.ndarray[np.floating]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).

        XI : list[np.ndarray[np.floating]]
            List of isoparametric coordinates for each direction xi, eta, and zeta.

        Returns
        -------
        sig : np.ndarray[np.floating]
            Stress tensor sigma in voight notation
            computed as a numpy array of shape (6, nb_param_pts).
        """
        eps = self.epsilon(U, XI)
        eps[3:] *= 2
        sig = self.H @ eps
        return sig

    def sigma_eig(
        self, U: np.ndarray[np.floating], XI: list[np.ndarray[np.floating]]
    ) -> np.ndarray[np.floating]:
        """
        Calculate the eigenvalues of the stress tensor for the IGAPatch based on
        the displacement field `U` and the isoparametric coordinates `XI`.

        Parameters
        ----------
        U : np.ndarray[np.floating]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).

        XI : list[np.ndarray[np.floating]]
            List of isoparametric coordinates for each direction xi, eta, and zeta.

        Returns
        -------
        sig_eig : np.ndarray[np.floating]
            Eigenvalues of the stress tensor computed as a numpy array of shape (nb_param_pts, 3).
        """
        sig = self.sigma(U, XI)
        sig = sig[np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2]])]
        sig_eig = np.linalg.eigvalsh(sig.transpose((2, 0, 1))).T
        order = np.argsort(np.abs(sig_eig), axis=0)
        sig_eig = sig_eig[order, np.arange(sig_eig.shape[1])[None]]
        return sig_eig

    def von_mises(
        self, U: np.ndarray[np.floating], XI: list[np.ndarray[np.floating]]
    ) -> np.ndarray[np.floating]:
        """
        Calculate the von Mises stress for the IGAPatch based on the displacement field `U`
        and the isoparametric coordinates `XI`.

        Parameters
        ----------
        U : np.ndarray[np.floating]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).

        XI : list[np.ndarray[np.floating]]
            List of isoparametric coordinates for each direction xi, eta, and zeta.

        Returns
        -------
        vm : np.ndarray[np.floating]
            Von Mises stress computed as a numpy array of shape (nb_param_pts,).
        """
        sig_eig = self.sigma_eig(U, XI)
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
        Save data for visualization in ParaView.

        Parameters
        ----------
        U : np.ndarray[np.floating]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).
        path : str
            Path to save the data.
        name : str
            Name of the saved data.
        n_eval_per_elem : int, optional
            Number of evaluations per element, by default 10.
        """
        fields = self.make_paraview_fields(U)
        self.spline.saveParaview(
            self.ctrl_pts, path, name, n_eval_per_elem=n_eval_per_elem, fields=fields
        )

    def make_paraview_fields(self, U: np.ndarray[np.floating]):
        """
        Make data fields for visualization in ParaView.

        Parameters
        ----------
        U : np.ndarray[np.floating]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).
        """
        u = U.reshape(self.ctrl_pts.shape)[None]
        epsilon = functools.partial(compute_epsilon, self, u[0])
        sigma = functools.partial(compute_sigma, self, u[0])
        sigma_eig = functools.partial(compute_sigma_eig, self, u[0])
        von_mises = functools.partial(compute_von_mises, self, u[0])
        fields = {
            "U": u,
            "epsilon": epsilon,
            "sigma": sigma,
            "sigma_eig": sigma_eig,
            "von_mises": von_mises,
        }
        return fields


def compute_epsilon(patch_obj, u_patch, spline, XI):
    return patch_obj.epsilon(u_patch, XI).T[None]


def compute_sigma(patch_obj, u_patch, spline, XI):
    return patch_obj.sigma(u_patch, XI).T[None]


def compute_sigma_eig(patch_obj, u_patch, spline, XI):
    return patch_obj.sigma_eig(u_patch, XI).T[None]


def compute_von_mises(patch_obj, u_patch, spline, XI):
    return patch_obj.von_mises(u_patch, XI)[None, :, None]


class IGAPatchDensity(IGAPatch):
    def __init__(
        self,
        spline: BSpline,
        ctrl_pts: np.ndarray[np.floating],
        E: float,
        nu: float,
        d: np.ndarray[np.floating],
        F_N: np.ndarray[np.floating] = np.zeros((3, 2, 3), dtype="float"),
    ):
        super().__init__(spline, ctrl_pts, E, nu, F_N)
        xi_eta_zeta_pts_pds = self.spline.gauss_legendre_for_integration(
            (3 * self.spline.getDegrees()) // 2 + 1
        )  # TODO integration sampling
        ((self.xi, self.eta, self.zeta), (self.dxi, self.deta, self.dzeta)) = (
            xi_eta_zeta_pts_pds
        )
        self.d = d.reshape(self.ctrl_pts.shape[1:])

    def stiffness(self) -> sps.spmatrix:
        """
        Calculate the stiffness matrix for the IGAPatch.

        Returns
        -------
        K : sps.spmatrix
            Stiffness matrix computed based on the given parameters and operations.
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
        H = (
            self.H[:, :, None]
            * np.clip(
                self.spline(self.d[None], [self.xi, self.eta, self.zeta]).ravel(), 0, 1
            )[None, None, :]
        )
        K0 = Bxx.T @ sum((Bxx.multiply(H[0, 0, :, None]), Byy.multiply(H[0, 1, :, None]), Bzz.multiply(H[0, 2, :, None]), Bxy.multiply(H[0, 3, :, None]), Byz.multiply(H[0, 4, :, None]), Bxz.multiply(H[0, 5, :, None]))).T.multiply(W).T  # type: ignore
        K1 = Byy.T @ sum((Bxx.multiply(H[1, 0, :, None]), Byy.multiply(H[1, 1, :, None]), Bzz.multiply(H[1, 2, :, None]), Bxy.multiply(H[1, 3, :, None]), Byz.multiply(H[1, 4, :, None]), Bxz.multiply(H[1, 5, :, None]))).T.multiply(W).T  # type: ignore
        K2 = Bzz.T @ sum((Bxx.multiply(H[2, 0, :, None]), Byy.multiply(H[2, 1, :, None]), Bzz.multiply(H[2, 2, :, None]), Bxy.multiply(H[2, 3, :, None]), Byz.multiply(H[2, 4, :, None]), Bxz.multiply(H[2, 5, :, None]))).T.multiply(W).T  # type: ignore
        K3 = Bxy.T @ sum((Bxx.multiply(H[3, 0, :, None]), Byy.multiply(H[3, 1, :, None]), Bzz.multiply(H[3, 2, :, None]), Bxy.multiply(H[3, 3, :, None]), Byz.multiply(H[3, 4, :, None]), Bxz.multiply(H[3, 5, :, None]))).T.multiply(W).T  # type: ignore
        K4 = Byz.T @ sum((Bxx.multiply(H[4, 0, :, None]), Byy.multiply(H[4, 1, :, None]), Bzz.multiply(H[4, 2, :, None]), Bxy.multiply(H[4, 3, :, None]), Byz.multiply(H[4, 4, :, None]), Bxz.multiply(H[4, 5, :, None]))).T.multiply(W).T  # type: ignore
        K5 = Bxz.T @ sum((Bxx.multiply(H[5, 0, :, None]), Byy.multiply(H[5, 1, :, None]), Bzz.multiply(H[5, 2, :, None]), Bxy.multiply(H[5, 3, :, None]), Byz.multiply(H[5, 4, :, None]), Bxz.multiply(H[5, 5, :, None]))).T.multiply(W).T  # type: ignore
        K: sps.spmatrix = sum((K0, K1, K2, K3, K4, K5))  # type: ignore
        return K

    def sigma(
        self, U: np.ndarray[np.floating], XI: list[np.ndarray[np.floating]]
    ) -> np.ndarray[np.floating]:
        """
        Calculate the stress tensor (sigma) for the IGAPatch based on
        the displacement field `U` and the isoparametric coordinates `XI`.

        Parameters
        ----------
        U : np.ndarray[np.floating]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).

        XI : list[np.ndarray[np.floating]]
            List of isoparametric coordinates for each direction xi, eta, and zeta.

        Returns
        -------
        sig : np.ndarray[np.floating]
            Stress tensor sigma in voight notation
            computed as a numpy array of shape (6, nb_param_pts).
        """
        eps = self.epsilon(U, XI)  # (6, nb_param_pts)
        eps[3:] *= 2
        H = (
            self.H[:, :, None]
            * np.clip(self.spline(self.d[None], XI).ravel(), 0, 1)[None, None, :]
        )  # (6, 6, nb_param_pts)
        sig = (H * eps[None, :, :]).sum(axis=1)  # sig_ij = H_ikj * eps_kj
        return sig  # (6, nb_param_pts)

    def density(self, XI: list[np.ndarray[np.floating]]) -> np.ndarray[np.floating]:
        return np.clip(self.spline(self.d[None], XI).ravel(), 0, 1)[None]

    def make_paraview_fields(self, U: np.ndarray[np.floating]):
        """
        Make data fields for visualization in ParaView.

        Parameters
        ----------
        U : np.ndarray[np.floating]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).
        """
        u = U.reshape(self.ctrl_pts.shape)[None]
        density = functools.partial(compute_density, self)
        epsilon = functools.partial(compute_epsilon, self, u[0])
        sigma = functools.partial(compute_sigma, self, u[0])
        sigma_eig = functools.partial(compute_sigma_eig, self, u[0])
        von_mises = functools.partial(compute_von_mises, self, u[0])
        fields = {
            "U": u,
            "density": density,
            "epsilon": epsilon,
            "sigma": sigma,
            "sigma_eig": sigma_eig,
            "von_mises": von_mises,
        }
        return fields


def compute_density(patch_obj, spline, XI):
    return patch_obj.density(XI).T[None]
