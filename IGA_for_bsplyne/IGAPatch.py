import numpy as np
import numpy.typing as npt
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
    ctrl_pts : npt.NDArray[np.float_]
        Control points defining the patch geometry.
    E : float
        Young's modulus of the material.
    nu : float
        Poisson's ratio of the material.
    xi : npt.NDArray[np.float_]
        Discretization of the isoparametric coordinate xi.
    dxi : npt.NDArray[np.float_]
        Corresponding weights of xi.
    eta : npt.NDArray[np.float_]
        Discretization of the isoparametric coordinate eta.
    deta : npt.NDArray[np.float_]
        Corresponding weights of eta.
    zeta : npt.NDArray[np.float_]
        Discretization of the isoparametric coordinate zeta.
    dzeta : npt.NDArray[np.float_]
        Corresponding weights of zeta.
    F_N : npt.NDArray[np.float_]
        Surfacic forces applied on the corresponding side of the patch.
    """
    spline: BSpline
    ctrl_pts: npt.NDArray[np.float_]
    E: float
    nu: float
    xi: npt.NDArray[np.float_]
    dxi: npt.NDArray[np.float_]
    eta: npt.NDArray[np.float_]
    deta: npt.NDArray[np.float_]
    zeta: npt.NDArray[np.float_]
    dzeta: npt.NDArray[np.float_]
    F_N: npt.NDArray[np.float_]
    
    def __init__(self, spline: BSpline, ctrl_pts: npt.NDArray[np.float_], E: float, nu: float, F_N: npt.NDArray[np.float_]=np.zeros((3, 2, 3), dtype='float')):
        """
        Initialize the IGAPatch with the given parameters.

        Parameters
        ----------
        spline : BSpline
            B-spline volume used as the patch.
        ctrl_pts : npt.NDArray[np.float_]
            Control points defining the patch geometry.
        E : float
            Young's modulus of the material.
        nu : float
            Poisson's ratio of the material.
        F_N : npt.NDArray[np.float_], optional
            Surfacic forces applied on the corresponding side of the patch.
            Its shape should be (3(param), 2(side), 3(phy)).
            By default np.zeros((3, 2, 3), dtype='float').
        """
        self.spline = spline
        self.ctrl_pts = ctrl_pts
        self.H = np.array([[1 - nu,     nu,     nu,          0,          0,          0], 
                           [    nu, 1 - nu,     nu,          0,          0,          0], 
                           [    nu,     nu, 1 - nu,          0,          0,          0], 
                           [     0,      0,      0, (0.5 - nu),          0,          0], 
                           [     0,      0,      0,          0, (0.5 - nu),          0], 
                           [     0,      0,      0,          0,          0, (0.5 - nu)]], dtype='float')*E/((1 + nu)*(1 - 2*nu))
        xi_eta_zeta_pts_pds = self.spline.gauss_legendre_for_integration(self.spline.getDegrees() + 1)                                                        # TODO integration sampling
        ((self.xi, self.eta, self.zeta), (self.dxi, self.deta, self.dzeta)) = xi_eta_zeta_pts_pds
        self.F_N = F_N # shape (3 (xi, eta, or zeta), 2 (front or back), 3 (vector (x, y, z)))
        
    def jacobian(self, dN_dXI: tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix]) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
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
        J : npt.NDArray[np.float_]
            Jacobian matrix, with shape (3(phy), 3(param), nb_intg_pts).
        Jinv : npt.NDArray[np.float_]
            Inverse of the Jacobian matrix, with shape (3(param), 3(phy), nb_intg_pts).
        detJ : npt.NDArray[np.float_]
            Determinant of the Jacobian matrix, with shape (nb_intg_pts,).
        """
        J = np.concatenate([(self.ctrl_pts.reshape((3, -1))@dN_dxi.T)[:, None] for dN_dxi in dN_dXI], axis=1) # shape(3(phy), 3(param), nb_intg_pts)
        Jinv = np.linalg.inv(J.transpose((2, 0, 1))).transpose((1, 2, 0)) # shape(3(param), 3(phy), nb_intg_pts)
        detJ = np.linalg.det(J.transpose((2, 0, 1))) # vector of nb_intg_pts values
        return J, Jinv, detJ

    def grad_N(self, Jinv: npt.NDArray[np.float_], dN_dXI: tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix]) -> npt.NDArray[np.object_]:
        """
        Calculate the gradient of shape functions with respect to physical space.

        Parameters
        ----------
        Jinv : npt.NDArray[np.float_]
            Inverse of the Jacobian matrix, with shape (3(phy), 3(param), nb_intg_pts).
        dN_dXI : tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix]
            Tuple of sparse matrices representing the derivatives of shape 
            functions with respect to the isoparametric space.

        Returns
        -------
        dN_dX : npt.NDArray[np.object_]
            Gradient of shape functions with respect to physical space.
            Numpy array of shape (3(phy),) containing `sps.spmatrix` objects of shape (nb_intg_pts, nb_ctrl_pts).
        """
        dN_dX = np.array([sum([dN_dXI[j].multiply(Jinv[j, i, :, None]) for j in range(3)]) for i in range(3)])
        return dN_dX
    
    def make_W(self, detJ: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        """
        Compute the measure for intagrating in the physical space as ( abs(det(dX_dXI))*dxi*deta*dzeta ).
        This is the element-wise product of the absolute value of detJ and the Kronecker product 
        of dxi, deta, and dzeta.

        Parameters
        ----------
        detJ : npt.NDArray[np.float_]
            Array containing the determinant values.
            Its shape is (nb_intg_pts,).

        Returns
        -------
        W : npt.NDArray[np.float_]
            Array containing the measure for intagrating in the physical space.
            Its shape is (nb_intg_pts,).
        """
        W = np.abs(detJ)*np.outer(np.outer(self.dxi, self.deta), self.dzeta).ravel()
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
        
        Z = sps.coo_matrix(dN_dx.shape, dtype='float')
        Bxx = sps.hstack((dN_dx, Z, Z))
        Byy = sps.hstack((Z, dN_dy, Z))
        Bzz = sps.hstack((Z, Z, dN_dz))
        Bxy = sps.hstack((dN_dy, dN_dx, Z))
        Byz = sps.hstack((Z, dN_dz, dN_dy))
        Bxz = sps.hstack((dN_dz, Z, dN_dx))
        K0 = Bxx.T @ sum((self.H[0, 0]*Bxx, self.H[0, 1]*Byy, self.H[0, 2]*Bzz, self.H[0, 3]*Bxy, self.H[0, 4]*Byz, self.H[0, 5]*Bxz)).T.multiply(W).T # type: ignore
        K1 = Byy.T @ sum((self.H[1, 0]*Bxx, self.H[1, 1]*Byy, self.H[1, 2]*Bzz, self.H[1, 3]*Bxy, self.H[1, 4]*Byz, self.H[1, 5]*Bxz)).T.multiply(W).T # type: ignore
        K2 = Bzz.T @ sum((self.H[2, 0]*Bxx, self.H[2, 1]*Byy, self.H[2, 2]*Bzz, self.H[2, 3]*Bxy, self.H[2, 4]*Byz, self.H[2, 5]*Bxz)).T.multiply(W).T # type: ignore
        K3 = Bxy.T @ sum((self.H[3, 0]*Bxx, self.H[3, 1]*Byy, self.H[3, 2]*Bzz, self.H[3, 3]*Bxy, self.H[3, 4]*Byz, self.H[3, 5]*Bxz)).T.multiply(W).T # type: ignore
        K4 = Byz.T @ sum((self.H[4, 0]*Bxx, self.H[4, 1]*Byy, self.H[4, 2]*Bzz, self.H[4, 3]*Bxy, self.H[4, 4]*Byz, self.H[4, 5]*Bxz)).T.multiply(W).T # type: ignore
        K5 = Bxz.T @ sum((self.H[5, 0]*Bxx, self.H[5, 1]*Byy, self.H[5, 2]*Bzz, self.H[5, 3]*Bxy, self.H[5, 4]*Byz, self.H[5, 5]*Bxz)).T.multiply(W).T # type: ignore
        K: sps.spmatrix = sum((K0, K1, K2, K3, K4, K5)) # type: ignore
        return K
    
    def rhs(self) -> npt.NDArray[np.float_]:
        """
        Calculate the right-hand side (rhs) vector for the IGAPatch.

        Returns
        -------
        rhs : npt.NDArray[np.float_]
            The computed rhs vector based on the given parameters and operations.
        """
        rhs = np.zeros(self.ctrl_pts.size, dtype='float')
        XI = [self.xi, self.eta, self.zeta]
        dXI = [self.dxi, self.deta, self.dzeta]
        for axis in range(3):
            for side in range(2):
                if not np.allclose(self.F_N[axis, side], 0):
                    xi = [XI[i] if i!=axis else np.array([self.spline.bases[i].span[0] if side==0 else self.spline.bases[i].span[1]]) for i in range(3)]
                    N = self.spline.DN(xi)
                    dN1, dN2 = [self.spline.DN(xi, k=k.tolist()) for ax, k in enumerate(np.eye(3)) if ax!=axis]
                    dS = np.linalg.norm(np.cross(self.ctrl_pts.reshape((3, -1)) @ dN1.T, self.ctrl_pts.reshape((3, -1)) @ dN2.T, axis=0), axis=0)*np.outer(*[dXI[i] for i in range(3) if i!=axis]).ravel()
                    intg = N.multiply(dS[:, None]).sum(axis=0).A.ravel() # type: ignore
                    rhs += np.outer(self.F_N[axis, side], intg).ravel()
        return rhs
    
    def area_border(self, axis: int, front_side: bool) -> float:
        XI = [self.xi, self.eta, self.zeta]
        xi = [XI[i] if i!=axis else np.array([self.spline.bases[i].span[1] if front_side else self.spline.bases[i].span[0]]) for i in range(3)]
        dXI = [self.dxi, self.deta, self.dzeta]
        dxi = [dXI[i] for i in range(3) if i!=axis]
        dN1, dN2 = [self.spline.DN(xi, k=k.tolist()) for ax, k in enumerate(np.eye(3)) if ax!=axis]
        dSdXI = np.linalg.norm(np.cross(self.ctrl_pts.reshape((3, -1)) @ dN1.T, self.ctrl_pts.reshape((3, -1)) @ dN2.T, axis=0), axis=0)
        dXI_area = np.outer(*dxi).ravel()
        area = np.dot(dSdXI, dXI_area)
        return area
    
    def epsilon(self, U: npt.NDArray[np.float_], XI: list[npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
        """
        Calculate the strain tensor (epsilon) for the IGAPatch based on 
        the displacement field `U` and the isoparametric coordinates `XI`.

        Parameters
        ----------
        U : npt.NDArray[np.float_]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).

        XI : list[npt.NDArray[np.float_]]
            List of isoparametric coordinates for each direction xi, eta, and zeta.

        Returns
        -------
        eps : npt.NDArray[np.float_]
            Strain tensor epsilon in voight notation 
            computed as a numpy array of shape (6, nb_param_pts).
        """
        for xi, basis in zip(XI, self.spline.bases):
            xi[np.isclose(xi, basis.span[0], atol=1e-5)] += 1e-5
            xi[np.isclose(xi, basis.span[1], atol=1e-5)] -= 1e-5
        dN_dXI = tuple(self.spline.DN(XI, k=1).tolist())
        _, Jinv, _ = self.jacobian(dN_dXI) # dXI_dX
        dN_dx, dN_dy, dN_dz = self.grad_N(Jinv, dN_dXI)
        eps = np.array([dN_dx @ U[0].ravel(), 
                        dN_dy @ U[1].ravel(), 
                        dN_dz @ U[2].ravel(), 
                        0.5*(dN_dy @ U[0].ravel() + dN_dx @ U[1].ravel()), 
                        0.5*(dN_dz @ U[1].ravel() + dN_dy @ U[2].ravel()), 
                        0.5*(dN_dz @ U[0].ravel() + dN_dx @ U[2].ravel())])
        return eps
    
    def sigma(self, U: npt.NDArray[np.float_], XI: list[npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
        """
        Calculate the stress tensor (sigma) for the IGAPatch based on 
        the displacement field `U` and the isoparametric coordinates `XI`.

        Parameters
        ----------
        U : npt.NDArray[np.float_]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).

        XI : list[npt.NDArray[np.float_]]
            List of isoparametric coordinates for each direction xi, eta, and zeta.

        Returns
        -------
        sig : npt.NDArray[np.float_]
            Stress tensor sigma in voight notation 
            computed as a numpy array of shape (6, nb_param_pts).
        """
        eps = self.epsilon(U, XI)
        eps[3:] *= 2
        sig = self.H @ eps
        return sig
    
    def sigma_eig(self, U: npt.NDArray[np.float_], XI: list[npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
        """
        Calculate the eigenvalues of the stress tensor for the IGAPatch based on 
        the displacement field `U` and the isoparametric coordinates `XI`.

        Parameters
        ----------
        U : npt.NDArray[np.float_]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).

        XI : list[npt.NDArray[np.float_]]
            List of isoparametric coordinates for each direction xi, eta, and zeta.

        Returns
        -------
        sig_eig : npt.NDArray[np.float_]
            Eigenvalues of the stress tensor computed as a numpy array of shape (nb_param_pts, 3).
        """
        sig = self.sigma(U, XI)
        sig = sig[np.array([[0, 3, 5], 
                            [3, 1, 4], 
                            [5, 4, 2]])]
        sig_eig = np.linalg.eigvalsh(sig.transpose((2, 0, 1))).T
        order = np.argsort(np.abs(sig_eig), axis=0)
        sig_eig = sig_eig[order, np.arange(sig_eig.shape[1])[None]]
        return sig_eig
    
    def von_mises(self, U: npt.NDArray[np.float_], XI: list[npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
        """
        Calculate the von Mises stress for the IGAPatch based on the displacement field `U` 
        and the isoparametric coordinates `XI`.

        Parameters
        ----------
        U : npt.NDArray[np.float_]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).

        XI : list[npt.NDArray[np.float_]]
            List of isoparametric coordinates for each direction xi, eta, and zeta.

        Returns
        -------
        vm : npt.NDArray[np.float_]
            Von Mises stress computed as a numpy array of shape (nb_param_pts,).
        """
        sig_eig = self.sigma_eig(U, XI)
        vm = (1/np.sqrt(2))*np.linalg.norm(sig_eig - np.roll(sig_eig, -1, axis=0), axis=0)
        return vm
    
    def save_paraview(self, U: npt.NDArray[np.float_], path: str, name: str, n_eval_per_elem: int=10):
        """
        Save data for visualization in ParaView.

        Parameters
        ----------
        U : npt.NDArray[np.float_]
            Displacement field as a numpy array of shape (3(phy), nb_ctrl_pts).
        path : str
            Path to save the data.
        name : str
            Name of the saved data.
        n_eval_per_elem : int, optional
            Number of evaluations per element, by default 10.
        """
        u = U.reshape((3, -1))[None]
        epsilon = lambda spline, XI: self.epsilon(U, XI).T[None]
        sigma = lambda spline, XI: self.sigma(U, XI).T[None]
        sigma_eig = lambda spline, XI: self.sigma_eig(U, XI).T[None]
        von_mises = lambda spline, XI: self.von_mises(U, XI)[None, :, None]
        fields = {"U": u, "epsilon": epsilon, "sigma": sigma, "sigma_eig": sigma_eig, "von_mises": von_mises}
        self.spline.saveParaview(self.ctrl_pts, path, name, n_eval_per_elem=n_eval_per_elem, fields=fields)

