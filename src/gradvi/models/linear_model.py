
import numpy as np
import logging

from ..utils.decorators import run_once
from ..utils.logs import CustomLogger
from ..normal_means import NormalMeans
from ..normal_means import NormalMeansFromPosterior

from . import coordinate_ascent_step as ca_step
from . import elbo_nmeans as elbo_py

class LinearModel:
    """
    LinearModel calculates the objective function 
    and gradients of a linear model (normal distribution).
        y ~ Normal(Xb, s2)
        b ~ g(w)
    given the data {X, y, b} and the prior distribution family g(w),
    where w are the parameters of the prior.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Training data (explanatory variables)

    y : array of shape (n_samples,)
        Target values (response variables). 

    b : array of shape(n_features,)
        For objtype == "direct", these are the coefficients for the
        regression problem y ~ Normal(Xb, s2).
        For objtype == "reparametrize", these are the 
        'unshrinked-coefficients', such that y ~ Normal(XM(b), s2).
        That is, the coefficients are obtained by applying shrinkage
        operator M on this vector.

    s2 : float
        Residual variance of the linear model.

    prior : gradvi.priors object
        Prior distribution g(w).

    dj : array of shape (n_features,)
        Precalculated norm of each column of X, that is sum_i(X_ij^2).

    objtype : str
        The type of objective to use for the linear model.
        Can be "reparametrize" or "direct.

    invert_method : str, default = 'hybr'
        Type of method to use for inverting the 
        Posterior Means Operator. Should be one of
            
            - 'newton'
            - 'fssi-linear'
            - 'fssi-cubic'
            - 'hybr'

    invert_options : dict
        A dict of options for inverting the Posterior Means Operator.
        Accepts the following generic options:

            tol : float
                Tolerance for termination.
            maxiter : int
                Maximum number of iterations to be performed.
            ngrid : int
                Number of grids for method `fssi`

    v2inv : array of shape (n_features, n_prior)
        Precalculated term for estimating ELBO when using Ash prior.

    """

    def __init__(self, X, y, b, s2, prior, 
            dj = None, 
            objtype = "reparametrize", 
            v2inv = None,
            debug = False,
            invert_method = None,
            invert_options = {}
            ):

        self._X  = X
        self._y  = y
        self._b  = b
        self._s2 = s2
        self._prior   = prior
        self._objtype = objtype
        self._dj      = self.set_xvar(dj)
        self._nm_sj2  = self._s2 / self._dj # this is the variance for the normal means model

        self._n, self._p = self._X.shape

        # set debug options
        self._is_debug = debug
        logging_level  = logging.DEBUG if debug else logging.INFO
        self.logger    = CustomLogger(__name__, level = logging_level)

        # required only for ELBO calculation
        self._v2inv = v2inv

        # required for direct objtype
        self._invert_method = invert_method
        self._invert_opts   = invert_options


    def get_normal_means_model(self):
        nm  = NormalMeans(
                self._b, 
                self._prior, 
                self._nm_sj2, 
                scale = self._s2, 
                d = self._dj)
        return nm



    def get_normal_means_model_from_posterior(self):
        nm  = NormalMeansFromPosterior(
                self._b, 
                self._prior, 
                self._nm_sj2, 
                scale = self._s2, 
                d = self._dj,
                method = self._invert_method,
                **self._invert_opts
                )
        return nm


    def set_xvar(self, dj):
        if dj is None:
            dj = np.sum(np.square(self._X), axis = 0)
        return dj


    def set_coef(self):
        if self._objtype == "reparametrize":
            nm = self.get_normal_means_model()
            self._coef = nm.shrinkage_operator(jac = False)
        elif self._objtype == "direct":
            self._coef = self._b
        return


    def set_coef_inv(self):
        if self._objtype == "reparametrize":
            self._coef_inv = self._b
        elif self._objtype == "direct":
            nm = self.get_normal_means_model_from_posterior()
            self._coef_inv = nm.response
        return


    def solve(self):
        if self._objtype == "reparametrize":
            self.solve_reparametrize()
        elif self._objtype == "direct":
            self.solve_direct()
        return


    def Xdotv(self, v):
        return np.dot(self._X, v)


    def XTdotv(self, v):
        return np.dot(self._X.T, v)


    @run_once
    def solve_reparametrize(self, jac = True):
        """
        Calculates the function and gradients for the linear model
        using the 'reparametrize' objective function.
        """
        self.logger.debug(f"Calculating reparametrized Linear Model objective with {self._prior.prior_type} prior")
        self.logger.debug(f"Residual variance = {self._s2}")

        # Initialize the Normal Means model
        nm = self.get_normal_means_model()

        # shrinkage operator and penalty operator
        # M(b) and rho(b)
        Mb, Mb_bgrad, Mb_wgrad, Mb_sj2grad = nm.shrinkage_operator(jac = True)
        lj, l_bgrad,  l_wgrad,  l_sj2grad  = nm.penalty_operator(jac = True)

        # gradients with respect to s2
        Mb_s2grad = Mb_sj2grad / self._dj
        l_s2grad  = l_sj2grad  / self._dj

        # Objective function
        #r = self._y - np.dot(self._X, Mb)
        r    = self._y - self.Xdotv(Mb)
        rTr  = np.sum(np.square(r))
        #rTX  = np.dot(r.T, self._X)
        rTX  = self.XTdotv(r)
        obj  = (0.5 * rTr / self._s2) + np.sum(lj)
        obj += 0.5 * (self._n - self._p) * np.log(2 * np.pi * self._s2)

        # Gradients
        bgrad  = - (rTX * Mb_bgrad / self._s2) + l_bgrad
        wgrad  = - np.dot(rTX, Mb_wgrad) / self._s2  + np.sum(l_wgrad, axis = 0)
        s2grad = - 0.5 * rTr / (self._s2 * self._s2) \
                 - np.dot(rTX, Mb_s2grad) / self._s2 \
                 + np.sum(l_s2grad) \
                 + 0.5 * (self._n - self._p) / self._s2

        self._objective = obj
        self._bgrad     = bgrad
        self._wgrad     = wgrad
        self._wmod_grad = self._prior.wmod_grad(wgrad)
        self._s2grad    = s2grad
        return


    @run_once
    def solve_direct(self):
        """
        Calculates the function and gradients for the linear model
        using the 'direct' objective function.
        """
        self.logger.debug(f"Calculating Linear Model objective with {self._prior.prior_type} prior")
        self.logger.debug(f"Residual variance = {self._s2}")

        # Initialize the Normal Means model
        nm = self.get_normal_means_model_from_posterior()

        # Penalty operator rho(b)
        lj, l_bgrad,  l_wgrad,  l_sj2grad  = nm.penalty_operator(jac = True)

        # gradients with respect to s2
        l_s2grad  = l_sj2grad  / self._dj

        # Objective function
        #r = self._y - np.dot(self._X, self._b)
        r = self._y - self.Xdotv(self._b)
        rTr  = np.sum(np.square(r))
        #rTX  = np.dot(r.T, self._X)
        rTX  = self.XTdotv(r)
        obj  = (0.5 * rTr / self._s2) + np.sum(lj)
        obj += 0.5 * (self._n - self._p) * np.log(2 * np.pi * self._s2)

        # Gradients
        bgrad  = - (rTX / self._s2) + l_bgrad
        wgrad  = np.sum(l_wgrad, axis = 0)
        s2grad = - 0.5 * rTr / (self._s2 * self._s2) \
                 + np.sum(l_s2grad) \
                 + 0.5 * (self._n - self._p) / self._s2

        self._objective = obj
        self._bgrad     = bgrad
        self._wgrad     = wgrad
        self._wmod_grad = self._prior.wmod_grad(wgrad)
        self._s2grad    = s2grad
        return


    def elbo(self, method = "mrash"):
        sk = self._prior.sk
        wk = self._prior.w
        if method == "mrash":
            x = ca_step.elbo(
                    self._X, self._y, self.coef, self._s2, sk, wk,
                    dj = self._dj, s2inv = self._v2inv)
        elif method == "nmeans":
            x = elbo_py.ash(
                    self._X, self._y, self.coef, self._s2, self._prior,
                    dj = self._dj, phijk = None, mujk = None, varjk = None, 
                    eps = 1e-8)
        return x


    # =====================================================
    # Attributes
    # =====================================================
    @property
    def objective(self):
        self.solve()
        return self._objective


    @property
    def bgrad(self):
        self.solve()
        return self._bgrad


    @property
    def wgrad(self):
        self.solve()
        return self._wgrad


    @property
    def wmod_grad(self):
        self.solve()
        return self._wmod_grad


    @property
    def s2grad(self):
        self.solve()
        return self._s2grad


    @property
    def gradients(self):
        self.solve()
        return self._bgrad, self._wmod_grad, self._s2grad


    @property
    def coef(self):
        self.set_coef()
        return self._coef


    @property
    def coef_inv(self):
        self.set_coef_inv()
        return self._coef_inv
