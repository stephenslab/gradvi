import numpy as np

from . import LinearRegression
from ..models import TrendfilteringModel
from ..models import basis_matrix as gvbm
from ..optimize import moving_average as gvma

class Trendfiltering(LinearRegression):

    def __init__(
        self, method = 'L-BFGS-B', obj = 'reparametrize',
        fit_intercept = False, options = None, standardize = True,
        invert_method = None, invert_options = None,
        maxiter = 2000, display_progress = False, tol = 1e-9,
        get_elbo = False, function_call_py = True, lbfgsb_call_py = True,
        optimize_b = True, optimize_s = True, optimize_w = True,
        standardize_basis = False, scale_basis = False,
        hessian_callable = None, debug = False):

        # intercept is controlled from this Class.
        # Hence it is set to false for Parent class
        super().__init__(method = method, obj = obj, 
            fit_intercept = False, options = options,
            invert_method = invert_method, invert_options = invert_options,
            maxiter = maxiter, display_progress = display_progress, tol = tol,
            get_elbo = get_elbo, function_call_py = True, lbfgsb_call_py = True,
            optimize_b = optimize_b, optimize_s = optimize_s, optimize_w = optimize_w, 
            hessian_callable = hessian_callable, debug = debug)

        self._tf_standardize_basis = standardize_basis
        self._tf_standardize_y = standardize
        self._tf_scale_basis = scale_basis

        return


    def fit(self, y, degree, prior, y_init = None, s2_init = None):

        n = y.shape[0]

        # unique variables for Trendfiltering class
        self._tf_intercept = np.mean(y) if self._tf_standardize_y else 0.0
        self._tf_ystd = np.std(y - self._tf_intercept) if self._tf_standardize_y else 1.0
        self._tf_degree = degree

        # Trendfiltering inverse matrix is required if the TF matrix is scaled or standardized.
        self._tf_X = None
        self._tf_fstd = None
        self._tf_floc = None
        if self._tf_standardize_basis or self._tf_scale_basis:
            self._tf_X = gvbm.trendfiltering(n, degree)
            tf_Xinv = gvbm.trendfiltering_inverse(n, degree) # Xinv is only needed for initializing, not for the model.

        # scale y to mean zero and variance 1
        yscale = (y - self._tf_intercept) / self._tf_ystd

        if y_init is None:
            y_init  = gvma.moving_average(yscale)
        else:
            y_init = (y_init - self._tf_intercept) / self._tf_ystd

        if s2_init is None:
            s2_init = np.var(yscale - y_init)
        else:
            s2_init = s2_init / np.square(self._tf_ystd)


        # The input X (and dj), and b_init depends on scaling of TF basis.
        if self._tf_standardize_basis:
            # use scaled TF basis, but save the scaling factors, std and loc
            self._X, self._tf_fstd, self._tf_floc = gvbm.center_and_scale_tfbasis(self._tf_X)
            Xinv = tf_Xinv * self._tf_fstd.reshape(-1,1)
            Xinv[0, :] = 1 / n
            b_init = np.dot(Xinv, y_init)
            self._dj = None
        elif self._tf_scale_basis:
            # do not standardize TF basis, but scale it such that sum(dj) = N
            self._tf_fstd = np.sqrt(np.sum(np.square(self._tf_X)) / n)
            self._X = self._tf_X / self._tf_fstd
            b_init = np.dot(tf_Xinv, y_init) * self._tf_fstd
            self._dj = None
        else:
            #self._X = self._tf_X.copy()
            #b_init = np.dot(tf_Xinv, y_init)
            self._X = None
            b_init = gvbm.discrete_difference(y_init, self._tf_degree)
            self._dj = gvbm.get_dj_lowmem(n, self._tf_degree)

        super().fit(self._X, yscale, prior, b_init = b_init, t_init = None, s2_init = s2_init, dj = self._dj)

        return


    def get_new_model(self, b, s2, prior, v2inv = None):
        model = TrendfilteringModel(
                    self._X, self._y, b, s2, prior,
                    dj = self._dj,
                    objtype = self._objtype,
                    v2inv = v2inv,
                    debug = self._is_debug,
                    invert_method = self._invert_method,
                    invert_options = self._invert_options,
                    tf_degree = self._tf_degree,
                    tfbasis_matrix = self._tf_X, # this is always unscaled
                    tfbasis_scale_factors = (self._tf_fstd, self._tf_floc),
                    standardize_basis = self._tf_standardize_basis,
                    scale_basis = self._tf_scale_basis
                    )
        return model


    # =====================================================
    # Attributes specific to trendfiltering
    # =====================================================
    @property
    def residual_var(self):
        return self._res.residual_var * self._tf_ystd


    @property
    def ypred(self):
        return np.dot(self._X, self.coef) * self._tf_ystd + self._tf_intercept
