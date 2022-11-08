import numpy as np

from . import LinearRegression
from ..models import TrendfilteringModel
from ..models import basis_matrix as gvbm
from ..optimize import moving_average as gvma

class Trendfiltering(LinearRegression):

    def __init__(
        self, method = 'L-BFGS-B', obj = 'reparametrize',
        fit_intercept = False, options = None,
        invert_method = None, invert_options = None,
        maxiter = 2000, display_progress = False, tol = 1e-9,
        get_elbo = False, function_call_py = True, lbfgsb_call_py = True,
        optimize_b = True, optimize_s = True, optimize_w = True,
        scale_tfbasis = True,
        debug = False):

        super().__init__(method = method, obj = obj, 
            fit_intercept = fit_intercept, options = options,
            invert_method = invert_method, invert_options = invert_options,
            maxiter = maxiter, display_progress = display_progress, tol = tol,
            get_elbo = get_elbo, function_call_py = True, lbfgsb_call_py = True,
            optimize_b = optimize_b, optimize_s = optimize_s, optimize_w = optimize_w, 
            debug = debug)

        self._scale_tfbasis = scale_tfbasis

        return


    def fit(self, y, degree, prior, y_init = None, s2_init = None):

        n = y.shape[0]
        if y_init is None:  y_init  = gvma.moving_average(y)
        if s2_init is None: s2_init = np.var(y - y_init)

        # unique variables for Trendfiltering class
        self._tf_degree = degree
        self._tf_X = gvbm.trendfiltering(n, degree)
        self._tf_fstd = None
        self._tf_floc = None

        tf_Xinv = gvbm.trendfiltering_inverse(n, degree) # Xinv is only needed for initializing, not for the model.

        # The input X (and dj), and b_init depends on scaling of TF basis.
        if self._scale_tfbasis:
            # use scaled TF basis, but save the scaling factors, std and loc
            self._X, self._tf_fstd, self._tf_floc = gvbm.center_and_scale_tfbasis(self._tf_X)
            Xinv = tf_Xinv * self._tf_fstd.reshape(-1,1)
            Xinv[0, :] = 1 / n
            b_init = np.dot(Xinv, y_init)
        else:
            self._X = self._tf_X.copy()
            b_init = np.dot(tf_Xinv, y_init) 


        super().fit(self._X, y, prior, b_init = b_init, t_init = None, s2_init = s2_init)

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
                    scale_tfbasis = self._scale_tfbasis
                    )
        return model


    @property
    def ypred(self):
        return np.dot(self._X, self.coef)
