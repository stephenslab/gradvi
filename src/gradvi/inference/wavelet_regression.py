"""
Use wavelet matrices for signal processing
"""
# Author: Saikat Banerjee

import numpy as np
from scipy import optimize as sp_optimize
import logging

from ..normal_means import NormalMeansFromPosterior
from ..utils.logs   import CustomLogger
from ..optimize.moving_average import moving_average
#from . import coordinate_descent_step as cd_step
#from . import elbo as elbo_py

from . import GradVIBase
from . import OptimizeResult
from . import opt_utils

# import shrinkage_operator_inverse

#from libgradvi_plr_mrash import plr_mrash as flib_penmrash
#from libgradvi_lbfgs_driver import lbfgsb_driver as flib_lbfgsb_driver


class WaveletRegression(GradVIBase):
    """
    Wavelet Regression minimizes the following objective function
        || y - x ||^2 + rho(W.x) ...... (1)
    given the data {y} and the wavelet matrix W. Note the difference
    with Linear Regression solver, where we solve for
        || y - D.b ||^2 + rho(b) ...... (2)
    We call W the forward Wavelet matrix, which multiplied by the 
    signal gives the coefficients. In the above notation,
    D is the reverse of W and is called the reverse Wavelet matrix
    for the purpose of this documentation. We note that 
        x := D.b
        W.D = 1
        W.x = W.D.b = b
    For solving Wavelet Regression, we only need W and dj, where
    dj is the sum of rows of D'D.


    Parameters
    ----------
    method : str, default = 'L-BFGS-B'
        Which method to use for the gradient descent minimization.
        Should be 'L-BFGS-B' or 'CG'.

    options : dict
        A dictionary of options for gradient descent minimization. 
        Values in this dict will have higher priority over initially
        specified argument. For example, if both `maxiter = 2000`
        and `options = {'maxiter': 1000}` is provided,
        then `maxiter` will be 1000.

    maxiter : int
        Maximum number of iterations allowed for the minimization.

    display_progress: bool, default = False
        Whether to print summary of each iteration during the
        minimization.

    tol : float, default = 1e-9
        Tolerance for termination. When `tol` is specified, the selected
        minimization algorithm sets some relevant solver-specific tolerance(s)
        equal to `tol`. For detailed control, use solver-specific
        `options`.

    invert_method : str, default = 'newton'
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

    debug : bool
        Whether to print debug messages


    Attributes
    ----------
    b_post : array
        Posterior mean of the coefficients.

    prior : object
        Estimate of the prior distribution.

    signal : array
        Estimate of the noised / denoised signal obtained after processing

    Examples
    --------
    """

    def __init__(
        self, method = 'L-BFGS-B', options = None,
        invert_method = None, invert_options = None,
        maxiter = 2000, display_progress = False, tol = 1e-9,
        function_call_py = True, lbfgsb_call_py = True,
        optimize_b = True, optimize_s = True, optimize_w = True,
        logscale_s = False, debug = False):

        self._method       = method.lower()

        # set default options for different minimization solvers
        if options is None:
            self._opts = dict()
        else:
            self._opts = options
        self._opts.setdefault('maxiter', maxiter)
        self._opts.setdefault('disp', display_progress)
        if self._method == 'l-bfgs-b':
            # Function tolerance. stop when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
            self._opts.setdefault('ftol', tol)
            # Gradient tolerance. stop when ``max{|proj g_i | i = 1, ..., n} <= gtol``
            self._opts.setdefault('gtol', tol)
            self._opts.setdefault('maxfun', maxiter * 10)
        elif self._method == 'cg':
            self._opts.setdefault('gtol', tol)
        self._is_f_fortran      = not function_call_py
        self._is_lbfgsb_fortran = not lbfgsb_call_py

        # set which parameters to optimize in the minimization solver
        self._is_opt_list = [optimize_b, optimize_w, optimize_s]
        self._is_logscale_s = logscale_s

        # set debug options
        self._is_debug = debug
        logging_level  = logging.DEBUG if debug else logging.INFO
        self.logger    = CustomLogger(__name__, level = logging_level)

        # set options for inversion of the Posterior Means Operator
        self._invert_method = invert_method
        self._invert_options = invert_options
        if self._invert_options is None: self._invert_options = {}
        return


    def fit(self, Wfwd, y, prior, x_init = None, s2_init = None, dj = None, Wrev = None):
        """
        Fit Wavelet model.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The observed signal.

        prior : gradvi.priors object
            Prior distribution g(w).

        Wfwd : ndarray of shape (p, n_samples)
            The forward Wavelet matrix, such that Wfwd . y = b

        x_init : array-like of shape (n_samples,), optional
            Initial guess for the de-noised signal x.

        s2_init : float, optional
            Initial guess of the standard deviation of white
            Gaussian noise

        dj : ndarray, optional
            Diagonal elements of D'D where D = Wfwd.
            Often, it is more practical to provide dj instead of
            providing Wfwd.

        Wrev : ndarray of shape (n_samples, p)
            The reverse Wavelet matrix, such that Wrev . b = y

        Returns
        -------
        self : object
            Fitted Estimator.
        """

        # dimensions of the problem
        n = y.shape[0]
        k = prior.k

        # This values will not change during the optimization
        self._Wfwd      = Wfwd
        self._Wrev      = Wrev
        self._y         = y
        self._dj        = dj

        # handle dj and Wrev
        if self._dj is None and self._Wrev is None:
            nrow, ncol  = self._Wfwd.shape
            if nrow == ncol:
                try:
                    self._Wrev = np.linalg.inv(self._Wfwd)
                except LinAlgErr as err:
                    raise
            else:
                raise ValueError("The reverse wavelet matrix is not invertible. Please provide dj or Wfwd explicitly.")
        if self._dj is None:
            self._dj    = np.sum(np.square(self._Wrev), axis = 0)
        self._prior     = prior.copy() # do not update the original prior

        # Initialization
        b, s2 = self.initialize_params(x_init, s2_init)
        w = self._prior.wmod_init
        self._init_params = tuple([b, w, s2]) # immutable
        
        # Solver depending on the specified options:
        if self._method == 'L-BFGS-B' and self._is_lbfgsb_fortran:
            self._res = self.fit_fortran()
        else:
            self._res = self.fit_python()
        # end of solver
    
        return



    def fit_fortran(self):
        """
        Run L-BFGS-B from FORTRAN
        """
        raise NotImplementedError("fit_fortran is not implemented yet.")


    def fit_python(self):
        """
        Fit a solver using scipy.optimize.minimize
        """
        # This values will not change during the optimization
        n = self._y.shape[0]
        k = self._prior.k

        # Bounds for optimization
        xbounds = [(None, None) for x in self._init_params[0]]
        wbounds = self._prior.bounds
        s2bound = [(None, None)] if self._is_logscale_s else [(1e-8, None)]
        # bounds can be used only with L-BFGS-B.
        bounds = None
        if self._method == 'l-bfgs-b':
            bounds = opt_utils.combine_optparams([xbounds, wbounds, s2bound],
                                                 self._is_opt_list)
        
        # keep track of the objective function, and other parameters 
        # during every iteration of the solver.
        self._h_path = list()
        self._nclbk  = 0
        self._nfev   = 0
        self._njev   = 0
        xinit = opt_utils.combine_optparams(self._init_params, self._is_opt_list)
        plr_min = sp_optimize.minimize(
                self.get_wavelet_func,
                xinit,
                method = self._method,
                jac = True,
                bounds = bounds,
                callback = self.callback,
                options = self._opts
                )

        # Return values
        #b, wk, logs2 = opt_utils.split_optparams(plr_min.x, self._init_params, self._is_opt_list)
        #x = np.dot(self._Winv, b)
        x, wk, s2 = opt_utils.split_optparams(plr_min.x, self._init_params, self._is_opt_list)
        b = np.dot(self._Wfwd, x)
        if self._is_logscale_s: s2 = np.exp(s2)
        self._prior.update_wmod(wk)
        self._niter = plr_min.nit

        res = OptimizeResult(
                signal = x,
                b_post = b,
                residual_var = s2,
                prior = self._prior,
                success = plr_min.success,
                status  = plr_min.status,
                message = plr_min.message,
                fun     = plr_min.fun,
                grad    = plr_min.jac,
                fitobj = plr_min)

        # Debug logging
        self.logger.info (f"Terminated at iteration   {self._niter}.")
        self.logger.debug(f'Number of iterations:     {self._niter}')
        self.logger.debug(f'Number of callbacks:      {self._nclbk}')
        self.logger.debug(f'Number of function calls: {self._nfev}')
        return res


    def initialize_params(self, x_init, s2_init):
        """
        Initialize the optimization parameters.
        """
        n = self._y.shape[0]
        k = self._prior.k

        if x_init is None: 
            x_init = moving_average(self._y, n = 20)

        if s2_init is None: 
            s2_init = np.var(self._y - x_init)

        s2_init = max(1e-8, s2_init) # cannot be zero or negative
        if self._is_logscale_s:
            return x_init, np.log(s2_init)
        else:
            return x_init, s2_init


    def get_wavelet_func(self, params):
        """
        Calculate the objective function and its gradients with respect to the parameters
        """
        # get coefficients array b, prior parameters wk
        # and residual variance s2 from the solver
        # ===========
        x, wk, s2 = opt_utils.split_optparams(params, self._init_params, self._is_opt_list)
        b = np.dot(self._Wfwd, x)

        if self._is_logscale_s: s2 = np.exp(s2)
        self._prior.update_wmod(wk)

        r = self._y - x
        rTr = np.dot(r, r)

        nm = NormalMeansFromPosterior(
                b, self._prior, s2 / self._dj, 
                scale = s2, d = self._dj,
                method = self._invert_method, **self._invert_options)

        Pb, dPdb, dPdw, dPdsj2 = nm.penalty_operator(jac = True)
        dPds2 = dPdsj2 / self._dj
    
        h     = (0.5 * rTr / s2) + np.sum(Pb)
        dhdx  = - r / s2 + np.dot(self._Wfwd.T, dPdb)
        dhdw  = np.sum(dPdw, axis = 0)
        dhda  = self._prior.wmod_grad(dhdw)
        if self._is_logscale_s:    
            dhds2 = - 0.5 * rTr / s2 + np.sum(dPds2 * s2)
        else:
            dhds2 = - 0.5 * rTr / (s2 * s2) + np.sum(dPds2)
        grad  = opt_utils.combine_optparams([dhdx, dhda, dhds2], self._is_opt_list)
        
        # Book-keeping
        self._nfev += 1
        self._njev += 1
        self._c_fun = h # store current function value
        return h, grad


    def callback(self, params):
        self._nclbk += 1
        # The objective function can be called multiple times in between callbacks
        # We append to paths only during callbacks
        self._h_path.append(self._c_fun)
        # Debug; Yes, I entered callback.
        self.logger.debug(f'Callback iteration {self._nclbk}')
        return


    # =====================================================
    # Attributes specific to wavelet regression
    # =====================================================
    @property
    def residual_var(self):
        return self._res.residual_var


    @property
    def signal(self):
        return self._res.signal

    @property
    def elbo(self):
        x = self._res.fun + 0.5 * np.sum(np.log(self._dj))
        return x
