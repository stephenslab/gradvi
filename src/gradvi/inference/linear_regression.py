"""
Unified interfaces to linear regression algorithms
"""
# Author: Saikat Banerjee

import numpy as np
from scipy import optimize as sp_optimize
import logging
import time

from ..models import LinearModel
from ..normal_means import NormalMeans, NormalMeansFromPosterior
from ..utils.logs   import CustomLogger
from ..utils import project
#from . import coordinate_descent_step as cd_step
#from . import elbo as elbo_py

from . import GradVIBase
from . import OptimizeResult
from . import opt_utils

# import shrinkage_operator_inverse

#from libgradvi_plr_mrash import plr_mrash as flib_penmrash
#from libgradvi_lbfgs_driver import lbfgsb_driver as flib_lbfgsb_driver


class LinearRegression(GradVIBase):
    """
    LinearRegression fits a linear model (normal distribution)
        y ~ Normal(Xb, s2)
        b ~ g(w)
    given the data {X, y} and the prior distribution family g(w),
    where w are the parameters of the prior.
    Mean field variational empirical Bayes is designed
    to estimate s2, w and the posterior mean of b.
    
    We have implemented two strategies for the objective function:

        - 'direct' uses the objective function :math:`h(b, w, s2)`.
        It requires calculating :math:`t = M^{-1}(b)`.
        - 'parametrize' uses a reparametrization of the objective 
        function, :math:`\tilde{h}(t, w, s2)`.

    Parameters
    ----------
    method : str, default = 'L-BFGS-B'
        Which method to use for the gradient descent minimization.
        Should be 'L-BFGS-B' or 'CG'.

    obj : str, default = 'parametrize'
        Which objective function to use for minimization. Should be
        'direct' or 'parametrize'.

    fit_intercept : bool, default = True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

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

    get_elbo : bool, default = False
        Whether to calculate the ELBO at each step of the minimization.
        Our objective function(s) are not equal to the ELBO,
        so the ELBO is not guranteed to decrease during the
        iterations of the solver.

    debug : bool
        Whether to print debug messages


    Attributes
    ----------
    coef : array
        Posterior mean of the coefficients.

    prior : object
        Estimate of the prior distribution.

    Examples
    --------
    """

    def __init__(
        self, method = 'L-BFGS-B', obj = 'reparametrize',
        fit_intercept = True, options = None, 
        invert_method = None, invert_options = None,
        maxiter = 2000, display_progress = False, tol = 1e-9,
        get_elbo = False, function_call_py = True, lbfgsb_call_py = True,
        optimize_b = True, optimize_s = True, optimize_w = True,
        debug = False):

        self._is_intercept = fit_intercept
        self._method       = method.lower()
        self._objtype      = obj.lower()
        self._is_elbo_calc = get_elbo

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

        # set logger for this class
        self._is_debug = debug
        self.logger    = CustomLogger(__name__, is_debug = self._is_debug)

        # set options for inversion of the Posterior Means Operator
        self._invert_method  = invert_method
        self._invert_options = invert_options
        if self._invert_options is None: self._invert_options = {}
        return


    def fit(self, X, y, prior, b_init = None, t_init = None, s2_init = None, dj = None):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        prior : gradvi.priors object
            Prior distribution g(w).

        b_init : array-like of shape (n_features,)
            Initial guess of the coefficients b.

        t_init : array-like of shape (n_features,)
            Initial guess of M^{-1}(b).

        s2_init : float
            Initial guess of the residual variance.

        Returns
        -------
        self : object
            Fitted Estimator.
        """

        # dimensions of the problem
        n, p = X.shape
        k    = prior.k

        # This values will not change during the optimization
        self._X         = X
        self._intercept = np.mean(y) if self._is_intercept else 0
        self._y         = y - self._intercept
        self._dj        = np.sum(np.square(self._X), axis = 0) if dj is None else dj
        self._prior     = prior.copy() # do not update the original prior

        # Initialization
        b, s2 = self.initialize_params(b_init, t_init, s2_init)
        w = self._prior.wmod_init
        #logs2 = np.log(s2)
        #self._init_params = tuple([b, w, logs2]) # immutable
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
        return


    def fit_python(self):
        """
        Fit a solver using scipy.optimize.minimize
        """
        # This values will not change during the optimization
        n, p = self._X.shape
        k    = self._prior.k

        ## Precompute v2inv for ELBO calculation when using ash prior
        self._v2inv = None
        if self._is_elbo_calc:
            sk = self._prior.sk
            self._v2inv = np.zeros((p, k))
            self._v2inv[:, 1:] = 1 / (self._dj.reshape(p, 1) + 1 / np.square(sk[1:]).reshape(1, k - 1))

        # Bounds for optimization
        bbounds = [(None, None) for x in self._init_params[0]]
        wbounds = self._prior.bounds
        s2bound = [(1e-8, None)]
        #s2bound = [(None, None)]
        # bounds can be used only with L-BFGS-B.
        bounds = None
        if self._method == 'l-bfgs-b':
            bounds = opt_utils.combine_optparams([bbounds, wbounds, s2bound],
                                                 self._is_opt_list)
        
        # keep track of the objective function, and other parameters 
        # during every iteration of the solver.
        self._h_path     = list()
        self._elbo_path  = list()
        self._nclbk = 0
        self._nfev  = 0
        self._njev  = 0
        xinit = opt_utils.combine_optparams(self._init_params, self._is_opt_list)
        opt_start_time = time.time()
        plr_min = sp_optimize.minimize(
                self.get_model_func,
                xinit,
                method = self._method,
                jac = True,
                bounds = bounds,
                callback = self.callback,
                options = self._opts
                )
        opt_end_time = time.time()

        # Return values
        #b, wk, logs2 = opt_utils.split_optparams(plr_min.x, self._init_params, self._is_opt_list)
        #s2 = np.exp(logs2)
        b, wk, s2 = opt_utils.split_optparams(plr_min.x, self._init_params, self._is_opt_list)
        self._prior.update_wmod(wk)
        self._niter = plr_min.nit
        model = self.get_new_model(b, s2, self._prior)

        res = OptimizeResult(
                b_post  = model.coef,
                b_inv   = model.coef_inv,
                residual_var = s2,
                prior   = self._prior,
                optim_time = opt_end_time - opt_start_time,
                success = plr_min.success,
                status  = plr_min.status,
                message = plr_min.message,
                fun     = plr_min.fun,
                grad    = plr_min.jac,
                fitobj  = plr_min)

        # Debug logging
        self.logger.debug(f"Terminated at iteration {self._niter}.")
        self.logger.debug(f'Number of iterations:     {self._niter}')
        self.logger.debug(f'Number of callbacks:      {self._nclbk}')
        self.logger.debug(f'Number of function calls: {self._nfev}')
        return res


    def initialize_params(self, b_init, t_init, s2_init):
        """
        Initialize the optimization parameters.

        The initial values of theta_init depends on the minimization problem.
        
        +--------+--------+--------+----------------------------+
        | b_init | t_init | theta_init for minimization problem |
        +        +        +-------------------------------------+
        |        |        | Direct | Reparametrize              |
        +--------+--------+--------+----------------------------+
        | None   | None   | zero   | zero                       |
        +--------+--------+--------+----------------------------+
        | None   | t0     | M(t0)  | t0                         |
        +--------+--------+--------+----------------------------+
        | b0     | None   | b0     | Minv(b0)                   |
        +--------+--------+--------+----------------------------+
        | b0     | t0     | b0     | Minv(b0)                   |
        +--------+--------+--------+----------------------------+
        """
        n, p = self._X.shape
        k    = self._prior.k

        # we want the initial variance to be less than the true variance
        # ad-hoc division by 0.
        # TO-DO: work on a better initialization of s2
        var_init   = np.var(self._y) / 10. if s2_init is None else s2_init
        theta_init = np.zeros(p)
        coef_init  = np.zeros(p)
        if b_init is None:
            if t_init is not None:
                if s2_init is None: var_init   = np.var(self._y - np.dot(self._X, t_init)) / 10.
                theta_init = t_init.copy()
                if self._objtype == "direct":
                    #lm = self.get_new_model(t_init, s2_init, self._prior)
                    nm = NormalMeans(
                            t_init, self._prior, var_init / self._dj,
                            scale = s2_init, d = self._dj)
                    coef_init = nm.shrinkage_operator(jac = False)
                    #coef_init = lm.coef
        else:
            if s2_init is None: var_init   = np.var(self._y - np.dot(self._X, b_init))
            if self._objtype == "reparametrize":
                # Get inverse if using parametrized objective
                nm = NormalMeansFromPosterior(
                        b_init, self._prior, var_init / self._dj,
                        scale = var_init, d = self._dj)
                theta_init = nm.response
            coef_init  = b_init.copy()

        # Return correct values
        if self._objtype == "reparametrize":
            return theta_init, var_init

        return coef_init, var_init


    def get_new_model(self, b, s2, prior, v2inv = None):
        model = LinearModel(self._X, self._y, b, s2, prior,
                    dj = self._dj,
                    objtype = self._objtype,
                    v2inv = v2inv,
                    debug = self._is_debug,
                    invert_method = self._invert_method,
                    invert_options = self._invert_options
                    )
        return model


    def get_res_normal_means(self):
        # y = theta = self._res.b_inv
        # prior 
        # sj2 = s2 / dj = self._res.residual_var / self._dj
        theta = self._res.b_inv
        prior = self._res.prior
        sj2   = self._res.residual_var / self._dj
        opts  = {'scale': self._res.residual_var, 'd': self._dj}
        model = NormalMeans(theta, prior, sj2, **opts)
        return model


    def get_elbo(self, b, s2, prior):
        model = self.get_new_model(b, s2, prior, v2inv = self._v2inv)
        return model.elbo()


    def get_model_func(self, params):
        """
        Calculate the objective function and its gradients with respect to the parameters
        """
        # get coefficients array b, prior parameters wk
        # and residual variance s2 from the solver
        #b, wk, logs2 = opt_utils.split_optparams(params, self._init_params, self._is_opt_list)
        #s2 = np.exp(logs2)
        b, wk, s2 = opt_utils.split_optparams(params, self._init_params, self._is_opt_list)
        self._prior.update_wmod(wk)

        # get a new model and calculate the objectives and gradients
        model = self.get_new_model(b, s2, self._prior)
        h     = model.objective
        dhdb  = model.bgrad
        dhdw  = model.wmod_grad
        dhds2 = model.s2grad # for s2
        #dhds2 = model.s2grad * s2 # for log(s2)
        grad  = opt_utils.combine_optparams([dhdb, dhdw, dhds2], self._is_opt_list)
        
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

        # Calculate ELBO if requested / only for ASH prior
        if self._is_elbo_calc:
            #b, wk, logs2 = opt_utils.split_optparams(params, self._init_params, self._is_opt_list)
            #s2 = np.exp(logs2)
            b, wk, s2 = opt_utils.split_optparams(params, self._init_params, self._is_opt_list)
            elbo = self.get_elbo(b, s2, self._prior)
            self._elbo_path.append(elbo)

        # Debug; Yes, I entered callback.
        self.logger.debug(f'Callback iteration {self._nclbk}')
        return


    # =====================================================
    # Attributes specific to linear regression
    # =====================================================
    @property
    def residual_var(self):
        return self._res.residual_var


    @property
    def intercept(self):
        return self._intercept


    @property
    def elbo_path(self):
        if self._is_elbo_calc:
            return self._elbo_path
        else:
            fdj = 0.5 * np.sum(np.log(self._dj))
            return self._h_path + fdj
