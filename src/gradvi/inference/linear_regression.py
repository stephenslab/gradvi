"""
Unified interfaces to linear regression algorithms
"""
# Author: Saikat Banerjee

import numpy as np
from scipy import optimize as sp_optimize
from scipy.optimize import OptimizeResult as spOptimizeResult
import logging
import numbers

from ..models.linear_model import LinearModel
from ..utils.logs    import MyLogger
from . import coordinate_descent_step as cd_step
from . import elbo as elbo_py

from . import GradVI, GradVIResult

import shrinkage_operator_inverse

#from libgradvi_plr_mrash import plr_mrash as flib_penmrash
#from libgradvi_lbfgs_driver import lbfgsb_driver as flib_lbfgsb_driver


class LinearRegression(GradVI):
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
    prior : gradvi.priors object, default = gradvi.priors.ASH(k=10)
        Prior distribution g(w), along with initial values of w.

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

    display_progress: bool, default = True
        Whether to print summary of each iteration during the
        minimization.

    tol : float, default = 1e-9
        Tolerance for termination. When `tol` is specified, the selected
        minimization algorithm sets some relevant solver-specific tolerance(s)
        equal to `tol`. For detailed control, use solver-specific
        `options`.

    inversion_method : str, default = 'newton'
        Type of inversion method to use for inversion of the 
        Posterior Means Operator. Should be one of
            
            - 'newton'
            - 'fssi-linear'
            - 'fssi-cubic'
            - 'hybr'

    inversion_options : dict
        A dict of options for inversion of the Posterior Means Operator.
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
    coef_ : numpy.posterior mean of the coefficients

    prior_ :

    Examples
    --------
    """

    def __init__(
        self, prior = prior, method = 'L-BFGS-B', obj = 'parametrize',
        fit_intercept = True, options = None, 
        inversion_method = 'hybr', inversion_options = None,
        maxiter = 2000, display_progress = True, tol = 1e-9,
        get_elbo = False, function_call_py = True, lbfgsb_call_py = True,
        optimize_b = True, optimize_s = True, optimize_w = True,
        debug = True):

        self._prior        = prior
        self._is_intercept = use_intercept
        self._method       = method.lower()
        self._objtype      = obj.lower()
        self._is_elbo_calc = calculate_elbo

        # set default options for different minimization solvers
        if options is None:
            self._opts = dict()
        else:
            self._opts = options
        self._opts.setdefault('maxiter' : maxiter)
        self._opts.setdefault('disp' : display_progress)
        if self._method == 'l-bfgs-b':
            # Function tolerance. stop when ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
            self._opts.setdefault('ftol' : tol)
            # Gradient tolerance. stop when ``max{|proj g_i | i = 1, ..., n} <= gtol``
            self._opts.setdefault('gtol' : tol)
            self._opts.setdefault('maxfun' : maxiter * 10)
        elif self._method == 'cg':
            self._opts.setdefault('gtol' : tol)
        self._is_f_fortran      = not function_call_py
        self._is_lbfgsb_fortran = not lbfgsb_call_py

        # set which parameters to optimize in the minimization solver
        self._is_opt_b = optimize_b
        self._is_opt_s = optimize_s
        self._is_opt_w = optimize_w

        # set debug options
        self._is_debug = debug
        logging_level  = logging.DEBUG if debug else logging.INFO
        self.logger    = MyLogger(__name__, level = logging_level)

        # set options for inversion of the Posterior Means Operator
        self._inversion_method = inversion_method.lower()
        if inversion_options is None:
            self._inversion_options  = {'maxiter': maxiter,
                                        'tol': tol,
                                        'ngrid': None}
        return


    def fit(self, X, y, b_init = None, t_init = None, s2_init = None):
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

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
        k = self._prior.k

        # This values will not change during the optimization
        self._X  = X
        self._intercept = np.mean(y) if self._is_intercept else 0
        self._y = y - self._intercept
        self._dj = np.sum(np.square(self._X), axis = 0)

        # Initialization
        self._b_init, self._s2_init = self.initialize_params(b_init, t_init, s2_init)
        self._w_init = self._prior.w_init
        self._init_params = self.combine_optparams(self._b_init, self._w_init, self._s2_init)
        
        # Solver depending on the specified options:
        if self._method == 'L-BFGS-B' and self._is_lbfgsb_fortran:
            self._res = self.fit_fortran()
        else:
            self._res = self.fit_python()
        # end of solver
    
        return self


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

        var_init   = np.var(self._y) if s2_init is None else s2_init
        theta_init = np.zeros(p)
        coef_init  = np.zeros(p)
        if b_init is None:
            if t_init is not None:
                if s2_init is None: var_init   = np.var(self._y - np.dot(self._X, t_init))
                theta_init = t_init.copy()
                if self._objtype != "reparametrize":
                    coef_init  = shrinkage_operator(t_init)
        else:
            if s2_init is None: var_init   = np.var(self._y - np.dot(self._X, b_init))
            if self._objtype == "parametrize":
                # Get inverse if using parametrized objective
                theta_init = shrinkage_operator_inverse(b_init, t_init)
            coef_init  = b_init.copy()

        # Return correct values
        if self._objtype == "parametrize":
            return theta_init, var_init

        return coef_init, var_init


    def obj_fun_jac(self, params):
        """
        Calculate the objective function and its gradients with respect to the parameters
        """
        # get coefficients array b, prior parameters wk
        # and residual variance s2 from the solver
        b, wk, s2 = self.split_optparams(params)
        prior = self._prior.update_wmod(wk)

        model = LinearModel(self._X, self._y, b, s2, prior, 
                    dj = self._dj, objtype = self._objtype,
                    debug = self._debug)

        h = model.objective
        dhdb, dhdw, dhds2 = model.gradients
        grad = self.combine_optparams(dhdb, dhdw, dhds2)
        
        # Book-keeping
        self._nfev_count += 1
        self._njev_count += 1
        self._current_obj = h 
        self._current_s2  = s2
        self._current_wk  = wk
        self._current_b   = b 
        return h, grad


    def get_coef(self, b, s2, prior):
        model = LinearModel(self._X, self._y, b, s2, prior,
                    dj = self._dj, objtype = self._objtype,
                    debug = self._debug)
        return model.coef


    def get_coef_inv(self, b, s2, prior):
        model = LinearModel(self._X, self._y, b, s2, prior,
                    dj = self._dj, objtype = self._objtype,
                    debug = self._debug)
        return model.coef_inv


    def callback(self, params):
        self._clbk_count += 1
        # The objective function can be called multiple times in between callbacks
        # We append to paths only during callbacks
        self._hpath.append(self._current_obj)

        # Calculate ELBO if requested / only for ASH prior
        if self._is_elbo_calc:
            # get the ash prior wk and sk
            wk = self._prior.w
            sk = self._prior.sk
            # get the current value of the coefficients
            coef = self._current_b
            if self._objtype == "parametrize":
                coef = self.theta_to_coef(self._current_b, wk, self._current_s2)

            # calculate ELBO and append to path
            elbo = cd_step.elbo(self._X, self._y, sk, coef, wk, self._current_s2,
                                 dj = self._dj, s2inv = self._v2inv)
            #elbo = elbo_py.scalemix(self._X, self._y, sk, coef, wk, self._current_s2,
            #                        dj = self._dj, phijk = None, mujk = None, varjk = None, eps = 1e-8)
            self._elbo_path.append(elbo)
        self.logger.debug(f'Callback iteration {self._callback_count}')
        return


    def fit_fortran(self):
        """
        Run L-BFGS-B from FORTRAN
        """
        res = GradVIResult(b_post = f_bpost,
                           b_inv  = f_theta,
                           residual_var = f_s2,
                           prior = self._prior,
                           success = f_success,
                           status  = f_status,
                           message = f_message,
                           fun = f_obj,
                           grad = f_grad,
                           fitobj = None)
        return res


    def fit_python(self):
        """
        Fit a solver using scipy.optimize.minimize
        """
        # This values will not change during the optimization
        n, p = self._X.shape
        k = self._prior.k

        ## Precompute v2inv for ELBO calculation when using ash prior
        if self._is_elbo_calc:
            sk = self._prior.sk
            self._v2inv = np.zeros((p, k))
            self._v2inv[:, 1:] = 1 / (self._dj.reshape(p, 1) + 1 / np.square(sk[1:]).reshape(1, k - 1))

        # Bounds for optimization
        bbounds = [(None, None) for x in self._b_init]
        wbounds = self._prior.bounds
        s2bound = [(1e-8, None)]
        # bounds can be used only with L-BFGS-B.
        bounds = None
        if self._method == 'l-bfgs-b':
            bounds  = self.combine_optparams(bbounds, wbounds, s2bound)
        
        # keep track of the objective function, and other parameters 
        # during every iteration of the solver.
        self._h_path     = list()
        self._elbo_path  = list()
        self._clbk_count = 0
        self._nfev_count = 0
        self._njev_count = 0
        plr_min = sp_optimize.minimize(self.obj_fun_jac,
                                       self._init_params,
                                       method = self._method,
                                       jac = True,
                                       bounds = bounds,
                                       callback = self.callback,
                                       options = self._opts
                                       )
        # Return values
        b, wk, s2 = self.split_optparams(plr_min.x.copy())
        self._prior.update_wmod(wk)
        self._niter = plr_min.nit

        res = GradVIResult(b_post = self.get_coef(b, s2, self._prior),
                           b_inv  = self.get_coef_inv(b, s2, self._prior),
                           residual_var = s2,
                           prior = self._prior,
                           success = plr_min.success,
                           status  = plr_min.status,
                           message = plr_min.message,
                           fun     = plr_min.fun,
                           grad    = plr_min.grad,
                           fitobj = plr_min)

        # Debug logging
        self.logger.info (f"mr.ash.pen terminated at iteration {self._niter}.")
        self.logger.debug(f'Number of iterations: {self._niter}')
        self.logger.debug(f'Number of callbacks: {self._clbk_count}')
        self.logger.debug(f'Number of function calls: {self._nfev_count}')
        return res


    #def split_optparams(self, optparams):
    #    n, p = self._X.shape
    #    k    = self._prior.k
    #    idx  = 0
    #    if self._is_opt_b:
    #        bj = optparams[:p]. copy()
    #        idx += p
    #    else:
    #        bj = self._b_init
    #    if self._is_opt_w:
    #        wk = optparams[idx:idx+k].copy()
    #        idx += k
    #    else:
    #        ak = self._w_init
    #    if self._is_opt_s:
    #        s2 = optparams[idx].copy()
    #    else:
    #        s2 = self._s2_init
    #    return bj, ak, s2


    def split_optparams(self, optparams):
        i = 0
        idx = 0
        params   = [None for x in self._init_params]
        opt_list = [self._is_opt_b, self._is_opt_w, self._is_opt_s]
        for i, (val, is_opt) in enumerate(zip(self._init_params, opt_list)):
            if is_opt:
                if isinstance(val, np.ndarray):
                    size = val.shape[0]
                    params[i] = optparams[idx: idx + size]
                    idx += size
                elif isinstance(val, numbers.Real):
                    params[i] = optparams[idx]
                    idx += 1
            else:
                params[i] = val.copy()
        return tuple(params)
            

    def combine_optparams(self, bj, ak, s2):
        optparams = np.array([])
        if any([isinstance(x, list) for x in [bj, ak, s2]]):
            optparams = list()
        for val, is_included in zip([bj, ak, s2], [self._is_opt_b, self._is_opt_w, self._is_opt_s]): 
            if is_included:
                if isinstance(val, np.ndarray):
                    optparams = np.concatenate((optparams, val))
                elif isinstance(val, numbers.Real):
                    optparams = np.concatenate((optparams, np.array([val])))
                elif isinstance (val, list):
                    optparams += val
        return optparams

    
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
        return self._elbo_path
