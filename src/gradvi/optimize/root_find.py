
"""
Root finding algorithms for N equations.

It is different from minimizing a vector-valued function.
Note the derivative of N functions is vector of size(N,)
whereas the derivative of a function returning
N-dimensional vector has a shape of (N, N).
"""

import numpy as np
import collections
from scipy import optimize as sp_optimize
from scipy import interpolate as sp_interpolate

from ..utils.logs import CustomLogger

# Default tolerance
_epsilon = np.sqrt(np.finfo(float).eps)
# The golden ratio
_gold    = 1.61803399

# Error messages
_ITR_ERR_MSG = {
    0   : "No iterations performed",
    1   : "The solution converged",
    2   : "Encountered zero derivative",
    3   : "The iterates are converging but the solution did not reach tolerance. Try incresing the number of iterations",
    4   : "The iterates are diverging",
    5   : "The iterates are fluctuating",
    10  : "Number of iterations must be greater than `minbleed` to check for convergence.",
    -1  : "Unknown error",
    -2  : "Error in checking convergence status",
}

mlogger = CustomLogger(__name__)

def vec_root(func, x0, args = (), method = None, jac = None,
        bracket = None, tol = None, maxiter = None, fx = None,
        bounds = None, options = None):
    """
    Unified interface to all root finding algorithms.

    func : callable
        A function to find a root of. It is called as ``func(x0, *args)``
        where x0 is an array, at every element of which the function
        is evaluated.

    x0 : ndarray
        Initial guess.

    args : tuple, optional
        Extra arguments passed to the function and its Jacobian,
        called as ``func(x0, *args)``

    method : str, optional
        Type of solver. Should be one of

            - 'hybr'
            - 'newton'
            - 'trisection'
            - 'fssi-linear'
            - 'fssi-cubic'

    jac : bool
        If `jac` is True, `func` is assumed to return the value of the Jacobian
        along with the objective function. If false, then only the objective
        function is assumed to be obtained from `func`.

    bracket : list of two ndarrays
        An interval bracketing a root. ``func(x, *args)`` must have different
        signs at the two endpoints.

    tol : float, optional
        Tolerance for termination. When `tol` is specified, the selected
        minimization algorithm sets some relevant solver-specific tolerance(s)
        equal to `tol`. For detailed control, use solver-specific
        options.

    maxiter : int
        Maximum number of iterations to perform. Depending on the
        method each iteration may use several function evaluations.

    fx : ndarray
        For methods which support finding f(x) = y, fx gives the value of y.
        It is used by FSSI methods to find x = finv(y).

    options : dict, optional
        A dictionary of solver options.


    https://github.com/scipy/scipy/blob/v1.8.0/scipy/optimize/_minimize.py
    """
    x0 = np.atleast_1d(np.asarray(x0))
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)

    if not isinstance(args, tuple):
        args = (args,)

    if method is None:
        method = 'trisection'

    meth = method.lower()

    if options is None:
        options = {}
    options = dict(options)

    if tol is not None:
        options.set_default('tol', tol)

    if maxiter is not None:
        options.set_default('maxiter', maxiter)

    # check if optional parameters are supported by the selected method
    # - jac
    if meth in ('trisection') and bool(jac):
        mlogger.warn(f"Method {method} does not use gradient information (jac).")

    if meth in ('newton') and (jac is None or jac == False):
        raise ValueError(f"Method {method} requires gradient information (jac).")

    # - bracket
    if meth in ('hybr', 'newton', 'fssi-linear', 'fssi-cubic') and bracket is not None:
        mlogger.warn(f"Method {method} does not use bracketing information (bracket).")

    if meth in ('trisection') and bracket is None:
        mlogger.error(f"Brackets not provided for root finding method {method}")
        raise ValueError(f"Method {method} requires bracketing information (bracket).")

    # - bounds
    if meth in ('hybr', 'newton', 'trisection') and bounds is not None:
        mlogger.warn(f"Method {method} does not use bound information (bounds).")

    # - full_output
    #if meth in ('fssi-linear', 'fssi-cubic') and \
    #        options.get('full_output', False):
    #    mlogger.warn(f"Method {method} does not support the full_output option.")

    # - fx
    if meth in ('hybr', 'newton', 'trisection') and fx is not None:
        mlogger.warn(f"Method {method} does not support finding non-zero value of f(x)")

    if meth in ('fssi-linear', 'fssi-cubic') and fx is None:
        raise ValueError(f"Method {method} requires a target value of fx; it solves f(x) = y")

    if meth == 'hybr':
        res = root_hybr(func, x0, jac, args, **options)

    elif meth == 'newton':
        res = root_newton(func, x0, jac, args, **options)

    elif meth == 'trisection':
        res = root_trisection(func, args, bracket = bracket, **options)

    elif meth == 'fssi-linear':
        res = root_fssi(func, fx, args, interpolate = 'linear', bounds = bounds, **options)

    elif meth == 'fssi-cubic':
        res = root_fssi(func, fx, args, interpolate = 'cubic', bounds = bounds, **options)

    else:
        raise ValueError(f"Unknown solver {method}")

    return res

    

class RootResult(dict):
    """
    Represents the root finding result.

    Attributes
    ----------
    x : ndarray
        Estimated root location.

    fun : ndarray
        Value of the function at the root location.

    jac : ndarray
        Derivatives of the function at the root location.

    nfev : int
        Number of times the function was called.

    norm : float
        Value of the L_2 norm of `fun`.

    success : bool
        True if the routine converged.

    status : int
        Status returned by the routine.

    message : str
        Readable form of the status. May contain
        description of success / the cause of termination.

    niter : int
        Number of iterations needed to find the root.

    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1 
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def root_newton(func, x0, fprime,
        args = (), tol = _epsilon,
        maxiter = 50, full_output = None, 
        **unknown_options):
    """
    Find the root using Newton Raphson iteration.
    """

    _check_unknown_options(unknown_options)

    resnorm = np.inf  # norm of residuals
    status  = 0
    respath = list()
    nfev = 0

    # the last `nc` steps will be checked for convergene / divergence / fluctuation
    nc = 5

    # minimum number of iterations, below which it is not worth checking
    # convergence / divergence / fluctuation
    minbleed = 10 + nc

    for itr in range(maxiter):
        fval = func(x0, *args)
        nfev += 1
        if fprime is True:
            fval, jval = fval
        else:
            jval = fprime(x0, *args)
        resnorm = np.sqrt(np.sum(fval**2))
        respath.append(resnorm)
        # Convergence
        if resnorm < tol:
            status = 1
            break
        # Zero derivative
        if np.any(jval == 0):
            status = 2
            break
        newton_step = fval / jval
        # do some clever damping here,
        # or use Numerical Recipes in C ch. 9.6-9.7 and do line search
        x0 -= newton_step

    niter = itr + 1
    
    # Check for errors
    if status not in [1, 2] and maxiter > 0:
        # Set to unknown error
        status = -1
        # Try to find more information if possible
        if niter == maxiter:
            if maxiter >= minbleed:
                status = _check_convergence_status(respath[-nc:])
            else:
                status = 10

    if full_output:
        res = RootResult(
                x = x0, fun = fval, jac = jval,
                nfev = nfev, norm = resnorm,
                success = status == 1,
                niter = niter,
                status = status,
                message = _ITR_ERR_MSG[status],
                )
        return res
    return x0


def root_hybr(func, x0, fprime,
        args = (), tol = _epsilon,
        maxiter = 0, full_output = None,
        **unknown_options):

    """
    Find the root using scipy.optimize.root(method = 'hybr')

    WARN: niter has been set to number of function evaluations,
    which is wrong.

    INFO: Setting fprime = True is not suggested because it will
    assume a vector-valued function (instead of N functions)
    and will create a N x N matrix (with zero off-diagonal elements)
    for fprime.
    """

    _check_unknown_options(unknown_options)

    if maxiter == 0: maxiter = 200 * (x0.shape[0] + 1)

    # fprime = True is not suggessted 
    # because it will use the jacobian form
    # assuming that we are minimizing a vector 
    # valued function. 
    if fprime:
        def jfunc(x0, *args):
            f, fprime = func(x0, *args)
            return f, np.diag(f)
    else:
        jfunc = func

    opt = sp_optimize.root(
        jfunc, x0, 
        args = args,
        method = 'hybr',
        jac = fprime, 
        tol = tol,
        options = {'maxfev' : maxiter})

    # ===========================
    # opt.status (from scipy)
    # see scipy/optimize/_minpack_py.py, line 248
    # errors = {
    #     0: "Improper input parameters were entered.",
    #     1: "The solution converged.",
    #     2: "The number of calls to function has "
    #         "reached maxfev = %d." % maxfev,
    #     3: "xtol=%f is too small, no further improvement "
    #         "in the approximate\n  solution "
    #         "is possible." % xtol,
    #     4: "The iteration is not making good progress, as measured "
    #         "by the \n  improvement from the last five "
    #         "Jacobian evaluations.",
    #     5: "The iteration is not making good progress, "
    #         "as measured by the \n  improvement from the last "
    #         "ten iterations.",
    #     'unknown': "An error occurred."}
    # ===========================

    if full_output:
        resnorm = np.sqrt(np.sum(np.square(opt.fun)))
        res = RootResult(
                x = opt.x, fun = opt.fun, jac = opt.fjac,
                nfev = opt.nfev, norm = resnorm,
                success = opt.success, status = opt.status, 
                message = opt.message,
                niter = opt.nfev,
                )
        return res
    return opt.x


def _fssi_cubic_spline(x, y, dydx):
    """
    Fast spline estimation using derivative.

    Returns a spline generator S(xi), which approximates
    f(xi) for any new value of xi, given a set of known
    points (xj, yj) and the derivative df/dx at the points f(xj).
    """
    n = x.shape[0]
    c = np.empty((4, n-1), dtype = y.dtype)
    xdiff = np.diff(x)
    xdiff[np.where(xdiff == 0)] = 1e-8
    slope = (y[1:] - y[:-1]) / xdiff
    t = (dydx[:-1] + dydx[1:] - 2 * slope) / xdiff
    c[0] = t / xdiff
    c[1] = (slope - dydx[:-1]) / xdiff - t
    c[2] = dydx[:-1]
    c[3] = y[:-1]
    return sp_interpolate.PPoly(c, x)

    
def root_fssi(func, y, 
        args = (), interpolate = 'cubic', ngrid = 500,
        bounds = None, full_output = None,
        grid_scale = 'linear',
        **unknown_options):
    """
    Solve x given y and a monotonic function f(x) = y.
    We use the "Fast Switch and Spline Interpolation (FSSI)" scheme
    described by Tommasini and Olivieri, 2018.

    We obtain the values of f(xj) on a given grid of points xj, 
    for j = 1, ..., n to create the set (xj, yj). 
    The matrix (yj, xj) obtained by switching the arrays gives the
    exact values of g(yj) = xj of the inverse function on the grid yj.
    We then use a spline interpolation to obtain g(yi) = xi for all
    other points yi.

    Parameters
    ----------
    func : callable ``func(x, *args)``
        A function which returns the values f(x) and df/dx.
        Here x must be a 1-D array and `args` are the other (fixed)
        parameters of `func`. The `args` must be independent of `x`.

    y : array of size n
        User supplied value of y at which the inverse function g(y)
        is estimated.

    xmax : float
        Maximum observed value of g(y). The grid of xj is
        created from (0, xmax).
        
    """

    _check_unknown_options(unknown_options)

    if interpolate not in ['linear', 'cubic']:
        raise ValueError("FSSI interpolation method must be linear or cubic.")

    nfev = 0

    is_bound = tuple([False, False])
    if bounds is not None:
        is_bound = tuple([x is not None for x in bounds])

    if not all(is_bound):
        mlogger.warn ("The bounds for the x-grid in FSSI is not provided")
        ybd = np.array([np.min(y), np.max(y)])
        xbd = ybd.copy()
        # do not override user provided bounds
        for i, m in enumerate(is_bound):
            if m: xbd[i] = bounds[i]
                
        while True:
            fxbd, _ = func(np.atleast_1d(xbd), *args)
            nfev   += 1
            _m      = [(fxbd[0] < ybd[0]) or is_bound[0],  # do not override user provided bounds
                       (fxbd[1] > ybd[1]) or is_bound[1]]
            if all(_m):
                break
            xbd = _increase_bounds(xbd, _gold, _m)
        bounds = xbd

    xmin = bounds[0]
    xmax = bounds[1]

    if grid_scale == 'log' and any([x <= 0 for x in bounds]):
        raise ValueError("FSSI cannot use grid in log space because at least one of the bounds is zero or negative.")

    if grid_scale == 'log':
        xgrid = np.logspace(np.log10(xmin), np.log10(xmax), ngrid)
    else:
        xgrid = np.linspace(xmin, xmax, ngrid)

    ygrid, dfdx = func(xgrid, *args)
    nfev += 1

    if np.any(dfdx == 0):
        # force linear interpolation
        interpolate = 'linear'
        #raise ArithmeticError("Derivative of f(x) returns zero values.")


    if interpolate == 'linear':
        xr  = np.interp(y, ygrid, xgrid)
    elif interpolate == 'cubic':
        dgdy = 1 / dfdx
        cs  = _fssi_cubic_spline(ygrid, xgrid, dgdy)
        xr  = cs(y)

    if full_output:
        fxr, jxr = func(xr, *args)
        nfev += 1
        resnorm = np.sqrt(np.sum(np.square(fxr - y)))
        res = RootResult(
                x = xr, fun = fxr, jac = jxr,
                nfev = nfev, norm = resnorm,
                bounds = bounds, 
                success = True,
                grid = (ygrid, xgrid),
                niter = nfev, message = "FSSI method does not use iteration"
                )
        return res
    return xr


def root_trisection(func,
        args=(), bracket = None,
        tol=_epsilon, maxiter=500,
        full_output = None,
        **unknown_options):

    """
    Numpy array implementation of the trisection method.
    Instead of solving each array element sequentially, we update
    the brackets of all points in the solution vector with a
    single function call.

    Parameters
    ----------
    func : callable ``func(x0, *args)
        A function which returns the values of f(x). 
        The algorithms estimates the values of x for which f(x) = 0.

    a : array of size n
        Upper bracket of x, such that f(a) * f(b) < 0

    b : array of size n
        Lower bracket of x, such that f(a) * f(b) < 0
    """

    _check_unknown_options(unknown_options)

    if bracket is None:
        raise ValueError("Trisection - user must provide brackets.")

    a, b = tuple(bracket)

    # The Trisection.
    #
    #   a ----- x1 ----- x2 ----- b
    #

    n = a.shape

    # Mask for checking which elements in the array
    # have still not converged
    # umask comes from "update mask"
    umask = np.full(n, True, dtype = bool)

    fa = func(a, *args)
    fb = func(b, *args)
    nfev = 2

    if np.any(np.sign(fb) * np.sign(fa) > 0):
        raise ValueError("Trisection - function must have opposite signs at the boundaries.")

    # Switch around values so that we have strictly fb > fa.
    if not np.all(fb > fa):
        ridx    =  np.where(fb < fa)
        _b      = b[ridx]
        b[ridx] = a[ridx]
        a[ridx] = _b

    fn = fb.copy()
    x0 = b.copy()

    # Success / status check
    status = 0
    respath = list()
    nc = 5

    for itr in range(maxiter):
        # ============
        # Select new points, evaluate fun
        # ============
        x1 = (2.0 * a + b) / 3.0
        x2 = (2.0 * b + a) / 3.0
        fx1 = func(x1, *args)
        fx2 = func(x2, *args)
        nfev += 2

        # ============
        # set  x = x1 if |f(x1)| < |f(x2)|
        # else x = x2
        # ============
        x0[umask] =  x2[umask]
        fn[umask] = fx2[umask]
        _mx = np.logical_and(np.abs(fx1) < np.abs(fx2), umask)
        if np.any(_mx):
            x0[_mx] =  x1[_mx]
            fn[_mx] = fx1[_mx]
        # we are decreasing the L1 norm here.
        respath.append(np.sum(np.abs(fn)))

        umask = np.abs(fn) >= tol
        if np.all(umask == False):
            status = 1
            break
        # ============
        # Select new values of a and b
        # if umask is False, no need to update
        #
        # After update, there are three possibilities:
        #
        #   a ----- x1 ----- x2 ----- b
        #   =                         +   Start
        #   -       +        +        +   Case 1
        #   -       -        +        +   Case 2
        #   -       -        -        +   Case 3
        #
        # ============

        # Case 1. If fa * fx1 < 0, new boundary is [a, x1]
        _m1 = np.logical_and(fa * fx1 < 0, umask)
        if np.any(_m1):
            b[_m1] = x1[_m1]

        # Case 2. if fx1 * fx2 < 0, new boundary is [x1, x2]
        _m2 = np.logical_and(fx1 * fx2 < 0, umask)
        if np.any(_m2):
            b[_m2]  = x2[_m2]
            a[_m2]  = x1[_m2]
            fa[_m2] = fx1[_m2]

        # Case 3. if fa * fx2 > 0, new boundary is [x2, b]
        # but we do not need to calculate fa * fx2,
        # because this will include all cases where
        # case1 and case2 are false.
        _m3 = np.logical_and(_m1 == False, _m2 == False)
        _m3 = np.logical_and(_m3, umask)
        if np.any(_m3):
            a[_m3]  = x2[_m3]
            fa[_m3] = fx2[_m3]

    niter = itr + 1

    # Check for errors
    if status not in [1] and maxiter > 0:
        # Set to unknown error
        status = -1
        # Try to find more information if possible
        if niter == maxiter and maxiter >= 2:
            _rpath = respath[-nc:] if niter > nc else respath.copy()
            status = _check_convergence_status(_rpath)

    if full_output:
        resnorm = np.sqrt(np.sum(np.square(fn)))
        res = RootResult(
                x = x0, fun = fn,
                nfev = nfev, norm = resnorm,
                success = status == 1 , 
                status = status,
                message = _ITR_ERR_MSG[status],
                niter = niter,
                )
        return res
    return x0
        

def is_converging(y):
    return all([y[i] >= y[i+1] for i in range(len(y) - 1)])


def is_diverging(y):
    return all([y[i] <= y[i+1] for i in range(len(y) - 1)])


def is_fluctuating(y):
    is_conv = [y[i] >= y[i+1] for i in range(len(y) - 1)]
    is_divg = [y[i] <= y[i+1] for i in range(len(y) - 1)]
    return (any(is_conv) and any(is_divg))


def _check_convergence_status(x):
    status = -2
    if is_converging(x):
        status = 3
    elif is_diverging(x):
        status = 4
    elif is_fluctuating(x):
        status = 5
    return status

def _increase_bounds(bounds, eps, is_bound, ztol = 1e-4):
    bmin = bounds[0]
    bmax = bounds[1]
    if not is_bound[0]:
        if np.abs(bmin) < ztol:
            bmin -= ztol * 2
        elif bmin > ztol:
            bmin /= eps
        elif bmin < -ztol:
            bmin *= eps
    if not is_bound[1]:
        if np.abs(bmax) < ztol:
            bmax += ztol * 2
        elif bmax > ztol:
            bmax *= eps
        elif bmax < -ztol:
            bmax /= eps
    return [bmin, bmax]


def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        mlogger.warn(f"Unknown solver options: {msg}")
