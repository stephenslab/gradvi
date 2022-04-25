
"""
Calculate the inverse of the posterior mean operator
"""

import numpy as np
import collections
from scipy import optimize as sp_optimize
from scipy import interpolate as sp_interpolate

from . import NormalMeans
from . import nm_utils

from ..utils.exceptions import NMInversionError

MINV_FIELDS = ['x', 'xpath', 'objpath', 'success', 'message', 'niter', 'is_diverging']
class MinvInfo(collections.namedtuple('_MinvInfo', MINV_FIELDS)):
    __slots__ = ()


def _invert_hybr(b, prior, sj2, s2, dj, t0, **kwargs):

    def inv_func(z, b, prior, sj2, s2, dj):
        nm = NormalMeans.create(z, prior, sj2, scale = s2, d = dj)
        Mz = nm.shrinkage_operator(jac = False)
        return Mz - b

    tol = nm_utils.get_optional_arg('tol', 1.48e-8, **kwargs)

    opt = sp_optimize.root(
            inv_func,
            t0,
            args = (b, prior, sj2, s2, dj),
            method = 'hybr',
            jac = None,
            tol = tol)

    res = MinvInfo(x = opt.x, 
            xpath = None, 
            objpath = None, 
            niter = opt.nfev,
            success = opt.success, 
            message = opt.message,
            is_diverging = opt.success)

    return res


def _invert_newton(b, prior, sj2, s2, dj, t0, **kwargs):
    
    def inv_func(z, b, prior, sj2, s2, dj):
        nm = NormalMeans.create(z, prior, sj2, scale = s2, d = dj)
        Mz, zgrad, _, _ = nm.shrinkage_operator(jac = True)
        return Mz - b, zgrad

    tol     = nm_utils.get_optional_arg('tol', 1.48e-8, **kwargs)
    maxiter = nm_utils.get_optional_arg('maxiter', 50, **kwargs)

    x, xpath, objpath, resnorm = \
        rootfind_newton(
            inv_func, t0, True,
            args = (b, prior, sj2, s2, dj),
            tol = tol,
            maxiter = maxiter, 
            full_output = True)

    # Convergence information
    niter = len(xpath)
    success = False
    is_diverging = False
    if resnorm <= tol:
        success = True
        message = f"The solution converged after {niter} iterations."
    else:
        if niter < maxiter:
            message = f"Iteration stopped before reaching tolerance."
            raise NMInversionError('newton', message)
        else:
            if is_decreasing_monotonically(objpath):
                message = f"The solution is converging, but mean square difference did not reach tolerance.\n \
                    Try increasing the number of iterations."
            else:
                is_diverging = True
                message = f"The solution is diverging. Try different method for inverting."
                raise NMInversionError('newton', message, is_diverging = True)
    # Result
    res = MinvInfo(x = x,
            xpath = xpath,
            objpath = objpath,
            niter = niter,
            success = success,
            message = message,
            is_diverging = is_diverging)
    return res
    


def _invert_fssi(b, prior, sj2, s2, dj, interpolate = 'linear', **kwargs):


    def create_spline(x, y, dydx):
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

    
    ngrid = nm_utils.get_optional_arg('ngrid', 500, **kwargs)
    #
    # Get M^{-1}(b) for max(b)
    #
    bpos = np.abs(b)
    imax = np.argmax(bpos)
    ymax = np.array([bpos[imax]])
    xmax_opt = _invert_hybr(ymax, prior, sj2[imax], s2, dj[imax], np.zeros(1))
    if xmax_opt.success:
        xmax = 1.1 * xmax_opt.x[0]
        if xmax == 0:
            xmax = 10
    else:
        raise NMInversionError(f'fssi-{interpolate}', "Failed to get inverse for the chosen maximum value of posterior")
    #
    # Create a grid
    #
    xgrid = np.logspace(-4, np.log10(xmax), ngrid)
    # 
    nm = NormalMeans.create(xgrid, prior, sj2[0], scale = s2, d = dj[0])
    ygrid, xderiv, _, _ = nm.shrinkage_operator(jac = True)

    if np.any(xderiv == 0):
        raise NMInversionError(f'fssi-{interpolate}', "Derivative of shrinkage operator is zero for the chosen grid of responses.")
    dgrid = 1 / xderiv
    #xposgrid = np.logspace(-4, np.log10(xmax), ngrid)
    #print (f"Max values of b and M^{-1}(b) are {ymax}, {xmax}")
    #yposgrid = shrink_theta(xposgrid, std, wk, sk, np.ones(ngrid))
    #yposgrid, xderiv, _, _ = shrinkage_operator(xposgrid, std, wk, sk, np.ones(ngrid), jac = True)
    #dposgrid = 1 / xderiv
    #xgrid = np.concatenate((-xposgrid[::-1], xposgrid))
    #ygrid = np.concatenate((-yposgrid[::-1], yposgrid))
    #dgrid = np.concatenate((-dposgrid[::-1], dposgrid))
    if interpolate == 'linear':
        t_fssi = np.interp(bpos, ygrid, xgrid)
        t_fssi *= np.sign(b)
    elif interpolate == 'cubic':
        cs = create_spline(ygrid, xgrid, dgrid)
        t_fssi = cs(bpos)
        t_fssi *= np.sign(b)

    res = MinvInfo(x = t_fssi,
            xpath = None,
            objpath = None,
            niter = None,
            success = True,
            message = "Non-iterative method",
            is_diverging = None)

    return res


def rootfind_newton(func, x0, fprime,
        args=(), tol=1.48e-08,
        maxiter=50, full_output=None):
    """
    Newton Raphson iteration for vector input.
    """
    resnorm = np.inf  # norm of residuals
    objpath = list()
    xpath   = list()
    for itr in range(maxiter):
        fval = func(x0, *args)
        if fprime is True:
            fval, jval = fval
        else:
            jval = fprime(x0, *args)
        resnorm = np.sqrt(np.sum(fval**2))
        # keep full path if requested
        if full_output:
            xpath.append(x0.copy())
            objpath.append(resnorm)
        if resnorm < tol:
            break
        newton_step = fval / jval
        # do some clever damping here,
        # or use Numerical Recipes in C ch. 9.6-9.7 and do line search
        x0 -= newton_step
    if full_output:
         #return x0, fval, resnorm, newton_step, itr
        return x0, xpath, objpath, resnorm
    return x0


def is_decreasing_monotonically(x, nlast = 3):
    if len(x) > nlast:
        y = x[-nlast:]
        return all([y[i] >= y[i+1] for i in range(len(y) - 1)])
    return False