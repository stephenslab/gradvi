

import numpy as np

_gold = 1.61803399

def bracket_postmean(func, b, args = (), tol = 1e-4, gt_one = _gold):

    """
    Find the brackets of a zero function from posterior mean operator, that is
        f(x) = M(x) - b
    Assumes shrinkage properties of M(), in particular M(b) < b for all b.
    Hence, it cannot be used for other general functions.

    Parameters
    ----------
    func : callable
        The shrinkage function f, which is called as ``func(x, b, *args)``

    b : ndarray
        The vector b for calling the function ``func``

    args : tuple, optional
        Other arguments for calling ``func``

    tol : float, optional
        Converges when f(xup) > tol and f(xlo) < -tol

    gt_one : float, optional
        A factor which will be multiplied with the estimate of xup
        and xlo while searching the bracket.

    Returns
    -------
    xlo : ndarray
        Values of lower bracket such that f(xlo) < 0

    xup : ndarray
        Values of upper bracket such that f(xup) > 0
    """
    eps = min(1.0, np.std(b))

    xup = b.copy()
    xlo = b.copy()
    ipos = np.where(b > 0)[0] # b is always 1d
    ineg = np.where(b < 0)[0]

    # If b is positive, f(b - eps) < 0 always
    xlo[ipos] -= eps

    # If b is negative, f(b + eps) > 0 always
    xup[ineg] += eps

    nfev = 0
    xup[ipos] *= gt_one
    xlo[ineg] *= gt_one

    xev = np.zeros_like(b)
    xev[ipos] = xup[ipos] * gt_one
    xev[ineg] = xlo[ineg] * gt_one

    while True:

        fx = func(xev, b, *args)
        nfev += 1
        fneg = np.where(fx < tol)[0] # fx is always 1d
        fpos = np.where(fx > -tol)[0]
        # Re-evaluate xup[ipos]
        _fneg = np.intersect1d(fneg, ipos)
        # Re-evaluate xlo[ineg]
        _fpos = np.intersect1d(fpos, ineg)
        uidx = np.concatenate((_fneg, _fpos))
        if uidx.shape[0] == 0:
            break
        xev[uidx] *= gt_one

    xup[ipos] = xev[ipos]
    xlo[ineg] = xev[ineg]

#     while True:
#         fxup_est = func(xup, b, *args)
#         nfev += 1
#         fneg = np.where(fxup_est < tol)[0] # fxup_est is alway 1d

#         if fneg.shape[0] == 0:
#             break
#         xup[fneg] *= gt_one

#     while True:
#         fxlo_est = func(xlo, b, *args)
#         nfev += 1
#         fpos = np.where(fxlo_est > -tol)[0]

#         if fpos.shape[0] == 0:
#             break
#         xlo[fpos] *= gt_one

    return xlo, xup, nfev
