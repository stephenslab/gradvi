"""
Helper functions for optimization
"""

import numpy as np
import numbers

def split_optparams(x, plist, olist):
    """
    Split the array x of all optimization parameters 
    to context variables.

    Context variables are specific to the model being optimized.
    For example, in linear regression 
        y ~ Xb + e
        b ~ g(w)
        e ~ N(0, s2)
    the individual context variables are b (array of size p), 
    w (array of size k) and s2 (float).
    The array x is an array whose size depends on the parameters
    being optimized. If all the context variables are being
    optimized, then size(x) = p + k + 1. If only b and s2 
    are being optimized, then size(x) = p + 1

    Parameters
    ----------
    x : 1d array
        Array of parameters being optimized in a single vector. 
        In the above example, `x = np.concatenate((b, w, s2))`
        if all parameters are being optimized.
        If only b and s2 are being optimized, then
        `x = np.concatenate((b, s2))`.

    plist : list
        Initial values of context variables, used as reference
        for splitting x. Must contain a list of arrays such that
        sum(size(p_i)) = m, where p_i are the individual 
        elements of the list.
        `plist` must be of same length and same order as `olist`.

    olist : list of booleans
        Whether the context variable in plist is being optimized
        If yes, then it should be present in x. If no, then
        it is not present in x, and is obtained from plist.
        `olist` must be of same length and same order as `plist`.

    Returns
    -------
    xlist : list
        list of context variables

    """
    i     = 0
    idx   = 0
    pnew  = [None for x in plist]
    for i, (val, is_opt) in enumerate(zip(plist, olist)):
        if is_opt:
            if isinstance(val, np.ndarray):
                size    = val.shape[0]
                pnew[i] = x[idx: idx + size]
                idx    += size
            elif isinstance(val, numbers.Real):
                pnew[i] = x[idx]
                idx    += 1
        else:
            pnew[i] = val.copy()
    return tuple(pnew)


def combine_optparams(plist, olist):
    """
    Combine the context variables to a single 1d ndarray
    of optimization parameters.

    Context variables are specific to the model being optimized.
    For example, in linear regression
        y ~ Xb + e
        b ~ g(w)
        e ~ N(0, s2)
    the individual context variables are b (array of size p),
    w (array of size k) and s2 (float).
    The array x is an array whose size depends on the parameters
    being optimized. If all the context variables are being
    optimized, then size(x) = p + k + 1. If only b and s2
    are being optimized, then size(x) = p + 1

    Parameters
    ----------
    plist : list
        Values of context variables, which are concatenated to
        obtain the array of optimization parameters.
        `plist` must be of same length and same order as `olist`.

    olist : list of booleans
        Whether the context variable in plist is being optimized
        If yes, then it should be present in x. If no, then
        it is not included in x.
        `olist` must be of same length and same order as `plist`.

    Returns
    -------
    x : 1d array / list
        Array of optimization parameters in a single vector.
        In the above example, `x = np.concatenate((b, w, s2))`
        if all parameters are being optimized.
        If only b and s2 are being optimized, then
        `x = np.concatenate((b, s2))`.
        Note: If any element of plist is a <list> (instead of
        ndarray), then we return a <list> (instead of ndarray).
    """

    x = np.array([])
    if any([isinstance(p, list) for p in plist]):
        x = list()
    for val, is_opt in zip(plist, olist):
        if is_opt:
            if isinstance(val, np.ndarray):
                x = np.concatenate((x, val))
            elif isinstance(val, numbers.Real):
                x = np.concatenate((x, np.array([val])))
            elif isinstance (val, list):
                x += val
    return x
