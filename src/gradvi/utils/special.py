
import numpy as np

def logsumexp(a, axis = None, return_sign = False, keepdims = False):
    """
    log(sum(exp(A))) = M + log(sum(exp(A - M)))
    """

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if not keepdims:
        a_max = np.squeeze(a_max, axis = axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out
