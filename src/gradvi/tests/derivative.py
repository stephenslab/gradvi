import numpy as np
import numbers
from gradvi.priors import Prior

def numerical_derivative(cls, func, x, cpre_args=(), cpost_args=(),
                         fpre_args = (), fpost_args = (),
                         is_class_input = False, is_func_property = False,
                         eps = 1e-8, ckwargs = {}, fkwargs = {},
                         is_scale = False, is_sum = False):
    callargs = {
        'cpre': cpre_args,
        'cpost': cpost_args,
        'fpre': fpre_args,
        'fpost': fpost_args,
        'is_class_input': is_class_input,
        'is_func_property': is_func_property,
        'ckwargs': ckwargs,
        'fkwargs': fkwargs,
        'return_sum': is_sum,
        }
    if is_scale:
        dfdx = symmetric_diff(cls, func, x + eps, x - eps,
                    call_func_nmscale, eps = 1e-8, callargs = callargs)
    elif isinstance(x, numbers.Number):
        dfdx = symmetric_diff(cls, func, x + eps, x - eps,
                    call_func, eps = eps, callargs = callargs)
    elif isinstance(x, np.ndarray) or isinstance(x, list):
        dfdx = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            x1      = x.copy()
            x2      = x.copy()
            x1[i]  += eps
            x2[i]  -= eps
            dfdx[i] = symmetric_diff(cls, func, x1, x2,
                    call_func, eps = eps, callargs = callargs)
    elif isinstance(x, Prior):
        dfdx = np.zeros(x.k)
        for i in range(x.k):
            pr1 = x.copy()
            pr2 = x.copy()
            w1  = pr1.w.copy()
            w2  = pr2.w.copy()
            w1[i] += eps
            w2[i] -= eps
            pr1.update_w(w1)
            pr2.update_w(w2)
            diff = symmetric_diff(cls, func, pr1, pr2, call_func, eps = eps, callargs = callargs)
            dfdx[i] = np.sum(diff)
    return dfdx

def symmetric_diff(cls, func, x1, x2, func_wrapper, eps = 1e-8, callargs = {}):
    f1 = func_wrapper(cls, func, x1, **callargs)
    f2 = func_wrapper(cls, func, x2, **callargs)
    dfdx   = (f1 - f2) / (2. * eps)
    return dfdx


def call_func(cls, func, x, cpre = (), cpost = (), fpre = (), fpost = (),
              is_class_input = False, is_func_property = False,
              ckwargs = {}, fkwargs = {}, return_sum = False):
    # Call a function
    if cls is None:
        f = func(*fpre, x, *fpost, **fkwargs)

    # Call a class method
    else:
        # create new class
        if is_class_input:
            cl = cls(*cpre, x, *cpost, **ckwargs)
        else:
            cl = cls(*cpre, **ckwargs)

        # call the method
        if is_func_property:
            f = getattr(cl, func)
        elif is_class_input:
            f  = getattr(cl, func)(*fpre, **fkwargs)
        else:
            f  = getattr(cl, func)(*fpre, x, *fpost, **fkwargs)

    # Do we have to sum the output?
    if return_sum:
        return np.sum(f)
    return f

def call_func_nmscale(cls, func, scale, **callargs):
    args = callargs.copy()
    args['ckwargs']['scale'] = scale
    sj2 = scale / callargs['ckwargs']['d']
    f   = call_func(cls, func, sj2, **args)
    return f
