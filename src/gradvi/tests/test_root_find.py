
"""
Test all root finding algorithms
"""

import unittest
import numpy as np

from gradvi.optimize.root_find import vec_root
from gradvi.utils.logs import MyLogger
from gradvi.utils import unittest_tester as tester
from gradvi.tests import toy_priors
from gradvi.tests import toy_data

mlogger = MyLogger(__name__)

INVERT_METHODS = [
    'hybr',
    'newton',
    'trisection',
    'fssi-linear',
    'fssi-cubic',
    ]

def cubefun(x, a, b): 
    x2 = np.square(x)
    f = a * x2 * x - b
    return f

def cubefun_grad(x, a, b): 
    x2 = np.square(x)
    f  = a * x2 * x - b
    df = 3.0 * a * x2
    return f, df

def cubefun_inv(x, a): 
    x2 = np.square(x)
    f  = a * x2 * x
    df = 3.0 * a * x2
    return f, df

def cubefun_solve(a, b):
    xr = np.cbrt(b / a)
    return xr


def cubefun_bracket(a, b):
    xr = cubefun_solve(a, b)
    n  = xr.shape[0]
    xlo = xr - np.abs(np.random.normal(1, 0.2, size = n))
    xup = xr + np.abs(np.random.normal(1, 0.2, size = n))
    return [xlo, xup]


def sqfun(x, a, b):
    x2 = np.square(x)
    f = a * x2 - b
    return f


def sqfun_grad(x, a, b):
    x2 = np.square(x)
    f = a * x2 - b
    df = 2 * a * x
    return f, df


def sqfun_solve(a, b):
    xr = np.sqrt(b / a)
    return xr


class TestRootFind(unittest.TestCase):

    def test_root_result(self):
        np.random.seed(100)
        n = 100
        a = np.abs(np.random.normal(10, 1, size = n))
        b = np.abs(np.random.normal(2, 1, size = n))
        xr = dict()
        xt = dict()

        # Zero methods
        for meth in ['hybr', 'trisection', 'newton']:
            bracket  = cubefun_bracket(a, b) if meth == 'trisection' else None
            func     = cubefun_grad          if meth == 'newton' else cubefun
            jac      = True                  if meth == 'newton' else None
            xt[meth] = cubefun_solve(a, b)
            xr[meth] = vec_root(func, np.ones(n), args = (a, b), method = meth, bracket = bracket, jac = jac)

        # f(x) = y methods
        for meth in ['fssi-linear', 'fssi-cubic']:
            xt[meth] = cubefun_solve(2.0, b)
            xr[meth] = vec_root(cubefun_inv, np.zeros(n), 2.0, fx = b, method = meth)

        for meth in INVERT_METHODS:
            info_msg = f"Root using {meth} method should match analytical"
            err_msg  = f"Root does not match analytical for {meth} method"
            atol = 1e-8
            rtol = 1e-8
            if meth in ['fssi-linear', 'fssi-cubic']: 
                atol = 1e-4
            mlogger.info(info_msg)
            np.testing.assert_allclose(xr[meth], xt[meth], atol = atol, rtol = 1e-8, err_msg = err_msg)

        return


if __name__ == '__main__':
    tester.main()
