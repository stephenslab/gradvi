
"""
Provides the Normal Means model given the posterior b,
some prior g(w) and the variance of the Normal Means model.

We invert the posterior to obtain the responses and 
yield the penalty operator and its derivatives.
"""
import numpy as np
import logging

from . import nm_utils
from ..optimize.root_find import vec_root
from ..utils.exceptions import NormalMeansInversionError
from ..utils.logs import MyLogger
from ..utils.decorators import run_once

def NormalMeans(y, prior, sj2, **kwargs):
    return prior.normal_means(y, prior, sj2, **kwargs)


class NormalMeansFromPosterior:

    def __init__(self, b, prior, sj2, **kwargs):
        self._b     = b # the posterior mean
        self._prior = prior 
        self._sj2   = sj2

        # Get optional parameters from kwargs
        d                    = nm_utils.get_optional_arg('d', None, **kwargs)
        scale                = nm_utils.get_optional_arg('scale', None, **kwargs)
        self._z0             = nm_utils.get_optional_arg('t0', b.copy(), **kwargs)
        self._method         = nm_utils.get_optional_arg('method', None, **kwargs)
        self._scale, self._d = nm_utils.guess_nm_scale(self._sj2, scale, d)
        self._bracket        = nm_utils.get_optional_arg('bracket', None, **kwargs)

        # Logging
        self._is_debug       = nm_utils.get_optional_arg('debug', False, **kwargs)
        logging_level  = logging.DEBUG if self._is_debug else logging.INFO
        self.logger    = MyLogger(__name__, level = logging_level)

        # Method cannot be None
        if self._method is None:
            self._method = 'trisection'

        # Drop the optional arguments which have been processed
        self._kwargs = {
                k : v for (k, v) in kwargs.items() 
                    if k not in ['d', 'scale', 't0', 'method', 'bracket'] }
        # We always want full output here
        self._kwargs['full_output'] = True

        # Internal variable to store the inversion object
        self._binvobj = None

        return


    def get_nm_model(self, z):
        nm = NormalMeans(
                z, self._prior, self._sj2,
                scale = self._scale, d = self._d)
        return nm


    def _rootfind_zero_func(self, x):
        nm = self.get_nm_model(x)
        fx = nm.shrinkage_operator(jac = False)
        return fx - self._b


    def _rootfind_zero_func_jac(self, x):
        nm = self.get_nm_model(x)
        fx, df, _, _ = nm.shrinkage_operator(jac = True)
        return fx - self._b, df


    def _rootfind_invert_func_jac(self, x):

        sj2 = np.repeat(self._sj2[0], self._sj2.shape[0])
        if not np.allclose(sj2, self._sj2):
            raise ValueError(f"Invertion method cannot handle heterogeneous variance of Normal Means model.")
        dj  = np.repeat(self._d[0], self._d.shape[0])
        if not np.allclose(dj, self._d):
            raise ValueError(f"Invertion method cannot handle heterogenous norm of input matrix.")

        sj2 = self._sj2[0]
        dj  = self._d[0]
        nm = NormalMeans(x, self._prior, sj2, scale = self._scale, d = dj)
        fx, df, _, _ = nm.shrinkage_operator(jac = True)
        return fx, df


    def invert_postmean(self):
        try:
            method = self._method
            kwargs = self._kwargs
            x0 = self._z0.copy()

            self.logger.debug(f"Inverting using method {method}")

            if method == 'newton':
                self._binvobj = vec_root(
                        self._rootfind_zero_func_jac, x0,
                        method = method, jac = True,
                        options = self._kwargs)

            elif method == 'hybr':
                self._binvobj = vec_root(
                        self._rootfind_zero_func, x0,
                        method = method, 
                        options = self._kwargs)

            elif method == 'trisection':
                if self._bracket is None:
                    bup = self._b + 10.633 * self._b
                    blo = self._b - 10.633 * self._b
                    self._bracket = [blo, bup]
                self._binvobj = vec_root(
                        self._rootfind_zero_func, x0,
                        method = method,
                        bracket = self._bracket,
                        options = self._kwargs)

            elif method in ['fssi-linear', 'fssi-cubic']:
                options = self._kwargs
                options.setdefault('grid_scale', 'log')

                # Find the upper bound
                bmax = np.max(np.abs(self._b))
                func = lambda x: self._rootfind_invert_func_jac(x)[0] - bmax
                #bracket = [np.atleast_1d(bmax + 1.6 * bmax), np.atleast_1d(bmax - 1.6 * bmax)]
                #xup  = vec_root(func, np.atleast_1d(x0[0]), method = 'trisection', bracket = bracket, 
                #        options = {'full_output': False})
                xup = vec_root(func, np.atleast_1d(x0[0]), method = 'hybr')

                bounds = [1e-4, max(1, 1.633 * xup[0])]
                # 
                self._binvobj = vec_root(
                        self._rootfind_invert_func_jac, x0,
                        method = method, fx = np.abs(self._b),
                        bounds = bounds, 
                        options = options)
                self._binvobj.x = self._binvobj.x * np.sign(self._b)

            if self._binvobj is not None:
                self.logger.debug(f"{self._binvobj.message}")
            else:
                raise NormalMeansInversionError(f"Returned NoneType object when inverting posterior mean using {method} method")

            if not self._binvobj.success:
                if not method == 'trisection':
                    self.logger.error("Inversion using {method} method did not converge. Trying trisection method.")
                    self._method = 'trisection'
                    self.invert_postmean()
                #else:
                #    raise NormalMeansInversionError(f"Could not invert posterior mean using {method} method")

        except NormalMeansInversionError as err:
            self.logger.error(err.message)
            raise

        except BaseException as err:
            self.logger.error(f"Unexpected {err=}, {type(err)=}")
            raise

        return


    def penalty_operator(self, jac = True):
        z    = self.response
        nm   = self.get_nm_model(z)

        lambdaj = - nm.logML - 0.5 * np.square(z - self._b) / nm.yvar
        if jac:
            # Gradient with respect to b
            l_bgrad = (z - self._b) / nm.yvar

            # Gradient with respect to w
            l_wgrad = - nm.logML_wderiv

            # Gradient with respect to s2
            l_s2grad = -nm.logML_s2deriv + 0.5 * np.square(nm.logML_deriv)
            return lambdaj, l_bgrad, l_wgrad, l_s2grad
        return lambdaj


    @property
    def response(self):
        if self._binvobj is None:
            self.invert_postmean()
        return self._binvobj.x
