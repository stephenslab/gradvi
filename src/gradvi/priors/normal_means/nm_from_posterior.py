
"""
Provides the Normal Means model given the posterior b,
some prior g(w) and the variance of the Normal Means model.

We invert the posterior to obtain the responses and 
yield the penalty operator and its derivatives.
"""
import numpy as np
import logging

from . import NormalMeans
from . import nm_utils
from ._invert import _invert_hybr, _invert_newton, _invert_fssi

from ...utils.exceptions import NMInversionError
from ...utils.logs import MyLogger
from ...utils.decorators import run_once

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

        # Logging
        self._is_debug       = nm_utils.get_optional_arg('debug', False, **kwargs)
        logging_level  = logging.DEBUG if self._is_debug else logging.INFO
        self.logger    = MyLogger(__name__, level = logging_level)

        # Method cannot be None
        if self._method is None:
            self._method = 'hybr'

        # Drop the optional arguments which have been processed
        self._kwargs = {
                k : v for (k, v) in kwargs.items() 
                    if k not in ['d', 'scale', 't0', 'method'] }

        # Internal variable to store the inversion object
        self._binvobj = None

        return


    def invert_postmean(self):
        try:
            method = self._method
            kwargs = self._kwargs

            self.logger.debug(f"Inverting using method {method}")

            if method == 'newton':
                self._binvobj = _invert_newton(
                        self._b, self._prior, self._sj2, 
                        self._scale, self._d, self._z0, 
                        **kwargs
                        )

            elif method == 'hybr':
                self._binvobj = _invert_hybr(
                        self._b, self._prior, self._sj2,
                        self._scale, self._d, self._z0,
                        **kwargs
                        )

            elif method == 'fssi-linear':
                self._binvobj = _invert_fssi(
                        self._b, self._prior, self._sj2,
                        self._scale, self._d,
                        interpolate = 'linear', 
                        **kwargs
                        )

            elif method == 'fssi-cubic':
                self._binvobj = _invert_fssi(
                        self._b, self._prior, self._sj2,
                        self._scale, self._d,
                        interpolate = 'cubic', 
                        **kwargs
                        )
            if self._binvobj is not None:
                self.logger.debug(f"{self._binvobj.message}")
        except NMInversionError:
            self.logger.error(f"Failed to invert the posterior mean for Normal Means model")
            raise
        except BaseException as err:
            self.logger.error(f"Unexpected {err=}, {type(err)=}")
            raise
        return


    def get_nm_model(self, z):
        nm = NormalMeans.create(
                z, self._prior, self._sj2,
                scale = self._s2, d = self._dj)
        return nm


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
            # Not implemented yet
            return lambdaj, l_bgrad, l_wgrad
        return lambdaj


    @property
    def response(self):
        if self._binvobj is None:
            self.invert_postmean()
        return self._binvobj.x
