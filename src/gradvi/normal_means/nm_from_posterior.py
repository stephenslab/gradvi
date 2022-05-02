
"""
Provides the Normal Means model given the posterior b,
some prior g(w) and the variance of the Normal Means model.

We invert the posterior to obtain the responses and 
yield the penalty operator and its derivatives.
"""
import numpy as np
import logging
import numbers

from . import nm_utils
from ..optimize.root_find import vec_root
from ..optimize.bracket import bracket_postmean
from ..utils.exceptions import NormalMeansInversionError
from ..utils.logs import MyLogger
from ..utils.decorators import run_once
from . import NormalMeans


class NormalMeansFromPosterior:

    def __init__(self, b, prior, sj2, **kwargs):
        self._b     = b # the posterior mean
        self._prior = prior # could be any prior class 
        self._sj2   = sj2 # variance of the normal means model (for linear model, sj2 = scale / d)

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

        # Homogeneous and heterogeneous variance of Normal Means model
        self._is_homogeneous = self._check_homogeneous()
        if self._is_homogeneous:
            self._sj2   = _get_number_or_first_array_element(self._sj2)
            self._scale = _get_number_or_first_array_element(self._scale)
            self._d     = _get_number_or_first_array_element(self._d)

        # Method cannot be None
        if self._method is None:
            if self._is_homogeneous:
                self._method = 'fssi-cubic'
            else:
                self._method = 'trisection'

        # Cannot use FSSI for non-homogeneous Normal Means
        if self._method in ['fssi-cubic', 'fssi-linear'] and not self._is_homogeneous:
            raise ValueError(f"Invertion method cannot handle heterogeneous variance of Normal Means model.")
            
        # Drop the optional arguments which have been processed
        self._kwargs = {
                k : v for (k, v) in kwargs.items() 
                    if k not in ['d', 'scale', 't0', 'method', 'bracket'] }
        # We always want full output for root finding
        # Force replace.
        self._kwargs['full_output'] = True

        # Internal variable to store the inversion object
        self._binvobj = None

        # Keep a count of number of Normal Means calls
        self._nmcalls = 0

        return


    def get_nm_model(self, z):
        nm = NormalMeans.create(
                z, self._prior, self._sj2,
                scale = self._scale, d = self._d)
        self._nmcalls += 1
        return nm


    def validate_brackets(self):
        is_correct = False
        if self._bracket is not None:
            xlo, xup = self._bracket
            fxlo = self._f_zero(xlo, self._b)
            fxup = self._f_zero(xup, self._b)
            is_correct = np.all(fxlo * fxup < 0)
        if not is_correct:
            xlo, xup, nfev = bracket_postmean(self._f_zero, self._b)
            self._bracket = [xlo, xup]
        return


    def invert_postmean(self):
        try:
            method = self._method
            kwargs = self._kwargs
            x0 = self._z0.copy()

            self.logger.debug(f"Invert using {method} method")

            if method == 'newton':
                self._binvobj = vec_root(
                        self._f_jac_zero, x0, args = (self._b,),
                        method = method, jac = True, options = self._kwargs)

            elif method == 'hybr':
                self._binvobj = vec_root(
                        self._f_zero, x0, args = (self._b,), 
                        method = method, options = self._kwargs)

            elif method == 'trisection':
                self.validate_brackets()
                self._binvobj = vec_root(
                        self._f_zero, x0, args = (self._b,),
                        method = method, bracket = self._bracket,
                        options = self._kwargs)

            elif method in ['fssi-linear', 'fssi-cubic']:
                options = self._kwargs
                options.setdefault('grid_scale', 'log')

                # We will only use the positive values of b,
                # as M(x) is symmetric around zero.
                bpos = np.abs(self._b)

                # Find the bounds
                ybounds        = np.array([np.min(bpos), np.max(bpos)])
                xlo, xup, nfev = bracket_postmean(self._f_zero, ybounds)
                xbounds        = [xlo[0], xup[1]]
                if xbounds[0] <= 0: xbounds[0] = 1e-8
                if xbounds[1] == 0: xbounds[1] = 10.0
                # 
                self._binvobj = vec_root(
                        self._f_jac_inv, x0, method = method, fx = bpos,
                        bounds = xbounds, options = options)

                # And set the signs correctly
                self._binvobj.x = self._binvobj.x * np.sign(self._b)

            if self._binvobj is not None:
                self.logger.debug(f"{self._binvobj.message}")

            else:
                raise NormalMeansInversionError(f"Returned NoneType object when inverting posterior mean using {method} method")

            if not self._binvobj.success:
                if not method == 'trisection':
                    self.logger.error(f"Inversion using {method} method did not converge. Trying trisection method.")
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


    # =====================================================
    # Functions for root finding
    # =====================================================

    def _f_zero(self, x, b):
        nm = self.get_nm_model(x)
        fx = nm.shrinkage_operator(jac = False)
        return fx - b


    def _f_jac_zero(self, x, b):
        nm = self.get_nm_model(x)
        fx, df, _, _ = nm.shrinkage_operator(jac = True)
        return fx - b, df


    def _f_jac_inv(self, x):
        nm = self.get_nm_model(x)
        fx, df, _, _ = nm.shrinkage_operator(jac = True)
        return fx, df


    # =====================================================
    # Functions for checking homogeneous varaince 
    # =====================================================

    def _check_homogeneous(self):
        """
        Check that the variance of Normal Means model is homogeneous
        """
        if all([isinstance(x, numbers.Number) for x in [self._sj2, self._scale, self._d]]):
            return True

        is_equal_sj2   = _check_array_equal(self._sj2)
        is_equal_scale = _check_array_equal(self._scale)
        is_equal_d     = _check_array_equal(self._d)

        if is_equal_d and is_equal_scale and is_equal_sj2:
            return True

        return False

                
    # =====================================================
    # Attributes
    # =====================================================

    @property
    def response(self):
        if self._binvobj is None:
            self.invert_postmean()
        return self._binvobj.x

    @property
    def nm_calls(self):
        return self._nmcalls

# =========================================================


# =====================================================
# Some utility functions
# =====================================================

def _check_array_equal(x):
    if isinstance(x, numbers.Number):
        return True
    is_equal = False
    if isinstance(x, np.ndarray):
        xr = np.repeat(x[0], x.shape[0])
        is_equal = np.allclose(x, xr)
    return is_equal


def _get_number_or_first_array_element(x):
    if isinstance(x, numbers.Number):
        return x
    if isinstance(x, np.ndarray):
        return x[0]
