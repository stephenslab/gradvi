"""
Normal means model with adaptive shrinkage (ASH) prior.
    p(y_j | b_j, s_j^2) = N(y_j | m_j, s_j^2)
    p(m_j) = sum_k w_k N(m_j | 0, s_k^2)

For linear regression, s_j^2 = s^2 / d_j
where s^2 is the residual variance.

For the scaled prior, we have
    p(m_j) = sum_k w_k N(m_j | 0, s^2 * s_k^2)

"""

import numpy as np
import functools
import random
import logging
import numbers

from ..utils.logs import MyLogger
from ..utils.decorators import run_once
from ..utils.special import logsumexp
from ..utils.utils import get_optional_arg

from . import NormalMeans

class NMAsh(NormalMeans):
    """
    NMAsh provides the marginal log likelihood and its derivatives
    for the Normal Means model with adaptive shrinkage prior (Ash).
    We implement both the scaled and unscaled versions of the Ash
    prior.

    Parameters
    ----------
    y : ndarray of shape (n_features,)
        Responses of the normal means model

    prior : Ash(Prior) object

    sj2 : float / ndarray of shape (n_features,)
        The variance of the normal means model

    s2 : float, optional
        The scale factor of the ash prior, required for `ash_scaled` prior.
        Default is s2 = 1.0

    d : ndarray of shape (n_features,), optional
        d = s2 / sj2
        
    """

    def __init__(self, y, prior, sj2, **kwargs):
        # set debug options
        self._is_debug = kwargs['debug'] if 'debug' in kwargs.keys() else False
        logging_level  = logging.DEBUG if self._is_debug else logging.INFO
        self.logger    = MyLogger(__name__, level = logging_level)

        # Input
        self._y   = y
        self._wk  = prior.w
        self._sk  = prior.sk
        self._k   = prior.k
        self._sj2 = sj2
        self._n   = y.shape[0]
        self._is_scaled = prior.is_scaled

        self._scale = get_optional_arg('scale', 1.0, **kwargs)
        self._d     = get_optional_arg('d', None, **kwargs)
        if self._d is None: 
            self._d  = self._scale / self._sj2

        # Simplify computation for homogenous variance
        self._is_homogeneous = False
        if isinstance(self._sj2, numbers.Real):
            self._is_homogeneous = True

        if isinstance(self._sj2, np.ndarray) and np.all(self._sj2 == self._sj2[0]):
            self._is_homogeneous = True
            self._sj2 = self._sj2[0]
            self._d = self._scale / self._sj2

        self._nonzero_widx = np.where(self._wk != 0)[0]
        self.randomseed = random.random()

        # Precalculate stuff
        sk2    = np.square(self._sk).reshape(1, self._k)
        sj2_2d = self._sj2 if self._is_homogeneous else self._sj2.reshape(self._n, 1)
        if self._is_scaled:
            self._v2 = sk2 + (sj2_2d / self._scale)   # shape N x K for hetero, 1 x K for homo
        else:
            self._v2 = sk2 + sj2_2d                   # shape N x K for hetero, 1 x K for homo
        self._logv2 = np.log(self._v2)                # shape N x K for hetero, 1 x K for homo


    def __hash__(self):
        return hash(self.randomseed)


    @property
    def y(self):
        return self._y


    @property
    def yvar(self):
        return self._sj2


    def log_sum_wkLjk(self, logLjk):
        """
        log(sum(wk * Ljk))
        takes care of zero values in wk
        requires log(Ljk) as input
        """
        z = logLjk[:, self._nonzero_widx] + np.log(self._wk[self._nonzero_widx])
        return logsumexp(z, axis = 1)


    def logLjk(self, derive = 0):
        self.calculate_logLjk()
        return self._logLjk[derive]


    @run_once
    def calculate_logLjk(self, derive = 0):
        y2 = np.square(self._y).reshape(self._n, 1)
        tmp1 = self._logv2
        tmp2 = y2 / self._v2
        if self._is_scaled:
            tmp1 = tmp1 + np.log(self._scale)
            tmp2 = tmp2 / self._scale
        #
        self._logLjk = {}
        self._logLjk[0] = - 0.5 * (    tmp1 + tmp2) # N x K matrix
        self._logLjk[1] = - 0.5 * (3 * tmp1 + tmp2) # N x K matrix
        self._logLjk[2] = - 0.5 * (5 * tmp1 + tmp2) # N x K matrix
        return


    @property
    def logML(self):
        self.calculate_logML()
        return self._logML


    @run_once
    def calculate_logML(self):
        self._logML = - 0.5 * np.log(2 * np.pi) + self.log_sum_wkLjk(self.logLjk())
        return


    @property
    def ML_deriv_over_ML_y(self):
        self.calculate_ML_deriv_over_ML_y()
        return self._ML_deriv_over_ML_y


    @run_once
    def calculate_ML_deriv_over_ML_y(self):
        log_numerator   = self.log_sum_wkLjk(self.logLjk(derive = 1))
        log_denominator = self.log_sum_wkLjk(self.logLjk())
        self._ML_deriv_over_ML_y = - np.exp(log_numerator - log_denominator)
        return


    @property
    def logML_deriv(self):
        self.calculate_logML_deriv()
        return self._logML_deriv


    @run_once
    def calculate_logML_deriv(self):
        self._logML_deriv = self.ML_deriv_over_ML_y * self._y
        return


    @property
    def ML_deriv2_over_ML(self):
        self.calculate_ML_deriv2_over_ML()
        return self._ML_deriv2_over_ML


    @run_once
    def calculate_ML_deriv2_over_ML(self):
        log_numerator   = self.log_sum_wkLjk(self.logLjk(derive = 2))
        log_denominator = self.log_sum_wkLjk(self.logLjk())
        self._ML_deriv2_over_ML = self.ML_deriv_over_ML_y \
                                    + self._y * self._y * np.exp(log_numerator - log_denominator)
        return 


    @property
    def logML_deriv2(self):
        self.calculate_logML_deriv2()
        return self._logML_deriv2


    @run_once
    def calculate_logML_deriv2(self):
        self._logML_deriv2 = self.ML_deriv2_over_ML - np.square(self.logML_deriv)
        return


    @property
    def logML_wderiv(self):
        """
        dl/dw = Ljk(0) / (ML * sqrt(2 pi))
        log (dl/dw) = - 0.5 log(2 pi) + log(Ljk(0)) - logML
        """
        self.calculate_logML_wderiv()
        return self._logML_wderiv


    @run_once
    def calculate_logML_wderiv(self):
        self._logML_wderiv = np.exp(- 0.5 * np.log(2 * np.pi) + self.logLjk() - self.logML.reshape(self._n, 1))
        return


    @property
    def logML_deriv_wderiv(self):
        """
        dl'/dw, where l' = dl/db
        """
        self.calculate_logML_deriv_wderiv()
        return self._logML_deriv_wderiv
        

    @run_once
    def calculate_logML_deriv_wderiv(self):
        Ljk1_over_Ljk0 = np.exp(self.logLjk(derive = 1) - self.logLjk())
        self._logML_deriv_wderiv = - self.logML_wderiv \
            * (self._y.reshape(-1, 1) * Ljk1_over_Ljk0  + self.logML_deriv.reshape(self._n,1))
        return


    @property
    def logML_s2deriv(self):
        """
        dl/d(sj^2)
        """
        self.calculate_logML_sj2deriv()
        return self._logML_sj2deriv


    @run_once
    def calculate_logML_sj2deriv(self):
        y2  = np.square(self._y)
        log_denominator = self.log_sum_wkLjk(self.logLjk())
        log_numerator1  = self.log_sum_wkLjk(self.logLjk(derive = 1))
        if self._is_scaled:
            log_numerator2  = self.log_sum_wkLjk(self.logLjk(derive = 1) + self._logv2)
            self._logML_sj2deriv = \
                  0.5 * np.exp(log_numerator1 - log_denominator) * y2 / self._sj2 \
                - 0.5 * np.exp(log_numerator2 - log_denominator) * self._d
        else:
            log_numerator2  = self.log_sum_wkLjk(self.logLjk(derive = 1) - self._logv2)
            self._logML_sj2deriv = \
                  0.5 * np.exp(log_numerator2 - log_denominator) * y2 \
                - 0.5 * np.exp(log_numerator1 - log_denominator)
        return


    @property
    def logML_deriv_s2deriv(self):
        """
        d lNM' / d(sj^2)
        """
        self.calculate_logML_deriv_sj2deriv()
        return self._logML_deriv_sj2deriv


    @run_once
    def calculate_logML_deriv_sj2deriv(self):
        y2  = np.square(self._y)
        log_denominator = self.log_sum_wkLjk(self.logLjk())
        log_numerator1  = self.log_sum_wkLjk(self.logLjk(derive = 2))

        if self._is_scaled:
            log_numerator2  = self.log_sum_wkLjk(self.logLjk(derive = 2) + self._logv2)
            ML_deriv_sj2deriv_over_ML_y = \
                - 0.5 * np.exp(log_numerator1 - log_denominator) * y2 / self._sj2 \
                + 1.5 * np.exp(log_numerator2 - log_denominator) * self._d
        else:
            log_numerator2  = self.log_sum_wkLjk(self.logLjk(derive = 2) - self._logv2)
            ML_deriv_sj2deriv_over_ML_y = \
                - 0.5 * np.exp(log_numerator2 - log_denominator) * y2 \
                + 1.5 * np.exp(log_numerator1 - log_denominator)
        #
        self._logML_deriv_sj2deriv = \
              self._y * ML_deriv_sj2deriv_over_ML_y \
            - self.logML_deriv * self.logML_s2deriv
        return


    def posterior(self):
        self.logger.debug(f"Calculating posterior for NM model.")
        n = self._n
        k = self._k

        tmp = np.square(self._sk).reshape(1, k) / self._v2

        sj2 = self._sj2
        if isinstance(sj2, numbers.Real):
            sj2 = np.repeat(sj2, n)
        var = tmp * sj2.reshape(n, 1)
        mu  = tmp * self._y.reshape(n, 1)

        zjk  = np.zeros((n, k))
        phi  = np.zeros((n, k))
        inz  = self._nonzero_widx
        zjk[:, inz] = self.logLjk()[:, inz] + np.log(self._wk[inz])
        phi[:, inz] = np.exp(zjk - np.max(zjk, axis = 1, keepdims = True))
        phi        /= np.sum(phi, axis = 1, keepdims = True)
        return phi, mu, var


    @property
    def analytical_posterior_mean(self):
        phijk, mujk, varjk = self.posterior()
        return np.sum(phijk * mujk, axis = 1)
