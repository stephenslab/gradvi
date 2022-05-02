"""
Class for normal means model with Point Normal Prior
    p(y | b, s^2) = N(y | b, s^2)
    p(b) = (1 - pi) * d0 + pi * N(0, s2)
"""

import numpy as np
import functools
import random
import logging

from ..utils.logs import MyLogger
from ..utils.decorators import run_once

from . import NormalMeans

class NMPointNormal(NormalMeans):

    def __init__(self, y, prior, sj2, **kwargs):
        """
        y, sj2 are vectors of length N
        """
        self._y = y
        self._pi1 = prior.w[0]
        self._n   = y.shape[0]
        self._sj2 = sj2
        if not isinstance(self._sj2, np.ndarray):
            self._sj2 = np.repeat(self._sj2, self._n)

        # set debug options
        debug = False
        if 'debug' in kwargs.keys(): debug = kwargs['debug']
        self._is_debug = debug
        logging_level  = logging.DEBUG if debug else logging.INFO
        self.logger    = MyLogger(__name__, level = logging_level)

        # Precalculate stuff
        self._sk2   = np.array([0, prior.w[1]]).reshape(1, 2)
        self._v2    = self._sj2.reshape(self._n, 1) + self._sk2
        self._logv2 = np.log(self._v2)

        self.randomseed = random.random()


    def __hash__(self):
        return hash(self.randomseed)


    @property
    def y(self):
        return self._y


    @property
    def yvar(self):
        return self._sj2


    def log_sum_exponent(self, z):
        """
        log(sum(exp(z))) = M + log(sum(exp(z - M)))
        """
        zmax = np.max(z, axis = 1)
        logsum = np.log(np.sum(np.exp(z-zmax.reshape(-1, 1)), axis = 1)) + zmax
        return logsum


    def log_sum_wkLjk(self, logLjk):
        """
        log(sum(wk * Ljk))
        takes care of zero values in wk
        requires log(Ljk) as input
        """
        if (self._pi1 > 0) and (self._pi1 < 1):
            z = logLjk + np.log(np.array([1 - self._pi1, self._pi1]))
            zsum = self.log_sum_exponent(z)
        elif self._pi1 == 0:
            zsum = self.log_sum_exponent(logLjk[:, 0].reshape(-1, 1))
        elif self._pi1 == 1:
            zsum = self.log_sum_exponent(logLjk[:, 1].reshape(-1, 1))
        else:
            raise ValueError("Normal Means Point Normal encountered out-of-bounds value of pi1")
        return self.log_sum_exponent(z)



    def logLjk(self, derive = 0):
        self.calculate_logLjk()
        return self._logLjk[derive]


    @run_once
    def calculate_logLjk(self, derive = 0):
        """
        this is one part of the posterior in normal means model. LogLjk is defined as:    
            p(y | f, s2)   =   (1 / sqrt(2 * pi)) * sum_k [w_k * exp(logLjk)]            # derive = 0
            p'(y | f, s2)  = - (y / sqrt(2 * pi)) * sum_k [w_k * exp(logLjk)]            # derive = 1 (first derivative)
            p''(y | f, s2) = (y^2 / sqrt(2 * pi)) * sum_k [w_k * exp(logLjk)] + p' / y   # derive = 2 
        returns N x 2 matrix
        """
        #self.logger.debug(f"Calculating logLjk for NM model hash {self.__hash__()}")
        y2  = np.square(self._y).reshape(self._n, 1)
        logv2 = self._logv2
        y2overv2 = y2 / self._v2
        # N x K length vector of posterior variances
        self._logLjk = {}
        self._logLjk[0] = -0.5 * (logv2 + y2overv2)     # N x 2 matrix
        self._logLjk[1] = -0.5 * (3 * logv2 + y2overv2) # N x 2 matrix
        self._logLjk[2] = -0.5 * (5 * logv2 + y2overv2) # N x 2 matrix
        return


    @property
    def logML(self):
        """
        Log marginal likelihood
        """
        self.calculate_logML()
        return self._logML


    @run_once
    def calculate_logML(self):
        #self._logML = - 0.5 * np.log(2 * np.pi) + self.log_sum_wkLjk(self.logLjk())
        self._logML = - 0.5 * np.log(2 * np.pi) \
                        + np.log((1 - self._pi1) * np.exp(self.logLjk()[:, 0]) \
                        + (self._pi1) * np.exp(self.logLjk()[:, 1]))
        return


    @property
    def ML_deriv_over_ML_y(self):
        """
        ML_deriv / (ML * y)
        """
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
        """
        lNM' = ML_deriv / ML
        """
        self.calculate_logML_deriv()
        return self._logML_deriv


    @run_once
    def calculate_logML_deriv(self):
        #self._logML_deriv = self.ML_deriv_over_ML_y * self._y
        ML = np.exp(self.logML)
        numer = (1 - self._pi1) * np.exp(self.logLjk(derive = 1)[:, 0]) \
                  + self._pi1 * np.exp(self.logLjk(derive = 1)[:, 1])
        self._logML_deriv = - self._y * numer / (ML * np.sqrt(2. * np.pi))
        return


    @property
    def ML_deriv2_over_ML(self):
        """
        ML_deriv2 / ML
        """
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
        """
        lNM'' = (ML_deriv2 / ML) - (lNM')^2
        """
        self.calculate_logML_deriv2()
        return self._logML_deriv2


    @run_once
    def calculate_logML_deriv2(self):
        self._logML_deriv2 = self.ML_deriv2_over_ML - np.square(self.logML_deriv)
        return


    @property
    def logML_wderiv(self):
        _pi1deriv = self.logML_pi1deriv
        _sk2deriv = self.logML_sk2deriv
        return np.concatenate((_pi1deriv, _sk2deriv), axis = 1)



    @property
    def logML_pi1deriv(self):
        """
        d lNM / dp1 = (Lj2 - Lj1) / exp( lNM + 0.5 log(2 pi) )
        """
        self.calculate_logML_pi1deriv()
        return self._logML_pi1deriv


    @run_once
    def calculate_logML_pi1deriv(self):
        """
        d lNM / dp1 = (Lj2 - Lj1) / exp( lNM + 0.5 log(2 pi) )
        Instead of taking exponents of logLjk, try subtracting some large number
        """
        logm = np.max(self.logLjk())
        t0 = np.diff(np.exp(self.logLjk() - logm))
        self._logML_pi1deriv = t0 * np.exp(logm - self.logML.reshape(self._n, 1) - 0.5 * np.log(2 * np.pi))
        return


    @property
    def logML_sk2deriv(self):
        self.calculate_logML_sk2deriv()
        return self._logML_sk2deriv


    @run_once
    def calculate_logML_sk2deriv(self):
        """
        d lNM / dsk2 = p1 * exp(logLjk(2) * (y2 / v2 - 1) - lNM + 0.5 log(2 pi))
        """
        y2overv2 = (np.square(self._y) / self._v2[:, 1]).reshape(self._n, 1)
        logLjk2  = self.logLjk(derive = 1)[:, 1].reshape(self._n, 1)
        logS0    = self.logML.reshape(self._n, 1) + 0.5 * np.log(2 * np.pi)
        self._logML_sk2deriv = 0.5 * self._pi1 * (y2overv2 - 1.) * np.exp(logLjk2 - logS0)
        return


    @property
    def logML_deriv_wderiv(self):
        _pi1deriv = self.logML_deriv_pi1deriv
        _sk2deriv = self.logML_deriv_sk2deriv
        return np.concatenate((_pi1deriv, _sk2deriv), axis = 1)
        


    @property
    def logML_deriv_pi1deriv(self):
        """
        d lNM' / dw (eq. 65)
        """
        self.calculate_logML_deriv_pi1deriv()
        return self._logML_deriv_pi1deriv


    @run_once
    def calculate_logML_deriv_pi1deriv(self):
        logm1 = np.max(self.logLjk(derive = 1))
        t0 = np.diff(np.exp(self.logLjk(derive = 1) - logm1))
        t1 = t0 * np.exp(logm1 - self.logML.reshape(self._n, 1) - 0.5 * np.log(2 * np.pi))
        self._logML_deriv_pi1deriv = - (self._y.reshape(-1, 1) * t1) \
                                     - (self.logML_pi1deriv * self.logML_deriv.reshape(self._n,1))
        return


    @property
    def logML_deriv_sk2deriv(self):
        self.calculate_logML_deriv_sk2deriv()
        return self._logML_deriv_sk2deriv


    @run_once
    def calculate_logML_deriv_sk2deriv(self):
        y2overv2 = (np.square(self._y) / self._v2[:, 1]).reshape(self._n, 1)
        logLjk2  = self.logLjk(derive = 2)[:, 1].reshape(self._n, 1)
        logS0    = self.logML.reshape(self._n, 1) + 0.5 * np.log(2 * np.pi)
        self._logML_deriv_sk2deriv = \
            - 0.5 * self._pi1 * self._y.reshape(self._n, 1) * (y2overv2 - 3.) * np.exp(logLjk2 - logS0) \
            - self.logML_sk2deriv * self.logML_deriv.reshape(self._n,1)
        return


    @property
    def logML_s2deriv(self):
        """
        d lNM / d(sj^2)
        """
        self.calculate_logML_sj2deriv()
        return self._logML_sj2deriv


    @run_once
    def calculate_logML_sj2deriv(self):
        y2 = np.square(self._y)
        log_numerator1  = self.log_sum_wkLjk(self.logLjk(derive = 1) - self._logv2)
        log_numerator2  = self.log_sum_wkLjk(self.logLjk(derive = 1))
        log_denominator = self.log_sum_wkLjk(self.logLjk())
        self._logML_sj2deriv = 0.5 * y2 * np.exp(log_numerator1 - log_denominator) \
                               - 0.5 * np.exp(log_numerator2 - log_denominator)
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
        log_numerator1  = self.log_sum_wkLjk(self.logLjk(derive = 2) - self._logv2)
        log_numerator2  = self.log_sum_wkLjk(self.logLjk(derive = 2))
        log_denominator = self.log_sum_wkLjk(self.logLjk())
        ML_deriv_sj2deriv_over_ML_y = - 0.5 * y2 * np.exp(log_numerator1 - log_denominator) \
                                           + 1.5 * np.exp(log_numerator2 - log_denominator)
        #ML_deriv_s2deriv_over_ML = self._y * np.exp(log_numerator - log_denominator)
        self._logML_deriv_sj2deriv = self._y * ML_deriv_sj2deriv_over_ML_y - self.logML_deriv * self.logML_s2deriv
        return


    def posterior(self):
        self.logger.debug(f"Calculating posterior for NM model.")
        logLjk = self.logLjk()
        logm   = np.max(logLjk, axis = 1)
        phijk  = np.zeros((self._n, 2))
        varjk  = np.zeros((self._n, 2))
        mujk   = np.zeros((self._n, 2))
        phijk[:, 0] = (1 - self._pi1) * np.exp(logLjk[:,  0] - logm)
        phijk[:, 1] = self._pi1 * np.exp(logLjk[:, 1] - logm)
        phijk /= np.sum(phijk, axis = 1).reshape(self._n, 1)
        varjk[:, 1] = 1 / ((1 / self._sj2) + (1 / self._sk2[0, 1]))
        mujk[:, 1] = self._y * varjk[:, 1] / self._sj2
        return phijk, mujk, varjk


    @property
    def analytical_posterior_mean(self):
        phijk, mujk, varjk = self.posterior()
        return np.sum(phijk * mujk, axis = 1)
