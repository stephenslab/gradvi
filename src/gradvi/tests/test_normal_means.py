
import unittest
import numpy as np

from gradvi.normal_means import NormalMeans
from gradvi.utils.logs import MyLogger
from gradvi.utils import unittest_tester as tester
from gradvi.tests import toy_priors
from gradvi.tests import toy_data

mlogger = MyLogger(__name__)

class TestNormalMeansPy(unittest.TestCase):

    def test_NM(self):
        n = 100
        priors = toy_priors.get_all()
        for prior in priors:
            np.random.seed(100)
            self.y, self.sj2, self.scale, self.dj = \
                toy_data.get_normal_means(
                    prior, 
                    n = n,
                    dj = np.square(np.random.normal(1, 0.5, size = n)) * n
                    ) 
            nm = NormalMeans.create(self.y, prior, self.sj2, scale = self.scale, d = self.dj)
            self._logML_deriv(nm, prior)
            self._logML_deriv2(nm, prior)
            self._logML_wderiv(nm, prior)
            self._logML_deriv_wderiv(nm, prior)
            self._logML_s2deriv(nm, prior)
            self._logML_deriv_s2deriv(nm, prior)
        return


    def _logML_deriv(self, nm, prior, eps = 1e-8):
        info_msg  = f"Checking derivatives of Normal Means logML for {prior.prior_type} prior"
        error_msg = f"NM logML derivative does not match numeric results for {prior.prior_type} prior"

        mlogger.info(info_msg)
        nm_eps = NormalMeans.create(self.y + eps, prior, self.sj2, scale = self.scale, d = self.dj)
        d1 = nm.logML_deriv
        d2 = (nm_eps.logML - nm.logML) / eps
        self.assertTrue(np.allclose(d1, d2, atol = 1e-6, rtol = 1e-8), msg = error_msg)
        return


    def _logML_deriv2(self, nm, prior, eps = 1e-8):
        info_msg  = f"Checking second derivatives of Normal Means logML for {prior.prior_type} prior"
        error_msg = f"NM logML second derivative does not match numeric results for {prior.prior_type} prior"

        mlogger.info(info_msg)
        nm_eps = NormalMeans.create(self.y + eps, prior, self.sj2, scale = self.scale, d = self.dj)
        d1 = nm.logML_deriv2
        d2 = (nm_eps.logML_deriv - nm.logML_deriv) / eps
        self.assertTrue(np.allclose(d1, d2, atol = 1e-4, rtol = 1e-8), msg = error_msg)
        return


    def _logML_wderiv(self, nm, prior, eps = 1e-8):
        info_msg  = f"Checking wk derivatives of Normal Means logML for {prior.prior_type} prior"
        error_msg = f"NM logML wk derivative does not match numeric results for {prior.prior_type} prior"

        mlogger.info(info_msg)
        for i in range(prior.k):
            wkeps = prior.w.copy()
            wkeps[i] += eps
            prior_eps = toy_priors.get_from_same_class(prior, wkeps)
            nm_eps    = NormalMeans.create(self.y, prior_eps, self.sj2, scale = self.scale, d = self.dj)
            d1 = nm.logML_wderiv[:, i]
            d2 = (nm_eps.logML - nm.logML) / eps
            self.assertTrue(np.allclose(d1, d2, atol = 1e-6, rtol = 1e-8), msg = error_msg)
        return


    def _logML_deriv_wderiv(self, nm, prior, eps = 1e-8):
        info_msg  = f"Checking wk derivatives of Normal Means logML' for {prior.prior_type} prior"
        error_msg = f"NM logML' wk derivative does not match numeric results for {prior.prior_type} prior"

        mlogger.info(info_msg)
        for i in range(prior.k):
            wkeps = prior.w.copy()
            wkeps[i] += eps
            prior_eps = toy_priors.get_from_same_class(prior, wkeps)
            nm_eps    = NormalMeans.create(self.y, prior_eps, self.sj2, scale = self.scale, d = self.dj)
            d1 = nm.logML_deriv_wderiv[:, i]
            d2 = (nm_eps.logML_deriv - nm.logML_deriv) / eps
            self.assertTrue(np.allclose(d1, d2, atol = 1e-4, rtol = 1e-8), msg = error_msg)
        return


    def _logML_s2deriv(self, nm, prior, eps = 1e-8):
        info_msg  = f"Checking sj2 derivatives of Normal Means logML for {prior.prior_type} prior"
        error_msg = f"NM logML sj2 derivative does not match numeric results for {prior.prior_type} prior"

        mlogger.info(info_msg)
        sj2_eps = (self.scale + eps) / self.dj
        nm_eps = NormalMeans.create(self.y, prior, sj2_eps, scale = self.scale + eps, d = self.dj)
        d1 = nm.logML_s2deriv / self.dj
        d2 = (nm_eps.logML - nm.logML) / eps
        self.assertTrue(np.allclose(d1, d2, atol = 1e-6, rtol = 1e-8), msg = error_msg)
        return


    def _logML_deriv_s2deriv(self, nm, prior, eps = 1e-8):
        info_msg  = f"Checking sj2 derivatives of Normal Means logML' for {prior.prior_type} prior"
        error_msg = f"NM logML' sj2 derivative does not match numeric results for {prior.prior_type} prior"

        mlogger.info(info_msg)
        sj2_eps = (self.scale + eps) / self.dj
        nm_eps = NormalMeans.create(self.y, prior, sj2_eps, scale = self.scale + eps, d = self.dj)
        d1 = nm.logML_deriv_s2deriv / self.dj
        d2 = (nm_eps.logML_deriv - nm.logML_deriv) / eps
        self.assertTrue(np.allclose(d1, d2, atol = 1e-4, rtol = 1e-8), msg = error_msg)
        return


if __name__ == '__main__':
    tester.main()
