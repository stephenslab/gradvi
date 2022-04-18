import unittest
import numpy as np

from gradvi.models.normal_means_ash_scaled import NormalMeansASHScaled
from gradvi.utils.logs import MyLogger
from gradvi.utils import unittest_tester as tester

mlogger = MyLogger(__name__)

class TestNMAshScaledPy(unittest.TestCase):

    def _NM_data(self):
        n = 100
        s = 1.2
        k = 6
        np.random.seed(100)
        y = np.random.normal(0, 1, size = n)
        X = np.random.normal(0, 1, size = (n, 2000))
        d = np.sum(np.square(X), axis = 1)
        wk = np.zeros(k)
        wk[1:(k-1)] = np.repeat(1/(k-1), (k - 2))
        wk[k-1] = 1 - np.sum(wk)
        sk = np.arange(k)
        return y, s, wk, sk, d

    def test_NM_logML_derivative(self, eps = 1e-8):
        mlogger.info("Check derivatives of Normal Means logML (scaled prior)")
        y, s, wk, sk, dj = self._NM_data()
        nmash     = NormalMeansASHScaled(y, s, wk, sk, d = dj)
        nmash_eps = NormalMeansASHScaled(y + eps, s, wk, sk, d = dj)
        deriv_analytic  = nmash.logML_deriv
        deriv_numeric   = (nmash_eps.logML - nmash.logML) / eps
        self.assertTrue(np.allclose(deriv_analytic, deriv_numeric, atol = 1e-8, rtol = 1e-4), 
                        msg = "Normal Means logML derivative does not match numeric results (scaled prior)")
        return


    def test_NM_logML_derivative2(self, eps = 1e-8):
        mlogger.info("Check second derivatives of Normal Means logML (scaled prior)")
        y, s, wk, sk, dj = self._NM_data()
        nmash     = NormalMeansASHScaled(y, s, wk, sk, d = dj)
        nmash_eps = NormalMeansASHScaled(y + eps, s, wk, sk, d = dj)
        deriv_analytic = nmash.logML_deriv2
        deriv_numeric  = (nmash_eps.logML_deriv - nmash.logML_deriv) / eps
        self.assertTrue(np.allclose(deriv_analytic, deriv_numeric, atol = 1e-8, rtol = 1e-4),
                        msg = "Normal Means logML second derivative does not match numeric results (scaled prior)")
        return


    def test_NM_logML_wderiv(self, eps = 1e-8):
        mlogger.info("Check derivatives of Normal Means logML with respect to w_k (scaled prior)")
        y, s, wk, sk, dj = self._NM_data()
        nmash = NormalMeansASHScaled(y, s, wk, sk, d = dj)
        for i in range(wk.shape[0]):
            wkeps = wk.copy()
            wkeps[i] += eps
            nmash_eps = NormalMeansASHScaled(y, s, wkeps, sk, d = dj)
            deriv_analytic = nmash.logML_wderiv[:, i]
            deriv_numeric  = (nmash_eps.logML - nmash.logML) / eps
            self.assertTrue(np.allclose(deriv_analytic, deriv_numeric, atol = 1e-6, rtol = 1e-5),
                            msg = f"Normal Means logML derivative with respect to w_k does not match numeric results (scaled prior) for k = {i}")
        return


    def test_NM_logML_deriv_wderiv(self, eps = 1e-8):
        mlogger.info("Check derivatives of Normal Means logML' with respect to w_k (scaled prior)")
        y, s, wk, sk, dj = self._NM_data()
        nmash = NormalMeansASHScaled(y, s, wk, sk, d = dj)
        for i in range(wk.shape[0]):
            wkeps = wk.copy()
            wkeps[i] += eps
            nmash_eps = NormalMeansASHScaled(y, s, wkeps, sk, d = dj)
            deriv_analytic = nmash.logML_deriv_wderiv[:, i]
            deriv_numeric  = (nmash_eps.logML_deriv - nmash.logML_deriv) / eps
            self.assertTrue(np.allclose(deriv_analytic, deriv_numeric, atol = 1e-6, rtol = 1e-5),
                            msg = f"Normal Means logML' derivative with respect to w_k does not match numeric results (scaled prior) for k = {i}")
        return


    def test_NM_logML_s2deriv(self, eps = 1e-8):
        mlogger.info("Check derivatives of Normal Means logML with respect to s^2 (scaled prior)")
        y, s, wk, sk, dj = self._NM_data()
        nmash = NormalMeansASHScaled(y, s, wk, sk, d = dj)
        nmash_eps = NormalMeansASHScaled(y, s, wk, sk, d = dj)
        nmash_eps.set_s2_eps(eps)
        deriv_analytic = nmash.logML_s2deriv
        deriv_numeric  = (nmash_eps.logML - nmash.logML) / eps
        self.assertTrue(np.allclose(deriv_analytic, deriv_numeric),
                        msg = "Normal Means logML derivative with respect to s^2 does not match numeric results (scaled prior)")
        return



    def test_NM_logML_deriv_s2deriv(self, eps = 1e-8):
        mlogger.info("Check derivatives of Normal Means logML' with respect to s^2 (scaled prior)")
        y, s, wk, sk, dj = self._NM_data()
        nmash = NormalMeansASHScaled(y, s, wk, sk, d = dj)
        nmash_eps = NormalMeansASHScaled(y, s, wk, sk, d = dj)
        nmash_eps.set_s2_eps(eps)
        deriv_analytic = nmash.logML_deriv_s2deriv
        deriv_numeric  = (nmash_eps.logML_deriv - nmash.logML_deriv) / eps
        self.assertTrue(np.allclose(deriv_analytic, deriv_numeric),
                        msg = "Normal Means logML' derivative with respect to s^2 does not match numeric results (scaled prior)")
        return


if __name__ == '__main__':
    tester.main()
