import unittest
import numpy as np

from gradvi.models.normal_means_ash import NormalMeansASH
from gradvi.utils.logs import MyLogger
from gradvi.utils import unittest_tester as tester

mlogger = MyLogger(__name__)

class TestNMAshPy(unittest.TestCase):

    def _NM_data(self):
        n = 100
        s = 1.2
        k = 6
        np.random.seed(100)
        y = np.random.normal(0, 1, size = n)
        wk = np.zeros(k)
        wk[1:(k-1)] = np.repeat(1/(k-1), (k - 2))
        wk[k-1] = 1 - np.sum(wk)
        sk = np.arange(k)
        return y, s, wk, sk

    def test_NM_logML_derivative(self, eps = 1e-8):
        mlogger.info("Check derivatives of Normal Means logML")
        y, s, wk, sk = self._NM_data()
        nmash     = NormalMeansASH(y, s, wk, sk)
        nmash_eps = NormalMeansASH(y + eps, s, wk, sk)
        deriv_analytic  = nmash.logML_deriv
        deriv_numeric   = (nmash_eps.logML - nmash.logML) / eps
        self.assertTrue(np.allclose(deriv_analytic, deriv_numeric, atol = 1e-6, rtol = 1e-8), 
                        msg = "Normal Means logML derivative does not match numeric results")
        return


    def test_NM_logML_derivative2(self, eps = 1e-8):
        mlogger.info("Check second derivatives of Normal Means logML")
        y, s, wk, sk = self._NM_data()
        nmash     = NormalMeansASH(y, s, wk, sk)
        nmash_eps = NormalMeansASH(y + eps, s, wk, sk)
        deriv2_analytic = nmash.logML_deriv2
        deriv2_numeric  = (nmash_eps.logML_deriv - nmash.logML_deriv) / eps
        self.assertTrue(np.allclose(deriv2_analytic, deriv2_numeric),
                        msg = "Normal Means logML second derivative does not match numeric results")
        return


    def test_NM_logML_wderiv(self, eps = 1e-8):
        mlogger.info("Check derivatives of Normal Means logML with respect to w_k")
        y, s, wk, sk = self._NM_data()
        nmash = NormalMeansASH(y, s, wk, sk)
        for i in range(wk.shape[0]):
            wkeps = wk.copy()
            wkeps[i] += eps
            nmash_eps = NormalMeansASH(y, s, wkeps, sk)
            deriv_analytic = nmash.logML_wderiv[:, i]
            deriv_numeric  = (nmash_eps.logML - nmash.logML) / eps
            self.assertTrue(np.allclose(deriv_analytic, deriv_numeric),
                            msg = f"Normal Means logML derivative with respect to w_k does not match numeric results for component k = {i}")
        return


    def test_NM_logML_deriv_wderiv(self, eps = 1e-8):
        mlogger.info("Check derivatives of Normal Means logML' with respect to w_k")
        y, s, wk, sk = self._NM_data()
        nmash = NormalMeansASH(y, s, wk, sk)
        for i in range(wk.shape[0]):
            wkeps = wk.copy()
            wkeps[i] += eps 
            nmash_eps = NormalMeansASH(y, s, wkeps, sk)
            deriv_analytic = nmash.logML_deriv_wderiv[:, i]
            deriv_numeric  = (nmash_eps.logML_deriv - nmash.logML_deriv) / eps
            self.assertTrue(np.allclose(deriv_analytic, deriv_numeric),
                            msg = f"Normal Means logML' derivative with respect to w_k does not match numeric results for component k = {i}")
        return


    def test_NM_logML_deriv_s2deriv(self, eps = 1e-8):
        mlogger.info("Check derivatives of Normal Means logML' with respect to s^2")
        y, s, wk, sk = self._NM_data()
        nmash = NormalMeansASH(y, s, wk, sk)
        nmash_eps = NormalMeansASH(y, s, wk, sk)
        nmash_eps.set_s2_eps(eps)
        deriv_analytic = nmash.logML_deriv_s2deriv
        deriv_numeric  = (nmash_eps.logML_deriv - nmash.logML_deriv) / eps
        self.assertTrue(np.allclose(deriv_analytic, deriv_numeric),
                        msg = "Normal Means logML' derivative with respect to s^2 does not match numeric results")
        return


if __name__ == '__main__':
    tester.main()
