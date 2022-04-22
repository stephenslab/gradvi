
import unittest
import numpy as np

from gradvi.normal_means import NormalMeans as NMeans
from gradvi.utils.logs import MyLogger
from gradvi.utils import unittest_tester as tester
from gradvi.tests import toy_priors

mlogger = MyLogger(__name__)

class TestNMOperator(unittest.TestCase):

    def get_nm_data(self):
        np.random.seed(100)
        n  = 100
        y  = np.random.normal(0, 1, size = n)
        scale = 1.2**2
        #dj = np.square(np.random.normal(0, 1, size = n) * n)
        #dj = np.ones(n) * n
        dj = np.square(np.random.normal(1, 0.5, size = n)) * n
        sj2 = scale / dj
        return n, y, sj2, scale, dj



    def operator_provider(self, nm, otype, jac = True):
        if otype == 'shrinkage':
            res = nm.shrinkage_operator(jac = jac)
        elif otype == 'penalty':
            res = nm.penalty_operator(jac = jac)
        return res


    def test_operators(self):
        self.n, self.y, self.sj2, self.scale, self.dj = self.get_nm_data()
        priors = toy_priors.get_all()
        for otype in ['shrinkage', 'penalty']:
            for prior in priors:
                nm = NMeans.create(self.y, prior, self.sj2, scale = self.scale, d = self.dj)
                x, x_bd, x_wd, x_s2d = self.operator_provider(nm, otype, jac = True)
                self._b_deriv(prior,  x, x_bd,  otype)
                self._w_deriv(prior,  x, x_wd,  otype)
                self._s2_deriv(prior, x, x_s2d, otype)


    def _b_deriv(self, prior, x, x_bd, otype, eps = 1e-8):
        info_msg  = f"Checking derivatives of {otype} operator for {prior.prior_type} prior"
        error_msg = f"{otype} operator derivative does not match numeric results for {prior.prior_type} prior"

        mlogger.info(info_msg)
        nm_eps = NMeans.create(self.y + eps, prior, self.sj2, scale = self.scale, d = self.dj)
        x_eps  = self.operator_provider(nm_eps, otype, jac = False)
        d2 = (x_eps - x) / eps
        self.assertTrue(np.allclose(x_bd, d2, atol = 1e-6, rtol = 1e-8), msg = error_msg)
        return


    def _w_deriv(self, prior, x, x_wd, otype, eps = 1e-8):
        info_msg  = f"Checking wk derivatives of {otype} operator for {prior.prior_type} prior"
        error_msg = f"{otype} operator wk derivative does not match numeric results for {prior.prior_type} prior"

        mlogger.info(info_msg)
        for i in range(prior.k):
            wkeps = prior.w.copy()
            wkeps[i] += eps
            prior_eps = toy_priors.get_from_same_class(prior, wkeps)
            nm_eps    = NMeans.create(self.y, prior_eps, self.sj2, scale = self.scale, d = self.dj)
            x_eps     = self.operator_provider(nm_eps, otype, jac = False)
            d1 = x_wd[:, i]
            d2 = (x_eps - x) / eps
            self.assertTrue(np.allclose(d1, d2, atol = 1e-6, rtol = 1e-8), msg = error_msg)
        return


    def _s2_deriv(self, prior, x, x_s2d, otype, eps = 1e-8):
        info_msg  = f"Checking s2 derivatives of {otype} operator for {prior.prior_type} prior"
        error_msg = f"{otype} operator s2 derivative does not match numeric results for {prior.prior_type} prior"

        mlogger.info(info_msg)
        sj2_eps = (self.scale + eps) / self.dj
        nm_eps = NMeans.create(self.y, prior, sj2_eps, scale = self.scale + eps, d = self.dj)
        x_eps  = self.operator_provider(nm_eps, otype, jac = False)
        d1 = x_s2d / self.dj
        d2 = (x_eps - x) / eps
        self.assertTrue(np.allclose(d1, d2, atol = 1e-6, rtol = 1e-8), msg = error_msg)
        return


if __name__ == '__main__':
    tester.main()
