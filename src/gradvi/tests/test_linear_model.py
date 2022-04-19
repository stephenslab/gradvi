
import unittest
import numpy as np

from gradvi.models.linear_model import LinearModel
from gradvi.utils import unittest_tester as tester
from gradvi.utils.logs import MyLogger
from gradvi.tests import toy_priors

mlogger = MyLogger(__name__)

class TestLinearModel(unittest.TestCase):

    def get_lm_data(self, n = 100, p = 200, p_causal = 50, pve = 0.5, rho = 0.4):

        def sd2_from_pve (X, b, pve):
            return np.var(np.dot(X, b)) * (1 - pve) / pve

        np.random.seed(100)

        """
        Equicorr predictors
        X is sampled from a multivariate normal, with covariance matrix V.
        V has unit diagonal entries and constant off-diagonal entries rho.
        """
        iidX    = np.random.normal(size = n * p).reshape(n, p)
        comR    = np.random.normal(size = n).reshape(n, 1)
        X       = comR * np.sqrt(rho) + iidX * np.sqrt(1 - rho)
        bidx    = np.random.choice(p, p_causal, replace = False)
        b       = np.zeros(p)
        b[bidx] = np.random.normal(size = p_causal)
        s2      = sd2_from_pve(X, b, pve)
        y       = np.dot(X, b) + np.sqrt(s2) * np.random.normal(size = n)
        return X, y, b, s2


    def test_linear_model(self):
        self.X, self.y, self.b, self.s2 = self.get_lm_data()
        priors = toy_priors.get_all()
        for objtype in ["reparametrize"]:
            for prior in priors:
                lm = LinearModel(self.X, self.y, self.b, self.s2, prior, objtype = objtype)
                self._h_bderiv(lm, prior)
                self._h_aderiv(lm, prior)
                self._h_s2deriv(lm, prior)


    def _h_bderiv(self, lm, prior, eps = 1e-8):
        info_msg  = f"Linear Model. Checking dh/db for {lm._objtype} objective, {prior.prior_type} prior"
        error_msg = f"Derivative of linear model objective with respect to b does not match for {lm._objtype} objective, {prior.prior_type} prior"

        mlogger.info(info_msg)
        for i in range(self.b.shape[0]):
            b_eps = self.b.copy()
            b_eps[i] += eps
            lm_eps = LinearModel(self.X, self.y, b_eps, self.s2, prior, objtype = lm._objtype)
            d1 = lm.bgrad[i]
            d2 = (lm_eps.objective - lm.objective) / eps
            self.assertTrue(np.allclose(d1, d2, atol = 1e-5, rtol = 1e-8), msg = error_msg)
        return


    def _h_aderiv(self, lm, prior, eps = 1e-3):
        info_msg  = f"Linear Model. Checking dh/da for {lm._objtype} objective, {prior.prior_type} prior"
        error_msg = f"Derivative of linear model objective with respect to w does not match for {lm._objtype} objective, {prior.prior_type} prior"

        mlogger.info(info_msg)
        d1 = prior.wmod_grad(lm.wgrad)
        d2 = np.zeros(prior.k)
        for i in range(prior.k):
            amod_eps     = prior.wmod.copy()
            amod_eps[i] += eps
            prior_eps    = toy_priors.get_from_same_class(prior, prior.w)
            prior_eps.update_wmod(amod_eps)
            lm_eps       = LinearModel(self.X, self.y, self.b, self.s2, prior_eps, objtype = lm._objtype)
            d2[i]        = (lm_eps.objective - lm.objective) / eps
        self.assertTrue(np.allclose(d1, d2, atol = 0.1, rtol = 1e-8), msg = error_msg)
        return


    def _h_s2deriv(self, lm, prior, eps = 1e-8):
        info_msg  = f"Linear Model. Checking dh/ds2 for {lm._objtype} objective, {prior.prior_type} prior"
        error_msg = f"Derivative of linear model objective with respect to s2 does not match for {lm._objtype} objective, {prior.prior_type} prior"

        mlogger.info(info_msg)
        lm_eps = LinearModel(self.X, self.y, self.b, self.s2 + eps, prior, objtype = lm._objtype)
        d1 = lm.s2grad
        d2 = (lm_eps.objective - lm.objective) / eps
        self.assertTrue(np.allclose(d1, d2, atol = 1e-4, rtol = 1e-8), msg = error_msg)
        return


if __name__ == '__main__':
    teste.main()
