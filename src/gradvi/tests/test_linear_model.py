
import unittest
import numpy as np

from gradvi.models import LinearModel
from gradvi.utils import unittest_tester as tester
from gradvi.utils.logs import MyLogger
from gradvi.tests import toy_priors
from gradvi.tests import toy_data

mlogger = MyLogger(__name__)

class TestLinearModel(unittest.TestCase):


    def test_linear_model(self):
        self.X, self.y, self.b, self.s2 = \
            toy_data.get_linear_model(
                n = 20, p = 50)
        priors = toy_priors.get_all()
        for objtype in ["reparametrize", "direct"]:
            for prior in priors:
                lm = LinearModel(self.X, self.y, self.b, self.s2, prior, objtype = objtype)
                self._h_bderiv(lm, prior)
                self._h_aderiv(lm, prior)
                self._h_s2deriv(lm, prior)


    def _h_bderiv(self, lm, prior, eps = 1e-8):
        info_msg = f"dh/db numerical differentiation for {lm._objtype} objective of linear model, {prior.prior_type} prior"
        err_msg  = f"dh/db is not equal to numerical differentiation for {lm._objtype} objective of linear model, {prior.prior_type} prior"

        mlogger.info(info_msg)
        for i in range(self.b.shape[0]):
            b_eps = self.b.copy()
            b_eps[i] += eps
            lm_eps = LinearModel(self.X, self.y, b_eps, self.s2, prior, objtype = lm._objtype)
            d1 = lm.bgrad[i]
            d2 = (lm_eps.objective - lm.objective) / eps
            np.testing.assert_allclose(d1, d2, atol = 1e-5, rtol = 1e-8, err_msg = err_msg)
        return


    def _h_aderiv(self, lm, prior, eps = 1e-3):
        info_msg = f"dh/da numerical differentiation for {lm._objtype} objective of linear model, {prior.prior_type} prior"
        err_msg  = f"dh/da is not equal to numerical differentiation for {lm._objtype} objective of linear model, {prior.prior_type} prior"

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
        np.testing.assert_allclose(d1, d2, atol = 0.1, rtol = 1e-8, err_msg = err_msg)
        return


    def _h_s2deriv(self, lm, prior, eps = 1e-8):
        info_msg = f"dh/ds2 numerical diffentiation for {lm._objtype} objective of linear model, {prior.prior_type} prior"
        err_msg  = f"dh/da is not equal to numerical differentiation for {lm._objtype} objective of linear model, {prior.prior_type} prior"

        mlogger.info(info_msg)
        lm_eps = LinearModel(self.X, self.y, self.b, self.s2 + eps, prior, objtype = lm._objtype)
        d1 = lm.s2grad
        d2 = (lm_eps.objective - lm.objective) / eps
        np.testing.assert_allclose(d1, d2, atol = 1e-4, rtol = 1e-8, err_msg = err_msg)
        return


if __name__ == '__main__':
    teste.main()
