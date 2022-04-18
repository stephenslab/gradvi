import unittest
import numpy as np
np.random.seed(200)

from gradvi.models.plr_ash import PenalizedMrASH as PenMrASH
from gradvi.models.normal_means_ash import NormalMeansASH
from gradvi.utils import unittest_tester as tester
from gradvi.utils.logs import MyLogger

mlogger = MyLogger(__name__)

class TestPLRObjective(unittest.TestCase):

    def _ash_data(self, n = 200, p = 2000, p_causal = 5, pve = 0.5, rho = 0.0, k = 6, seed = None):

        def sd_from_pve (X, b, pve):
            return np.sqrt(np.var(np.dot(X, b)) * (1 - pve) / pve)

        if seed is not None: np.random.seed(seed)

        '''
        ASH prior
        '''
        wk = np.zeros(k)
        wk[1:(k-1)] = np.repeat(1/(k-1), (k - 2))
        wk[k-1] = 1 - np.sum(wk)
        sk = np.arange(k)
        '''
        Equicorr predictors
        X is sampled from a multivariate normal, with covariance matrix V.
        V has unit diagonal entries and constant off-diagonal entries rho.
        '''
        iidX    = np.random.normal(size = n * p).reshape(n, p)
        comR    = np.random.normal(size = n).reshape(n, 1)
        X       = comR * np.sqrt(rho) + iidX * np.sqrt(1 - rho)
        bidx    = np.random.choice(p, p_causal, replace = False)
        b       = np.zeros(p)
        b[bidx] = np.random.normal(size = p_causal)
        sigma   = sd_from_pve(X, b, pve)
        y       = np.dot(X, b) + sigma * np.random.normal(size = n)
        return X, y, b, sigma, wk, sk


    def test_deriv_basic(self, eps = 1e-8):
        self._test_objective_function_deriv(eps, is_prior_scaled = False)
        return


    def test_deriv_scaled(self, eps = 1e-8):
        self._test_objective_function_deriv(eps, is_prior_scaled = True)
        return


    def _test_objective_function_deriv(self, eps, is_prior_scaled = False):
        X, y, b, s, wk, sk = self._ash_data(seed = 100)
        pmash = PenMrASH(X, y, b, s, wk, sk, is_prior_scaled = is_prior_scaled)
        obj   = pmash.objective
        bgrad, wgrad, s2grad = pmash.gradients
        bgrad_numeric = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            bnew = b.copy()
            bnew[i] += eps
            pmash_beps = PenMrASH(X, y, bnew, s, wk, sk, debug = False, is_prior_scaled = is_prior_scaled)
            bgrad_numeric[i] = (pmash_beps.objective - obj) / eps
        wgrad_numeric = np.zeros(wk.shape[0])
        for i in range(wk.shape[0]):
            wknew = wk.copy()
            wknew[i] += eps
            pmash_weps = PenMrASH(X, y, b, s, wknew, sk, debug = False, is_prior_scaled = is_prior_scaled)
            wgrad_numeric[i] = (pmash_weps.objective - obj) / eps
        pmash_s2eps = PenMrASH(X, y, b, s, wk, sk, debug = False, is_prior_scaled = is_prior_scaled)
        pmash_s2eps.set_s2_eps(eps)
        s2grad_numeric = (pmash_s2eps.objective - obj) / eps
        mlogger.debug(f"Gradient with respect to sigma^2: analytic {s2grad:.5f}, numeric {s2grad_numeric:.5f}")
        wgrad_string = ','.join([f"{x:.5f}" for x in wgrad])
        wgrad_numeric_string = ','.join([f"{x:.5f}" for x in wgrad_numeric])
        mlogger.debug(f"Gradient with respect to w_k:")
        mlogger.debug(f"analytic {wgrad_string}")
        mlogger.debug(f"numeric  {wgrad_numeric_string}")
        self.assertTrue(np.allclose(bgrad, bgrad_numeric, atol = 1e-4, rtol = 1e-8),
            msg = "Objective function gradient with respect to b does not match numeric results")
        self.assertTrue(np.allclose(wgrad[1:], wgrad_numeric[1:], atol = 1e-2, rtol = 1e-8), 
            msg = "Objective function gradient with respect to w_k does not match numeric results")
        self.assertTrue(np.abs((wgrad[0] - wgrad_numeric[0]) / wgrad[0]) < 1e-6,
            msg = "Objective function gradient with respect to w_0 does not match numeric results")
        self.assertAlmostEqual(s2grad, s2grad_numeric, places = 3,
            msg = "Objective function gradient with respect to sigma^2 does not match numeric results")
        return


if __name__ == '__main__':
    tester.main()


