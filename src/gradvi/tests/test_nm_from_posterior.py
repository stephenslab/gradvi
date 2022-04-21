

"""
Test NormalMeansFromPosterior
test_method_comparsion
    All inversion methods yield similar result.

test_fssi_homogeneous_variance
    FSSI throws error if NM variance are unequal.

test_expectations
    Results should match some of our expectations:
        .. M(x) = b where x is obtained after inverting b
        .. For a NM model z ~ N(a, sj2), calculate posterior mean.
            Then, invert of the posterior mean equals z.
"""

import unittest
import numpy as np

from gradvi.priors.normal_means import NormalMeans
from gradvi.priors.normal_means import NormalMeansFromPosterior as NMFromPost
from gradvi.utils.logs import MyLogger
from gradvi.utils import unittest_tester as tester
from gradvi.tests import toy_priors

mlogger = MyLogger(__name__)

INVERT_METHODS = [
    'hybr',
    'newton',
    'fssi-linear',
    'fssi-cubic'
    ]

class TestNMFromPosterior(unittest.TestCase):


    def sample_normal_means(self, mean, var):
        p = mean.shape[0]
        if not isinstance(var, np.ndarray):
            var = np.repeat(var, p)
        y = np.random.multivariate_normal(mean, np.diag(var))
        return y


    def get_nm_data(self, prior, p = 500):
        np.random.seed(100)
        s2  = 1.2**2
        #dj = np.square(np.random.normal(1, 0.5, size = n)) * n
        dj  = np.ones(p) * p
        b   = prior.sample(p, seed = 200, scale = s2)
        sj2 = s2 / dj
        z   = self.sample_normal_means(b, sj2)
        return z, sj2, s2, dj


    def test_inversion(self):
        # =======
        # Tolerances
        # =======
        atol = {k : 1e-8 for k in INVERT_METHODS}
        for k in INVERT_METHODS:
            if k.startswith('fssi'): atol[k] = 1e-4

        priors = toy_priors.get_all(k = 10, scale = 10., sparsity = 0.3)
        for prior in priors:
            z, sj2, s2, dj = self.get_nm_data(prior)
            # =================
            # the Normal Means model with z as response
            # =================
            nm1 = NormalMeans.create(z, prior, sj2, scale = s2, d = dj)
            b   = nm1.analytical_posterior_mean
            if b is not None:
                for method in INVERT_METHODS:
                    # =================
                    # Debug messages
                    # =================
                    info_msg = f"Checking inversion of posterior mean, {prior.prior_type} prior, {method} method"
                    err_msg_z = f"Inverted posterior mean does not match response, {prior.prior_type} prior, {method} method"
                    err_msg_b = f"Posterior mean of inverse b does not match b, {prior.prior_type} prior, {method} method"
                    mlogger.info(info_msg)
                    # =================
                    # Invert
                    # =================
                    znew = NMFromPost(b, prior, sj2, scale = s2, d = dj, method = method, ngrid = 500).response
                    # =================
                    # the Normal Means model with znew as response
                    # =================
                    nm2  = NormalMeans.create(znew, prior, sj2, scale = s2, d = dj)
                    bnew = nm2.shrinkage_operator(jac = False)
                    # =================
                    # Check z = znew and b = bnew
                    # =================
                    #print ("z", np.max(np.abs(z - znew)))
                    #print ("b", np.max(np.abs(b - bnew)))
                    self.assertTrue(np.allclose(z, znew, atol = atol[method], rtol = 1e-8), msg = err_msg_z)
                    self.assertTrue(np.allclose(b, bnew, atol = atol[method], rtol = 1e-8), msg = err_msg_b)
        return


    def operator_provider(self, nm, otype, jac = True):
        if otype == 'penalty':
            res = nm.penalty_operator(jac = jac)
        return res


    def test_derivatives(self):
        priors = toy_priors.get_all(k = 10, scale = 10., sparsity = 0.3)
        otype  = 'penalty'
        for prior in priors:
            z, sj2, s2, dj = self.get_nm_data(prior)
            # We are not testing the inversion.
            # Let's assume z is the posterior
            nm = NMFromPost(z, prior, sj2, scale = s2, d = dj, method = 'fssi-cubic', ngrid = 500)
            x, x_bd, x_wd, x_s2d = nm.penalty_operator(jac = True)
            # =================
            # Check the penalty operator value
            # =================
            info_msg  = f"Checking value of NMFromPost {otype} operator for {prior.prior_type} prior"
            error_msg = f"NMFromPost {otype} does not match the {otype} from NormalMeans(response) for {prior.prior_type} prior"
            mlogger.info(info_msg)
            nm2  = NormalMeans.create(nm.response, prior, sj2, scale = s2, d = dj)
            lz   = nm2.penalty_operator(jac = False)
            self.assertTrue(np.allclose(lz, x, atol = 1e-4, rtol = 1e-8), msg = error_msg)
            # =================
            # Check the penalty operator derivatives
            # =================
            self._b_deriv(z, prior, sj2, s2, dj, x, x_bd, otype)
            self._w_deriv(z, prior, sj2, s2, dj, x, x_wd, otype)
            self._s2_deriv(z, prior, sj2, s2, dj, x, x_s2d, otype)
        return


    def _b_deriv(self, b, prior, sj2, s2, dj, x, x_bd, otype, eps = 1e-8, method = 'fssi-cubic'):
        info_msg  = f"Checking derivatives of NMFromPost {otype} operator for {prior.prior_type} prior"
        error_msg = f"NMFromPost {otype} operator derivative does not match numeric results for {prior.prior_type} prior"

        mlogger.info(info_msg)
        d2 = np.zeros_like(x_bd)
        for i in range(x_bd.shape[0]):
            b_eps     = b.copy()
            b_eps[i] += eps
            nm_eps    = NMFromPost(b_eps, prior, sj2, scale = s2, d = dj, method = method, ngrid = 500)
            x_eps     = self.operator_provider(nm_eps, otype, jac = False)
            d2[i]     = (np.sum(x_eps) - np.sum(x)) / eps
        self.assertTrue(np.allclose(x_bd, d2, atol = 1e-4, rtol = 1e-8), msg = error_msg)
        return


    def _w_deriv(self, b, prior, sj2, s2, dj, x, x_wd, otype, eps = 1e-8, method = 'fssi-cubic'):
        info_msg  = f"Checking wk derivatives of NMFromPost {otype} operator for {prior.prior_type} prior"
        error_msg = f"NMFromPost {otype} operator wk derivative does not match numeric results for {prior.prior_type} prior"

        mlogger.info(info_msg)
        for i in range(prior.k):
            wkeps     = prior.w.copy()
            wkeps[i] += eps
            prior_eps = toy_priors.get_from_same_class(prior, wkeps)
            nm_eps    = NMFromPost(b, prior_eps, sj2, scale = s2, d = dj, method = method, ngrid = 500)
            x_eps     = self.operator_provider(nm_eps, otype, jac = False)
            d1 = x_wd[:, i]
            d2 = (x_eps - x) / eps
            self.assertTrue(np.allclose(d1, d2, atol = 1e-4, rtol = 1e-8), msg = error_msg)
        return


    def _s2_deriv(self, b, prior, sj2, s2, dj, x, x_s2d, otype, eps = 1e-8, method = 'fssi-cubic'):
        info_msg  = f"Checking s2 derivatives of NMFromPost {otype} operator for {prior.prior_type} prior"
        error_msg = f"NMFromPost {otype} operator s2 derivative does not match numeric results for {prior.prior_type} prior"

        mlogger.info(info_msg)
        sj2_eps = (s2 + eps) / dj
        nm_eps  = NMFromPost(b, prior, sj2_eps, scale = s2 + eps, d = dj, method = method, ngrid = 500)
        x_eps   = self.operator_provider(nm_eps, otype, jac = False)
        d1 = x_s2d / dj
        d2 = (x_eps - x) / eps
        self.assertTrue(np.allclose(d1, d2, atol = 1e-6, rtol = 1e-8), msg = error_msg)
        return



if __name__ == '__main__':
    tester.main()
