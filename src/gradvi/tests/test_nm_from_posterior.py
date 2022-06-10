
"""
Test NormalMeansFromPosterior
test_method_comparsion
    All inversion methods yield similar result.

test_fssi_homogeneous_variance
    FSSI throws error if NM variance are unequal.

"""

import unittest
import numpy as np

from gradvi.normal_means import NormalMeans
from gradvi.normal_means import NormalMeansFromPosterior as NMFromPost
from gradvi.utils.logs import CustomLogger
from gradvi.utils import unittest_tester as tester
from gradvi.tests import toy_priors
from gradvi.tests import toy_data

mlogger = CustomLogger(__name__)

INVERT_METHODS = [
    'hybr',
    'trisection',
    'newton',
    'fssi-linear',
    'fssi-cubic'
    ]

class TestNMFromPosterior(unittest.TestCase):


    def test_inversion(self):
        # =======
        # Tolerances
        # =======
        atol = {k : 1e-6 for k in INVERT_METHODS}
        for k in INVERT_METHODS:
            if k == 'fssi-linear': atol[k] = 1e-2
            if k == 'fssi-cubic':  atol[k] = 1e-4
            if k == 'trisection':  atol[k] = 1e-1


        priors = toy_priors.get_all(k = 10, skbase = 10., sparsity = 0.3)
        for prior in priors:
            n      = 100
            s2     = 1.44
            dj     = np.ones(n) * n
            z, sj2 = toy_data.get_normal_means(prior, n = n, s2 = s2, dj = dj, seed = 100)
            # =================
            # the Normal Means model with z as response
            # =================
            nm1 = NormalMeans(z, prior, sj2, scale = s2, d = dj)
            b   = nm1.analytical_posterior_mean
            if b is not None:
                for method in INVERT_METHODS:
                    # =================
                    # Debug messages
                    # =================
                    info_msg  = f"Invert NM posterior mean operator M, {prior.prior_type} prior, {method} method"
                    err_msg_z = f"Response not equal to M^{1}(b), {prior.prior_type} prior, {method} method"
                    err_msg_b = f"M(M^{1}(b)) not equal to b, {prior.prior_type} prior, {method} method"
                    mlogger.info(info_msg)
                    # =================
                    # Invert
                    # =================
                    znew = NMFromPost(b, prior, sj2, scale = s2, d = dj, method = method).response
                    # =================
                    # the Normal Means model with znew as response
                    # =================
                    nm2  = NormalMeans(znew, prior, sj2, scale = s2, d = dj)
                    bnew = nm2.shrinkage_operator(jac = False)
                    # =================
                    # Check z = znew and b = bnew
                    # =================
                    #print ("z", np.max(np.abs(z - znew)))
                    #print ("b", np.max(np.abs(b - bnew)))
                    if not method == 'trisection':
                        np.testing.assert_allclose(z, znew, atol = atol[method], rtol = 1e-8, err_msg = err_msg_z)
                        np.testing.assert_allclose(b, bnew, atol = atol[method], rtol = 1e-8, err_msg = err_msg_b)
                    else:
                        zero = np.square(b - bnew) / b.shape[0]
                        np.testing.assert_allclose(zero, 0.0, atol = atol[method], rtol = 1e-8, err_msg = err_msg_b)
        return


    def operator_provider(self, nm, otype, jac = True):
        if otype == 'penalty':
            res = nm.penalty_operator(jac = jac)
        return res


    def test_derivatives(self):
        priors = toy_priors.get_all(k = 10, skbase = 10., sparsity = 0.3)
        otype  = 'penalty'
        for prior in priors:
            s2     = 1.44
            n      = 100
            dj     = np.ones(n) * n
            z, sj2 = toy_data.get_normal_means(prior, n = n, s2 = s2, dj = dj, seed = 100)
            # We are not testing the inversion.
            # Let's assume z is the posterior
            nm = NMFromPost(z, prior, sj2, scale = s2, d = dj, method = 'fssi-cubic', ngrid = 500)
            x, x_bd, x_wd, x_s2d = nm.penalty_operator(jac = True)
            # =================
            # Check the penalty operator value
            # =================
            info_msg  = f"f(b) should be equal to f(M(z)), NMFromPost {otype} operator, {prior.prior_type} prior"
            err_msg   = f"f(b) not equal to f(M(z)), NMFromPost {otype} operator, {prior.prior_type} prior"
            mlogger.info(info_msg)
            nm2  = NormalMeans(nm.response, prior, sj2, scale = s2, d = dj)
            lz   = nm2.penalty_operator(jac = False)
            np.testing.assert_allclose(lz, x, atol = 1e-4, rtol = 1e-8, err_msg = err_msg)
            # =================
            # Check the penalty operator derivatives
            # =================
            self._b_deriv(z, prior, sj2, s2, dj, x, x_bd, otype)
            self._w_deriv(z, prior, sj2, s2, dj, x, x_wd, otype)
            self._s2_deriv(z, prior, sj2, s2, dj, x, x_s2d, otype)
        return


    def _b_deriv(self, b, prior, sj2, s2, dj, x, x_bd, otype, eps = 1e-8, method = 'fssi-cubic'):
        info_msg  = f"df/db numerical differentiation, NMFromPost {otype} operator, {prior.prior_type} prior"
        err_msg = f"df/db not equal to numerical differentiation, NMFromPost {otype} operator, {prior.prior_type} prior"

        mlogger.info(info_msg)
        d2 = np.zeros_like(x_bd)
        for i in range(x_bd.shape[0]):
            b_eps     = b.copy()
            b_eps[i] += eps
            nm_eps    = NMFromPost(b_eps, prior, sj2, scale = s2, d = dj, method = method, ngrid = 500)
            x_eps     = self.operator_provider(nm_eps, otype, jac = False)
            d2[i]     = (np.sum(x_eps) - np.sum(x)) / eps
        np.testing.assert_allclose(x_bd, d2, atol = 1e-4, rtol = 1e-8, err_msg = err_msg)
        return


    def _w_deriv(self, b, prior, sj2, s2, dj, x, x_wd, otype, eps = 1e-8, method = 'fssi-cubic'):
        info_msg  = f"df/dw numerical differentiation, NMFromPost {otype} operator, {prior.prior_type} prior"
        err_msg = f"df/dw not equal to numerical differentiation, NMFromPost {otype} operator, {prior.prior_type} prior"

        mlogger.info(info_msg)
        for i in range(prior.k):
            wkeps     = prior.w.copy()
            wkeps[i] += eps
            prior_eps = toy_priors.get_from_same_class(prior, wkeps)
            nm_eps    = NMFromPost(b, prior_eps, sj2, scale = s2, d = dj, method = method, ngrid = 500)
            x_eps     = self.operator_provider(nm_eps, otype, jac = False)
            d1 = x_wd[:, i]
            d2 = (x_eps - x) / eps
            np.testing.assert_allclose(d1, d2, atol = 1e-4, rtol = 1e-8, err_msg = err_msg)
        return


    def _s2_deriv(self, b, prior, sj2, s2, dj, x, x_s2d, otype, eps = 1e-8, method = 'fssi-cubic'):
        info_msg  = f"df/ds2 numerical differentiation, NMFromPost {otype} operator, {prior.prior_type} prior"
        err_msg = f"df/ds2 not equal to numerical differentiation, NMFromPost {otype} operator, {prior.prior_type} prior"

        mlogger.info(info_msg)
        sj2_eps = (s2 + eps) / dj
        nm_eps  = NMFromPost(b, prior, sj2_eps, scale = s2 + eps, d = dj, method = method, ngrid = 500)
        x_eps   = self.operator_provider(nm_eps, otype, jac = False)
        d1 = x_s2d / dj
        d2 = (x_eps - x) / eps
        np.testing.assert_allclose(d1, d2, atol = 1e-6, rtol = 1e-8, err_msg = err_msg)
        return



if __name__ == '__main__':
    tester.main()
