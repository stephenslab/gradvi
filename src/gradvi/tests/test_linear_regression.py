
import unittest
import numpy as np

from gradvi.inference import LinearRegression
from gradvi.utils import unittest_tester as tester
from gradvi.utils.logs import MyLogger
from gradvi.tests import toy_priors
from gradvi.tests import toy_data

mlogger = MyLogger(__name__)

class TestLinearRegression(unittest.TestCase):


    def test_both_equal(self):
        x, y, b, s2 = toy_data.get_linear_model(pve = 0.75, standardize = True)
        prior = toy_priors.get_ash_scaled(k = 10, sparsity = None)
        # ================
        # Reparametrized objective
        # ================
        gv1 = LinearRegression(
                obj = "reparametrize",
                debug = False, 
                display_progress = False)
        gv1.fit(x, y, prior)
        # ================
        # Direct objective 
        # ================
        gv2 = LinearRegression(
                obj = "direct",
                debug = False,
                display_progress = False, 
                invert_method = "fssi-cubic")
        gv2.fit(x, y, prior)

        # ================
        # Coefficients are equal 
        # ================
        info_msg = f"Linear regression coefficients using reparametrized and direct objectives should be equal"
        err_msg  = f"Linear regression coefficients using reparametrized and direct objectives are different, {prior.prior_type} prior"
        mlogger.info(info_msg)
        np.testing.assert_allclose(gv1.coef, gv2.coef, atol = 0.1, rtol = 1e-8, err_msg = err_msg)
        
        # ================
        # ELBO
        # ================
        info_msg = f"At the optimum, the objective function should be equal to -ELBO"
        err_msg  = f"At the optimum, the objective function is not equal to -ELBO, {prior.prior_type} prior"
        mlogger.info(info_msg)
        dj = np.sum(np.square(x), axis = 0)
        hmin = gv2.fun
        hmin += 0.5 * np.sum(np.log(dj))
        elbo  = gv2.get_elbo(gv2.coef, gv2.residual_var, gv2.prior)
        #mlogger.info(f"Objective function: {-hmin}")
        #mlogger.info(f"ELBOs: {elbo}")
        np.testing.assert_almost_equal(hmin, elbo, decimal = 5, err_msg = err_msg)
        return


if __name__ == '__main__':
    tester.main()
