import unittest
import numpy as np
import os

from gradvi.tests import toy_data
from gradvi.tests import toy_priors
from gradvi.inference import Trendfiltering
from gradvi.utils.logs import CustomLogger
from gradvi.utils import project

mlogger = CustomLogger(__name__)

class TestFSSI(unittest.TestCase):
    
    #def test_zero_derivative(self):
    #    # I know the derivative of shrinkage function is zero at certain points
    #    # when using the following data and method of trendfiltering.
    #    seed = 105
    #    n = 256
    #    x = np.linspace(0, 1, n)
    #    knots = np.linspace(0, 1, 12)[1:-1]
    #    data  = toy_data.changepoint_from_bspline(x, knots, 0.5, degree = 1,
    #                signal = "normal", seed = seed, include_intercept = False)
    #    prior = toy_priors.get_ash_scaled(k = 20, sparsity = 0.9, skbase = 20)
    #    gv = Trendfiltering(maxiter = 200, obj = 'direct', scale_tfbasis = True)

    #    root_logger_name = project.get_name()
    #    with self.assertLogs(root_logger_name, level = 'WARNING') as cm:
    #        gv.fit(data.y, data.degree, prior)
    #        self.assertTrue(any(["Derivative of f(x) returns zero values. Forcing fssi-linear." in x for x in cm.output]))
    #    return


    def test_strictly_increasing_ygrid(self):
        # Use the following data and method for reproducing and checking the problem
        seed = 1005
        n = 512
        x = np.linspace(0, 1, n)
        knots = np.linspace(0, 1, 12)[1:-1]
        data  = toy_data.changepoint_from_bspline(x, knots, 0.5, degree = 1,
                    signal = "normal", seed = seed, include_intercept = False)
        prior = toy_priors.get_ash_scaled(k = 20, sparsity = 0.9, skbase = 20)
        gv = Trendfiltering(maxiter = 200, obj = 'direct', scale_tfbasis = True)

        root_logger_name = project.get_name()
        with self.assertLogs(root_logger_name, level = 'WARNING') as cm:
            gv.fit(data.y, data.degree, prior)
            self.assertTrue(any(["ygrid is not strictly increasing. Removing grid points." in x for x in cm.output]))
        return
