import unittest
import numpy as np
import pickle
import os

from gradvi.models import LinearModel
from gradvi.normal_means import NormalMeans
from gradvi.priors import Ash

from gradvi.utils import unittest_tester as tester
from gradvi.utils.logs import MyLogger

mlogger = MyLogger(__name__)


def get_ash_priors():
    sk = np.array([0.1, 0.5, 0.9])
    wk = np.array([0.5, 0.25, 0.25])
    prior1 = Ash(sk, wk = wk, scaled = False)
    prior2 = Ash(sk, wk = wk, scaled = True)
    return prior1, prior2


def get_lm_data():
    n = 5
    p = 6
    std = 0.9
    b  = np.array([1.21, 2.32, 0.01, 0.03, 0.11, 3.12])
    y  = np.array([3.5, 4.5, 1.2, 6.5, 2.8])
    XT = np.array([8.79, 6.11,-9.15, 9.57,-3.49, 9.84,
                   9.93, 6.91,-7.93, 1.64, 4.02, 0.15, 
                   9.83, 5.04, 4.86, 8.83, 9.80,-8.99,
                   5.45,-0.27, 4.85, 0.74,10.00,-6.02,
                   3.16, 7.98, 3.01, 5.80, 4.27,-5.31])
    X = XT.reshape(p, n).T
    return X, y, b, np.square(std)


def load_mrashpen_result():
    filename = "mrashpen_res.pkl"
    curdir   = os.path.dirname(os.path.realpath(__file__)) 
    filepath = os.path.join(curdir, filename)
    with open(filepath, "rb") as f:
        res = pickle.load(f)
    return res


class TestOldPLRAsh(unittest.TestCase):

    def test_all(self):
        X, y, b, s2 = get_lm_data()
        ash, ash_sc = get_ash_priors()
        mrashpen_res = load_mrashpen_result()
        dj = np.sum(np.square(X), axis = 0)
        sj2 = s2 / dj
        for prior in [ash, ash_sc]:
            # =========================
            # Normal Means Model
            # =========================
            mlogger.info (f"Normal Means model for {prior.prior_type} prior should match mrashpen results")
            nm = NormalMeans.create(b, prior, sj2, scale = s2, d = dj)
            mb, mb_bgrad, mb_wgrad, mb_s2grad = nm.shrinkage_operator()
            lj, lj_bgrad, lj_wgrad, lj_s2grad = nm.penalty_operator()
            res = {
                'logML': nm.logML,
                'logML_deriv': nm.logML_deriv,
                'logML_wderiv': nm.logML_wderiv,
                'logML_s2deriv': nm.logML_s2deriv,
                'logML_deriv2': nm.logML_deriv2,
                'logML_deriv_wderiv': nm.logML_deriv_wderiv,
                'logML_deriv_s2deriv': nm.logML_deriv_s2deriv,
                'shrinkage_mb': mb,
                'shrinkage_mb_bgrad': mb_bgrad,
                'shrinkage_mb_wgrad': mb_wgrad,
                'shrinkage_mb_s2grad': mb_s2grad,
                'penalty_lj': lj,
                'penalty_lj_bgrad': lj_bgrad,
                'penalty_lj_wgrad': np.sum(lj_wgrad, axis = 0),
                'penalty_lj_s2grad': lj_s2grad,
                }
            if prior.prior_type == 'ash':
                oldres = mrashpen_res['nmash']
            elif prior.prior_type == 'ash_scaled':
                oldres = mrashpen_res['nmash_scaled']
            self.compare_dict(res, oldres)

            # =========================
            # Linear Model
            # =========================
            mlogger.info (f"Reparametrized objective function for linear model with {prior.prior_type} prior should match mrashpen results")
            lm = LinearModel(X, y, b, s2, prior, objtype = "reparametrize")
            res = {
                'objective': lm.objective,
                'bgrad': lm.bgrad,
                'wgrad': lm.wgrad,
                's2grad': lm.s2grad,
                }
            if prior.prior_type == 'ash':
                oldres = mrashpen_res['plrash']
            elif prior.prior_type == 'ash_scaled':
                oldres = mrashpen_res['plrash_scaled']
            self.compare_dict(res, oldres)

        return


    def compare_dict(self, x1, x2):
        for key, val in x1.items():
            if key in x2.keys():
                info_msg = f"{key}"
                err_msg = f"{key} did not match mrashpen results"
                mlogger.info(info_msg)
                oldval = x2[key]
                np.testing.assert_allclose(val, oldval, atol = 1e-8, rtol = 1e-8, err_msg = err_msg)
        return


if __name__ == '__main__':
    tester.main()
