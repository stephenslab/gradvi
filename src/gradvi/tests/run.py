
from gradvi.utils.unittest_tester import UnittestTester

from gradvi.tests import TestNormalMeansPy, \
                         TestLinearModel, \
                         TestNMOperator, \
                         TestOldPLRAsh, \
                         TestNMFromPosterior, \
                         TestLinearRegression

def run_unittests():
    test_classes = [
            TestNormalMeansPy,
            TestNMOperator,
            TestLinearModel,
            TestOldPLRAsh,
            TestNMFromPosterior,
            TestLinearRegression,
            ]
    tester = UnittestTester(test_classes)
    tester.execute()
    del tester
    # =========================
    # if you want report for each class separately,
    # =========================
    #for mtest in test_classes:
    #    tester = UnittestTester(mtest)
    #    del tester
    return
